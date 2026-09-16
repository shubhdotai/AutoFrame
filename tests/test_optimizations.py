"""Regression tests for the streaming/batched pipeline and the accuracy fixes."""
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pytest

from autoclip.cropping import TrackCropper, crop_face, smooth_track_geometry
from autoclip.gating import contested_frames, select_tracks_for_asd, speech_mask
from autoclip.inference import resolve_durations
from autoclip.report import build_report
from autoclip.shortform import (
    _smooth_scores, build_reframe_plan, render_video, serialize_plan,
)
from autoclip.tracking import track_faces

CHECKPOINT = Path(__file__).resolve().parents[1] / "models/pretrain_AVA.model"
needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg"
)


def box(frame, x, y=10, w=40, h=50, conf=0.9):
    return {"frame": frame, "bbox": [x, y, x + w, y + h], "conf": conf}


# ---------------------------------------------------------------------------
# Encoder: windowing must not change results
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
@pytest.mark.parametrize("window,batch", [(100, 1), (100, 8), (40, 4), (20, 16)])
def test_streaming_encoder_matches_one_unbroken_pass(window, batch):
    """
    The whole point of the margin logic: chunking the encoders must be exact.

    Without margins, the ~9-frame temporal receptive field means every window
    boundary corrupts the frames around it.
    """
    import torch
    from autoclip.inference import EmbeddingEngine, load_model

    model = load_model(str(CHECKPOINT), device="cpu")
    rng = np.random.default_rng(0)
    n = 307
    frames = (rng.random((n, 112, 112)) * 255).astype(np.uint8)
    mfcc = rng.standard_normal((n * 4, 13)).astype(np.float32)

    with torch.no_grad():
        want_v = model.model.forward_visual_frontend(
            torch.from_numpy(frames.astype(np.float32))[None]).numpy()[0]
        want_a = model.model.forward_audio_frontend(
            torch.from_numpy(mfcc)[None]).numpy()[0]

    engine = EmbeddingEngine(model, device="cpu", window=window, batch=batch)
    engine.open_track("t", n, mfcc)
    for frame in frames:
        engine.push("t", frame)
    engine.close_track("t")
    got_a, got_v = engine.finish()["t"]

    np.testing.assert_allclose(got_v, want_v, atol=1e-4)
    np.testing.assert_allclose(got_a, want_a, atol=1e-4)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
def test_interleaved_tracks_do_not_contaminate_each_other():
    from autoclip.inference import EmbeddingEngine, load_model

    model = load_model(str(CHECKPOINT), device="cpu")
    rng = np.random.default_rng(7)
    spec = {"a": 213, "b": 47, "c": 131}
    data = {
        k: ((rng.random((n, 112, 112)) * 255).astype(np.uint8),
            rng.standard_normal((n * 4, 13)).astype(np.float32))
        for k, n in spec.items()
    }

    def solo(key):
        frames, mfcc = data[key]
        engine = EmbeddingEngine(model, device="cpu", window=100, batch=1)
        engine.open_track(key, len(frames), mfcc)
        for frame in frames:
            engine.push(key, frame)
        engine.close_track(key)
        return engine.finish()[key]

    want = {k: solo(k) for k in spec}

    engine = EmbeddingEngine(model, device="cpu", window=100, batch=8)
    for key, n in spec.items():
        engine.open_track(key, n, data[key][1])
    for i in range(max(spec.values())):
        for key, n in spec.items():
            if i < n:
                engine.push(key, data[key][0][i])
    for key in spec:
        engine.close_track(key)
    got = engine.finish()

    for key in spec:
        np.testing.assert_allclose(got[key][0], want[key][0], atol=1e-5)
        np.testing.assert_allclose(got[key][1], want[key][1], atol=1e-5)


# ---------------------------------------------------------------------------
# Detector: durations must never become padding
# ---------------------------------------------------------------------------

def test_durations_are_clamped_to_the_track_length():
    # A 31-frame track cannot supply a 6 s (150 frame) window; asking for one
    # used to zero-fill 79% of the sequence fed to a bidirectional GRU.
    assert resolve_durations((2, 4, 6), 31) == [31]
    assert resolve_durations((2, 4, 6), 500) == [50, 100, 150]
    assert resolve_durations((2, 4, 6), 120) == [50, 100, 120]
    assert resolve_durations((2,), 0) == []


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
def test_short_track_scores_are_independent_of_requested_durations():
    """With clamping, durations longer than the track collapse to one real pass."""
    import torch
    from autoclip.inference import load_model, score_detector

    model = load_model(str(CHECKPOINT), device="cpu")
    torch.manual_seed(0)
    a, v = torch.randn(31, 128), torch.randn(31, 128)
    np.testing.assert_allclose(
        score_detector(model, a, v, duration_set=(2, 4, 6)),
        score_detector(model, a, v, duration_set=(6,)),
        atol=1e-5,
    )


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
def test_every_frame_is_scored_in_a_full_window():
    from autoclip.inference import load_model, score_detector
    import torch

    model = load_model(str(CHECKPOINT), device="cpu")
    torch.manual_seed(1)
    n = 137
    scores = score_detector(model, torch.randn(n, 128), torch.randn(n, 128),
                            duration_set=(2,))
    assert scores.shape == (n,)
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def dense_track(start, end, x):
    frames = np.arange(start, end)
    return {"frame": frames, "bbox": np.tile([x, 10, x + 40, 60], (len(frames), 1))}


def test_single_track_scenes_are_not_scored():
    tracks = [dense_track(0, 30, 100)]
    selected, reasons = select_tracks_for_asd(tracks, 30, 1920, 594)
    assert selected == []
    assert reasons[0] == "uncontested"


def test_two_far_apart_faces_are_contested():
    tracks = [dense_track(0, 30, 10), dense_track(0, 30, 1800)]
    assert contested_frames(tracks, 30, 1920, 594).all()
    selected, _ = select_tracks_for_asd(tracks, 30, 1920, 594)
    assert selected == [0, 1]


def test_two_adjacent_faces_yielding_the_same_crop_are_not_contested():
    # Both centres map to the same clamped crop rectangle, so no score can
    # move the output.
    tracks = [dense_track(0, 30, 900), dense_track(0, 30, 903)]
    assert not contested_frames(tracks, 30, 1920, 594).any()
    assert select_tracks_for_asd(tracks, 30, 1920, 594)[0] == []


def test_overlap_pulls_in_the_whole_track():
    # Track 0 is alone for most of its life but contested at the end; it still
    # needs a score, because the contested frames depend on it.
    tracks = [dense_track(0, 40, 10), dense_track(30, 40, 1800)]
    selected, _ = select_tracks_for_asd(tracks, 40, 1920, 594)
    assert selected == [0, 1]


def test_silent_tracks_are_skipped():
    tracks = [dense_track(0, 30, 10), dense_track(0, 30, 1800)]
    silence = np.zeros(30, dtype=bool)
    selected, reasons = select_tracks_for_asd(tracks, 30, 1920, 594, speech=silence)
    assert selected == []
    assert set(reasons.values()) == {"silent"}


def test_score_all_tracks_overrides_gating():
    tracks = [dense_track(0, 30, 100)]
    selected, _ = select_tracks_for_asd(tracks, 30, 1920, 594, require_contested=False)
    assert selected == [0]


def test_speech_mask_marks_silence_and_keeps_quiet_speech():
    sr, fps = 16000, 25
    loud = (np.ones(sr) * 0.2).astype(np.float32)
    quiet = (np.ones(sr) * 0.01).astype(np.float32)   # about -40 dBFS
    silent = np.zeros(sr, dtype=np.float32)
    assert speech_mask(loud, sr, fps, fps=fps).all()
    assert speech_mask(quiet, sr, fps, fps=fps).all()
    assert not speech_mask(silent, sr, fps, fps=fps).any()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def test_crossing_faces_keep_their_identity():
    """
    Two faces swap sides. First-match IoU hands one track the other's box;
    best-match assignment does not.
    """
    detections = []
    for f in range(40):
        left = 100 + f * 10
        right = 490 - f * 10
        # Deliberately list the right-hand face first, which is what tempted
        # the greedy matcher into the wrong pairing.
        detections.append([box(f, right), box(f, left)])
    tracks = track_faces(detections, min_track=2, iou_threshold=0.3)
    assert len(tracks) == 2
    for track in tracks:
        xs = np.asarray(track["bbox"])[:, 0]
        # Each track must move monotonically: no jumping to the other face.
        steps = np.diff(xs)
        assert np.all(steps > 0) or np.all(steps < 0)


def test_low_confidence_detections_do_not_become_tracks():
    detections = [[box(f, 100, conf=0.2)] for f in range(30)]
    assert track_faces(detections, min_track=2, min_confidence=0.5) == []
    assert len(track_faces(detections, min_track=2, min_confidence=0.1)) == 1


def test_tiny_faces_are_filtered():
    detections = [[box(f, 100, w=6, h=6)] for f in range(30)]
    assert track_faces(detections, min_track=2, min_face_size=20) == []


def test_tracking_does_not_mutate_its_input():
    detections = [[box(f, 20), box(f, 22)] for f in range(30)]
    track_faces(detections, min_track=2)
    assert all(len(frame) == 2 for frame in detections)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def test_short_track_smoothing_does_not_drift_toward_the_origin():
    """
    scipy.signal.medfilt zero-pads, which dragged the first and last few
    centres of every track toward 0 and shrank the box with them.
    """
    bboxes = np.tile([500.0, 400.0, 560.0, 470.0], (11, 1))
    cx, cy, half = smooth_track_geometry(bboxes, kernel=13)
    assert np.allclose(cx, 530.0)
    assert np.allclose(cy, 435.0)
    assert np.allclose(half, 35.0)


def test_crop_face_pads_instead_of_shifting_at_the_frame_edge():
    frame = np.full((200, 200, 3), 7, dtype=np.uint8)
    crop = crop_face(frame, 5, 5, 40, out_size=64)
    assert crop.shape == (64, 64, 3)
    assert crop.max() >= 100          # grey padding is present
    assert crop.min() == 7            # real pixels survive


def test_crop_face_grayscale_and_center_half():
    frame = np.full((300, 300, 3), 30, dtype=np.uint8)
    crop = crop_face(frame, 150, 150, 50, out_size=112,
                     center_half=True, grayscale=True)
    assert crop.shape == (112, 112)
    assert crop.dtype == np.uint8


@needs_ffmpeg
def test_cropper_visits_every_track_frame_in_one_pass(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=25:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)

    tracks = [dense_track(0, 20, 60), dense_track(10, 40, 200)]
    seen = {0: [], 1: []}
    opened, closed = [], []
    TrackCropper(tracks, out_size=32).run(
        source,
        on_open=opened.append,
        on_frame=lambda i, local, crop: seen[i].append(local),
        on_close=closed.append,
        total_frames=50,
    )
    assert seen[0] == list(range(20))
    assert seen[1] == list(range(30))
    assert sorted(opened) == [0, 1] and sorted(closed) == [0, 1]


# ---------------------------------------------------------------------------
# Reframe plan
# ---------------------------------------------------------------------------

def test_unscored_tracks_are_followed_but_never_called_speaking():
    track = dense_track(0, 30, 800)
    plan = build_reframe_plan([track], [np.full(30, np.nan, np.float32)],
                              30, 1920, 1080)
    assert "speaker" not in plan["source"]
    assert (plan["track"] == 0).all()          # still framed as the subject
    assert (plan["crop_x1"] > 0).all()


def test_vertical_anchor_tracks_the_face_and_stays_in_bounds():
    high = dense_track(0, 20, 800)
    high["bbox"] = np.tile([800.0, 0.0, 860.0, 60.0], (20, 1))
    low = dense_track(0, 20, 800)
    low["bbox"] = np.tile([800.0, 1000.0, 860.0, 1060.0], (20, 1))
    nan = np.full(20, np.nan, np.float32)
    top = build_reframe_plan([high], [nan], 20, 1920, 1080)
    bottom = build_reframe_plan([low], [nan], 20, 1920, 1080)
    assert top["crop_y1"][0] < bottom["crop_y1"][0]
    for plan in (top, bottom):
        assert (plan["crop_y1"] >= 0).all()
        assert (plan["crop_y1"] + plan["crop_height"] <= 1080).all()


def test_person_boxes_frame_the_shot_when_no_face_is_visible():
    persons = [[{"frame": f, "bbox": [1500, 200, 1700, 900], "conf": 0.9}]
               for f in range(20)]
    plan = build_reframe_plan([], [], 20, 1920, 1080, persons_per_frame=persons)
    assert plan["source"][0] == "person"
    centred = build_reframe_plan([], [], 20, 1920, 1080)
    assert plan["crop_x1"][0] > centred["crop_x1"][0]


def test_speaker_margin_blocks_a_switch_on_a_hairline_lead():
    """The margin gates *switching*, not staying: a near-tie must not steal the camera."""
    tracks = [dense_track(0, 60, 100), dense_track(0, 60, 1700)]
    a = np.full(60, 0.90, np.float32)
    b = np.full(60, 0.95, np.float32)
    a[:10], b[:10] = 1.0, -5.0        # let track 0 take the camera first

    without = build_reframe_plan(tracks, [a, b], 60, 1920, 1080,
                                 score_smooth_window=1, speaker_margin=0.0)
    with_margin = build_reframe_plan(tracks, [a, b], 60, 1920, 1080,
                                     score_smooth_window=1, speaker_margin=0.5)
    assert without["track"][-1] == 1          # 0.05 ahead is enough to switch
    assert (with_margin["track"] == 0).all()  # ...but not once a margin is required


def test_follow_mode_eases_toward_a_moving_subject():
    frames = np.arange(60)
    moving = {"frame": frames,
              "bbox": np.stack([100 + frames * 20, np.full(60, 10.0),
                                140 + frames * 20, np.full(60, 60.0)], axis=1)}
    nan = np.full(60, np.nan, np.float32)
    locked = build_reframe_plan([moving], [nan], 60, 1920, 1080, motion="lock")
    follow = build_reframe_plan([moving], [nan], 60, 1920, 1080, motion="follow")
    assert len(set(locked["crop_x1"].tolist())) == 1
    assert follow["crop_x1"][-1] > follow["crop_x1"][0]


def test_smooth_scores_matches_the_reference_loop():
    rng = np.random.default_rng(3)
    arr = rng.standard_normal(200).astype(np.float32)
    window = 11
    half = window // 2
    want = np.array([arr[max(i - half, 0):min(i + half + 1, len(arr))].mean()
                     for i in range(len(arr))], dtype=np.float32)
    np.testing.assert_allclose(_smooth_scores(arr, window), want, atol=1e-5)


def test_plan_serialization_round_trips_both_axes():
    plan = build_reframe_plan([], [], 30, 1920, 1080)
    payload = json.loads(json.dumps(serialize_plan(plan)))
    assert len(payload["per_frame"]["crop_x1"]) == 30
    assert len(payload["per_frame"]["crop_y1"]) == 30


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_report_distinguishes_unscored_from_silent():
    tracks = [dense_track(0, 20, 100), dense_track(0, 20, 800)]
    scores = [np.full(20, -1.0, np.float32), np.full(20, np.nan, np.float32)]
    report = build_report("v.mp4", tracks, scores, frame_width=1920)
    silent, unscored = report["tracks"]
    assert silent["scored"] and silent["speaking_frames"] == 0
    assert not unscored["scored"]
    assert unscored["avg_score"] is None
    assert unscored["speaking_segments"] == []


def test_partially_scored_track_stays_json_serializable():
    """
    A track can be scored for most of its length and NaN-padded where the
    audio ran short. Summary stats must ignore the padding: json.dump writes
    a bare NaN otherwise, which is not valid JSON.
    """
    track = dense_track(0, 20, 100)
    score = np.concatenate([np.ones(15, np.float32), np.full(5, np.nan, np.float32)])
    report = build_report("v.mp4", [track], [score])
    item = report["tracks"][0]
    assert item["avg_score"] == pytest.approx(1.0)
    assert item["max_score"] == pytest.approx(1.0)
    json.dumps(report, allow_nan=False)


def test_smoothing_does_not_let_nan_bleed_across_a_window():
    got = _smooth_scores(np.array([2.0, np.nan, 4.0], np.float32), 3)
    np.testing.assert_allclose(got, [2.0, 3.0, 4.0])
    assert np.isnan(_smooth_scores(np.full(6, np.nan, np.float32), 5)).all()


def test_report_segments_match_the_threshold():
    track = dense_track(0, 20, 100)
    score = np.full(20, -1.0, np.float32)
    score[5:10] = 2.0
    report = build_report("v.mp4", [track], [score], smoothing_window=1)
    assert report["tracks"][0]["speaking_segments"] == [
        {"start_frame": 5, "end_frame": 9, "start_time_s": 0.2, "end_time_s": 0.4}
    ]


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_preprocess_emits_video_and_audio_from_one_call(tmp_path):
    from autoclip.media import preprocess_video, probe_video

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=50:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)

    video, audio = preprocess_video(source, tmp_path / "work", fps=25)
    info = probe_video(video)
    assert info["fps"] == pytest.approx(25, abs=0.1)
    assert info["n_frames"] == pytest.approx(50, abs=2)

    from scipy.io import wavfile
    rate, samples = wavfile.read(audio)
    assert rate == 16000 and samples.ndim == 1


@needs_ffmpeg
def test_preprocess_rejects_silent_input_before_encoding(tmp_path):
    from autoclip.media import preprocess_video

    source = tmp_path / "mute.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=160x120:rate=25:duration=1",
         "-c:v", "libx264", str(source)], check=True)
    with pytest.raises(RuntimeError, match="must.*contain an audio stream"):
        preprocess_video(source, tmp_path / "work")


@needs_ffmpeg
def test_render_upscales_to_a_requested_height(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)

    plan = build_reframe_plan([], [], 50, 640, 360)
    out = tmp_path / "vertical.mp4"
    render_video(source, source, plan, out, output_height=1920, progress=False)
    streams = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)]))
    video = next(s for s in streams["streams"] if s["codec_type"] == "video")
    assert video["height"] == 1920
    assert video["width"] * 16 == video["height"] * 9


@needs_ffmpeg
def test_frame_sink_terminates_when_audio_outlasts_the_video(tmp_path):
    """
    `-af apad` generates silence forever. Paired with `-t` that is exactly what
    short audio needs; unpaired, FFmpeg never sees end of stream and the encode
    hangs. Both paths must finish.
    """
    from autoclip.media import FrameSink, probe_video

    audio = tmp_path / "long.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=16000:duration=4",
         str(audio)], check=True)
    frame = np.zeros((64, 32, 3), dtype=np.uint8)

    # No declared duration: must stop at the shorter stream.
    unbounded = tmp_path / "unbounded.mp4"
    sink = FrameSink(unbounded, 32, 64, 25, audio_path=audio, quality="analysis")
    for _ in range(25):
        sink.write(frame)
    sink.close()
    assert probe_video(unbounded)["duration"] == pytest.approx(1.0, abs=0.2)

    # Declared duration: audio is padded out to match the video.
    bounded = tmp_path / "bounded.mp4"
    sink = FrameSink(bounded, 32, 64, 25, audio_path=audio,
                     audio_duration=1.0, quality="analysis")
    for _ in range(25):
        sink.write(frame)
    sink.close()
    assert probe_video(bounded)["duration"] == pytest.approx(1.0, abs=0.2)


@needs_ffmpeg
def test_debug_renderers_complete(tmp_path):
    """The diagnostic renderers run the same sink; they must terminate too."""
    from autoclip.visualize import (
        render_active_speaker_video, render_raw_detections_video,
    )

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=25:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=16000:duration=3",
         str(audio)], check=True)

    detections = [[box(f, 40)] for f in range(25)]
    raw = render_raw_detections_video(str(source), str(audio), detections,
                                      str(tmp_path / "raw.mp4"))
    assert Path(raw).stat().st_size > 0

    track = dense_track(0, 25, 40)
    speaker = render_active_speaker_video(
        str(source), str(audio), [track], [np.full(25, np.nan, np.float32)],
        str(tmp_path / "spk.mp4"))
    assert Path(speaker).stat().st_size > 0


# ---------------------------------------------------------------------------
# Artifact retention
# ---------------------------------------------------------------------------

def test_run_keeps_artifacts_off_by_default():
    from autoclip.cli import parser

    args = parser().parse_args(["run", "clip.mp4"])
    assert args.verbose is False


def test_analyze_and_render_do_not_take_a_verbose_flag():
    """Those two exist to write and to consume artifacts, so it has no meaning."""
    from autoclip.cli import parser

    for command in ("analyze", "render"):
        assert not hasattr(parser().parse_args([command, "clip.mp4"]), "verbose")


def test_run_main_reports_whether_it_wrote_anything(tmp_path):
    from autoclip.run import parse_args

    quiet = parse_args(["--videoPath", "x.mp4"])
    loud = parse_args(["--videoPath", "x.mp4", "--verbose"])
    assert quiet.verbose is False and loud.verbose is True
    assert quiet.workDir is None


def test_work_dir_is_redirectable(tmp_path):
    from autoclip.run import parse_args

    args = parse_args(["--videoPath", "x.mp4", "--workDir", str(tmp_path / "scratch")])
    assert args.workDir == str(tmp_path / "scratch")


@needs_ffmpeg
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
def test_default_run_leaves_only_the_video(tmp_path):
    """
    The point of the default: convert and get out. No JSON, no 25 fps working
    copy, no extracted audio, no scratch directory left behind.
    """
    from autoclip.cli import main

    source = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)

    out = tmp_path / "out"
    main(["run", str(source), "--output", str(out),
          "--weights", str(CHECKPOINT), "--device", "cpu"])
    assert [p.name for p in out.iterdir()] == ["vertical.mp4"]


@needs_ffmpeg
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the LR-ASD checkpoint")
def test_verbose_run_keeps_the_analysis(tmp_path):
    from autoclip.cli import main

    source = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)

    out = tmp_path / "out"
    main(["run", str(source), "--output", str(out), "--verbose",
          "--weights", str(CHECKPOINT), "--device", "cpu"])
    names = {p.name for p in out.iterdir()}
    assert {"vertical.mp4", "run.json", "tracks.json", "scores.json",
            "results.txt", "shortform_plan.json", "_work"} <= names
    assert (out / "_work/video_25fps.mp4").is_file()
    assert (out / "_work/audio_16khz.wav").is_file()
