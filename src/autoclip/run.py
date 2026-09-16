"""
End-to-end analysis CLI for the optimized ASD pipeline.

Usage:
    python -m autoclip.run --videoPath path/to/video.mp4

Outputs are written to ./out by default. Preprocessed media stays under
./out/_work so every runtime artifact lives inside the project output folder.

Pipeline shape:
    1. FFmpeg, one decode      -> 25 fps MP4 + 16 kHz mono WAV
    2. One decode              -> scene cuts AND sampled face boxes
    3. Hungarian IoU tracking  -> dense per-scene tracks
    4. Gating                  -> which tracks can change the crop at all
    5. One decode              -> streaming 112x112 crops into a batched encoder
    6. Detector replay         -> per-frame speaking logits
    7. Reports
"""

import argparse
import json
import os
import time

import numpy as np

from autoclip.cropping import TrackCropper
from autoclip.gating import select_tracks_for_asd, speech_mask
from autoclip.media import ensure_dir, preprocess_video, probe_video
from autoclip.report import write_report
from autoclip.scan import scan_video
from autoclip.scenes import scene_json
from autoclip.shortform import _crop_dims
from autoclip.tracking import track_faces


def _parse_durations(value):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _default_weights_path():
    return os.path.abspath("models/pretrain_AVA.model")


def _project_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _write_json(path, data):
    # allow_nan=False: bare NaN/Infinity is not valid JSON and most parsers
    # reject it. Fail loudly here rather than emit an artifact that only
    # breaks once something downstream tries to read it.
    with open(path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)


class _Stages:
    """Wall-clock per stage, so the cost profile is visible under --verbose."""

    def __init__(self, verbose=True):
        self.times = {}
        self.verbose = verbose
        self._t0 = None
        self._label = None

    def start(self, label, text):
        # The header prints in both modes: a 60-minute video should never sit
        # silent for 20 minutes. Only the timing breakdown is verbose-only.
        print(f"\n[{label}] {text}" if self.verbose else f"[{label}] {text}")
        self._label = label
        self._t0 = time.perf_counter()

    def stop(self, extra=""):
        elapsed = time.perf_counter() - self._t0
        self.times[self._label] = elapsed
        if self.verbose:
            suffix = f" | {extra}" if extra else ""
            print(f"      {elapsed:.1f}s{suffix}")
        return elapsed

    def summary(self):
        total = sum(self.times.values())
        lines = [f"  {k:<22} {v:7.1f}s  {v / total * 100:4.1f}%"
                 for k, v in self.times.items()]
        lines.append(f"  {'total':<22} {total:7.1f}s")
        return "\n".join(lines)


def _tracks_json(tracks, params):
    return {
        "scene_aware": True,
        "params": params,
        "tracks": [
            {
                "track_id": idx,
                "scene_start_frame": int(track.get("scene_start_frame", 0)),
                "scene_end_frame": int(track.get("scene_end_frame", track["frame"][-1])),
                "mean_conf": float(track.get("mean_conf", 1.0)),
                "frame": track["frame"].tolist(),
                "bbox": track["bbox"].tolist(),
            }
            for idx, track in enumerate(tracks)
        ],
    }


def _crop_json(tracks, fps, crop_scale, selected):
    crops = []
    for idx, track in enumerate(tracks):
        frames = track["frame"]
        bboxes = np.asarray(track["bbox"], dtype=float)
        crops.append({
            "track_id": idx,
            "scored": idx in selected,
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_time_s": round(float(frames[0]) / fps, 3),
            "end_time_s": round(float(frames[-1] + 1) / fps, 3),
            "frame_count": int(len(frames)),
            "crop_size": [112, 112],
            "crop_scale": float(crop_scale),
            "avg_bbox": [float(v) for v in bboxes.mean(axis=0)],
            "persisted_media": False,
        })
    return {
        "note": ("Face crops are streamed straight into the encoder as arrays; "
                 "no per-track media is written."),
        "crops": crops,
    }


def _scores_json(tracks, scores, fps, reasons):
    return {
        "fps": float(fps),
        "note": ("values is null for tracks that were not scored; NaN means the "
                 "track never had a speaking decision to make."),
        "scores": [
            {
                "track_id": idx,
                "start_frame": int(track["frame"][0]) if len(track["frame"]) else None,
                "scored": bool(np.isfinite(score).any()) if len(score) else False,
                "skip_reason": reasons.get(idx),
                "values": [None if not np.isfinite(v) else float(v) for v in score],
            }
            for idx, (track, score) in enumerate(zip(tracks, scores))
        ],
    }


def _trim_json(report):
    trims = []
    for track in report["tracks"]:
        for segment in track["speaking_segments"]:
            trims.append({"track_id": track["track_id"], **segment})
    return {"trims": trims}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimized LR-ASD pipeline")
    p.add_argument("--videoPath", required=True, help="Input video path")
    p.add_argument(
        "--savePath", "--outDir",
        default="out",
        dest="savePath",
        help="Output run directory (default: out)",
    )
    p.add_argument(
        "--pretrainModel",
        default=_default_weights_path(),
        help="LR-ASD weights path. Defaults to models/pretrain_AVA.model.",
    )

    p.add_argument(
        "--faceDetector",
        default="apple_vision",
        choices=["apple_vision", "yolo_face_person"],
        help="Face detector backend. apple_vision = VNDetectFaceRectanglesRequest. "
             "yolo_face_person = iitolstykh/YOLO-Face-Person-Detector (face + person).",
    )
    p.add_argument("--sampleEvery", type=int, default=2,
                   help="Run face detection on every Nth frame")
    p.add_argument("--numFaceDetWorkers", type=int, default=8,
                   help="apple_vision only: thread pool size")
    p.add_argument("--faceDetChunk", type=int, default=32,
                   help="Detector batch size")
    p.add_argument("--faceDetectSize", type=int, default=720,
                   help="apple_vision only: long edge fed to Vision (0 = native)")
    p.add_argument("--minFaceConf", type=float, default=0.0,
                   help="Drop detections below this confidence before tracking")
    p.add_argument("--minFaceFraction", type=float, default=0.0,
                   help="Drop faces smaller than this fraction of frame height")
    p.add_argument("--yoloWeights",
                   default="models/yolov8x_person_face.pt",
                   help="yolo_face_person only: path to YOLO weights")
    p.add_argument("--yoloBatch", type=int, default=16,
                   help="yolo_face_person only: inference batch size")
    p.add_argument("--yoloImgsz", type=int, default=640,
                   help="yolo_face_person only: input image size")
    p.add_argument("--yoloConf", type=float, default=0.4,
                   help="yolo_face_person only: confidence threshold")
    p.add_argument("--yoloIou", type=float, default=0.7,
                   help="yolo_face_person only: NMS IoU threshold")

    p.add_argument("--sceneThreshold", type=float, default=27.0,
                   help="PySceneDetect ContentDetector threshold")
    p.add_argument("--minSceneLen", type=int, default=15,
                   help="Minimum scene length in frames")
    p.add_argument("--sceneMode", default="content", choices=["content", "adaptive"],
                   help="adaptive is more robust to fast motion and flashes")

    p.add_argument("--minTrack", type=int, default=10)
    p.add_argument("--numFailedDet", type=int, default=10)
    p.add_argument("--minFaceSize", type=int, default=1)
    p.add_argument("--iouThreshold", type=float, default=0.5)

    p.add_argument("--cropScale", type=float, default=0.40)

    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--durationSet", default="2,4,6")
    p.add_argument("--encoderWindowFrames", type=int, default=100)
    p.add_argument("--encoderBatch", type=int, default=16,
                   help="Encoder windows stacked per forward pass")
    p.add_argument("--detectorBatch", type=int, default=64)
    p.add_argument("--scoreAllTracks", action="store_true",
                   help="Score every track, including ones whose score cannot "
                        "change the crop. Slower; useful for full reports.")
    p.add_argument("--silenceFloorDb", type=float, default=-45.0,
                   help="Tracks entirely quieter than this are not scored")
    p.add_argument("--minCropShiftPx", type=int, default=8,
                   help="Crop shift below which two faces count as equivalent")

    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--smoothingWindow", type=int, default=5)

    p.add_argument("--targetAspect", default="9:16")
    p.add_argument("--verbose", action="store_true",
                   help="Write every intermediate artifact (JSON reports, "
                        "normalized media) and print the full stage profile. "
                        "Off by default: a plain run produces only the video.")
    p.add_argument("--workDir", default=None,
                   help="Where normalized media is written. Defaults to "
                        "<savePath>/_work; the caller may point this at a "
                        "scratch directory it intends to delete.")
    p.add_argument("--noVisualize", action="store_true")
    p.add_argument("--bboxSmoothKernel", type=int, default=13)
    p.add_argument("--ffmpegThreads", type=int, default=8)
    p.add_argument("--encoder", default="auto",
                   help="Video encoder: auto, hardware, software, or an FFmpeg name")
    return p.parse_args(argv)


def main(argv=None):
    """
    Run the analysis and return it in memory.

    The returned dict is everything the renderer needs, so a combined
    analyse-and-render invocation never has to round-trip through JSON. Under
    `--verbose` the same data is also written out as artifacts.
    """
    args = parse_args(argv)
    duration_set = _parse_durations(args.durationSet)
    aspect_w, aspect_h = (int(v) for v in args.targetAspect.split(":"))
    out_dir = ensure_dir(os.path.abspath(_project_path(args.savePath)))
    weights_path = os.path.abspath(_project_path(args.pretrainModel))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"missing LR-ASD weights: {weights_path}")

    verbose = bool(args.verbose)

    def write_json(name, data):
        """Artifacts are opt-in; a default run leaves nothing behind but video."""
        if verbose:
            _write_json(os.path.join(out_dir, name), data)

    if verbose:
        print("=" * 70)
        print(f"input:           {args.videoPath}")
        print(f"output dir:      {out_dir}")
        print(f"weights:         {weights_path}")
        print(f"device:          {args.device}")
        print(f"sample 1-in-N:   {args.sampleEvery}")
        print(f"durationSet:     {duration_set}")
        print("=" * 70)

    stages = _Stages(verbose=verbose)
    work_dir = ensure_dir(
        os.path.abspath(_project_path(args.workDir)) if args.workDir
        else os.path.join(out_dir, "_work")
    )
    fps = 25

    # -- 1. normalize media -------------------------------------------------
    stages.start("1/6", "preprocessing -> 25 fps MP4 + 16 kHz audio (single decode)")
    video25, audio_path = preprocess_video(
        args.videoPath, work_dir, threads=args.ffmpegThreads,
        fps=fps, encoder=args.encoder,
    )
    info = probe_video(video25)
    stages.stop(f"{info['width']}x{info['height']} @ {info['fps']:.2f} fps")

    # -- 2. scenes + faces in one pass --------------------------------------
    stages.start("2/6", f"scene cuts + face detection ({args.faceDetector}), one decode")
    min_face_px = args.minFaceFraction * info["height"]
    scan_kwargs = dict(
        backend=args.faceDetector,
        sample_every_n=args.sampleEvery,
        batch_size=args.faceDetChunk,
        scene_threshold=args.sceneThreshold,
        min_scene_len=args.minSceneLen,
        scene_mode=args.sceneMode,
        progress=verbose,
        expected_frames=info["n_frames"],
        fps=fps,
        min_confidence=args.minFaceConf,
        min_face_size=min_face_px,
    )
    if args.faceDetector == "apple_vision":
        scan_kwargs.update(num_workers=args.numFaceDetWorkers,
                           detect_size=args.faceDetectSize)
    else:
        scan_kwargs.update(
            weights_path=os.path.abspath(_project_path(args.yoloWeights)),
            device=args.device, conf=args.yoloConf, iou=args.yoloIou,
            imgsz=args.yoloImgsz, batch_size=args.yoloBatch,
        )
    scan = scan_video(video25, **scan_kwargs)
    total_frames = scan["total_frames"]
    cuts = scan["cuts"]
    persons_per_frame = scan.get("persons_per_frame")

    write_json("scenes.json", scene_json(
        video25, scenes=scan["scenes"], cuts=cuts,
        threshold=args.sceneThreshold, min_scene_len=args.minSceneLen,
        mode=args.sceneMode,
    ))
    write_json("cuts.json", {"cuts": cuts})
    fd_json = {
        "detector": args.faceDetector,
        "fps": float(fps),
        "width": scan["width"],
        "height": scan["height"],
        "total_frames": total_frames,
        "sample_every_n": scan["sample_every_n"],
        "detections": scan["detections_per_frame"],
    }
    if persons_per_frame is not None:
        fd_json["persons"] = persons_per_frame
    write_json("face_detections.json", fd_json)
    stages.stop(f"{len(cuts)} cuts | "
                f"{sum(len(d) for d in scan['detections_per_frame'])} detections")

    # -- 3. tracking --------------------------------------------------------
    stages.start("3/6", "scene-aware tracking (Hungarian IoU)")
    track_params = {
        "num_failed_det": args.numFailedDet,
        "min_track": args.minTrack,
        "min_face_size": args.minFaceSize,
        "iou_threshold": args.iouThreshold,
        "min_confidence": args.minFaceConf,
    }
    tracks = track_faces(
        scan["detections_per_frame"],
        scene_cuts=cuts,
        num_failed_det=args.numFailedDet,
        min_track=args.minTrack,
        min_face_size=args.minFaceSize,
        iou_threshold=args.iouThreshold,
        min_confidence=args.minFaceConf,
    )
    write_json("tracks.json", _tracks_json(tracks, track_params))
    stages.stop(f"{len(tracks)} tracks")

    # -- 4. decide what actually needs scoring ------------------------------
    stages.start("4/6", "selecting tracks whose score can change the crop")
    crop_w, _crop_h = _crop_dims(scan["width"], scan["height"], aspect_w, aspect_h)
    speech = None
    if args.silenceFloorDb > -np.inf:
        from scipy.io import wavfile
        sr, audio = wavfile.read(audio_path)
        speech = speech_mask(audio, sr, total_frames, fps=fps,
                             floor_db=args.silenceFloorDb)
    selected, reasons = select_tracks_for_asd(
        tracks, total_frames, scan["width"], crop_w,
        min_shift_px=args.minCropShiftPx,
        speech=speech,
        require_contested=not args.scoreAllTracks,
    )
    selected_set = set(selected)
    frames_total = sum(len(t["frame"]) for t in tracks) or 1
    frames_sel = sum(len(tracks[i]["frame"]) for i in selected)
    stages.stop(f"{len(selected)}/{len(tracks)} tracks, "
                f"{frames_sel}/{frames_total} frames ({frames_sel / frames_total:.0%})")

    # -- 5. crops streamed into the batched encoder -------------------------
    stages.start("5/6", f"face crops + LR-ASD encoders (device={args.device})")
    from .inference import EmbeddingEngine, compute_mfcc, load_model, score_detector
    model = load_model(weights_path, device=args.device)
    mfcc = compute_mfcc(audio_path)
    engine = EmbeddingEngine(model, device=args.device,
                             window=args.encoderWindowFrames, batch=args.encoderBatch)

    def on_open(idx):
        frames = tracks[idx]["frame"]
        n = len(frames)
        lo = int(frames[0]) * 4
        track_mfcc = mfcc[lo: lo + n * 4]
        if track_mfcc.shape[0] < n * 4:
            track_mfcc = np.pad(track_mfcc, ((0, n * 4 - track_mfcc.shape[0]), (0, 0)))
        engine.open_track(idx, n, track_mfcc)

    cropper = TrackCropper(
        tracks, crop_scale=args.cropScale, out_size=112,
        smooth_kernel=args.bboxSmoothKernel, selected=selected_set,
        center_half=True, grayscale=True,
    )
    cropper.run(
        video25,
        on_open=on_open,
        on_frame=lambda idx, local, crop: engine.push(idx, crop),
        on_close=lambda idx: engine.close_track(idx),
        total_frames=total_frames,
    )
    embeddings = engine.finish()
    stages.stop(f"{engine.frames_encoded} frames encoded in {engine.windows_run} windows")

    # -- 6. detector replay -------------------------------------------------
    stages.start("6/6", "detector replay over cached embeddings")
    all_scores = []
    for idx, track in enumerate(tracks):
        n = len(track["frame"])
        if idx not in embeddings:
            all_scores.append(np.full(n, np.nan, dtype=np.float32))
            continue
        audio_embed, visual_embed = embeddings[idx]
        scores = score_detector(
            model, audio_embed, visual_embed,
            duration_set=duration_set, detector_batch=args.detectorBatch, fps=fps,
        )
        if len(scores) < n:
            scores = np.concatenate([scores, np.full(n - len(scores), np.nan, np.float32)])
        all_scores.append(scores[:n])
    write_json("scores.json", _scores_json(tracks, all_scores, fps, reasons))
    write_json("crops.json", _crop_json(tracks, fps, args.cropScale, selected_set))
    stages.stop()

    # -- 7. reports ---------------------------------------------------------
    results_path = None
    if verbose:
        stages.start("reports", "writing reports")
        results_path = os.path.join(out_dir, "results.txt")
        report = write_report(
            out_path=results_path,
            json_path=os.path.join(out_dir, "results.json"),
            video_path=args.videoPath,
            tracks=tracks,
            scores=all_scores,
            fps=fps,
            threshold=args.threshold,
            smoothing_window=args.smoothingWindow,
            frame_width=scan["width"],
        )
        write_json("trims.json", _trim_json(report))
        stages.stop()

    rendered_path = None
    if not args.noVisualize:
        from autoclip.visualize import (
            render_active_speaker_video, render_raw_detections_video,
        )
        if verbose:
            print("\n[debug] rendering diagnostic videos")
        t0 = time.perf_counter()
        raw_det_path = os.path.join(out_dir, "face_detections_debug.mp4")
        render_raw_detections_video(
            input_video_path=video25, audio_path=audio_path,
            detections_per_frame=scan["detections_per_frame"],
            persons_per_frame=persons_per_frame, out_path=raw_det_path, fps=fps,
        )
        rendered_path = os.path.join(out_dir, "active_speaker_debug.mp4")
        render_active_speaker_video(
            input_video_path=video25, audio_path=audio_path, tracks=tracks,
            scores=all_scores, out_path=rendered_path, threshold=args.threshold,
            smoothing_window=args.smoothingWindow,
            bbox_smooth_kernel=args.bboxSmoothKernel, fps=fps,
        )
        if verbose:
            print(f"      took {time.perf_counter() - t0:.1f}s")

    write_json("run.json", {
        "source_video": os.path.abspath(args.videoPath),
        "output_dir": out_dir,
        "weights": weights_path,
        "face_detector": args.faceDetector,
        "target_aspect": [aspect_w, aspect_h],
        "tracks_scored": len(selected),
        "tracks_total": len(tracks),
        "stage_seconds": {k: round(v, 2) for k, v in stages.times.items()},
        "artifacts": {
            "scenes": "scenes.json",
            "cuts": "cuts.json",
            "face_detections": "face_detections.json",
            "face_detections_debug_video": None if args.noVisualize else "face_detections_debug.mp4",
            "tracks": "tracks.json",
            "crops": "crops.json",
            "scores": "scores.json",
            "results": "results.json",
            "trims": "trims.json",
            "debug_video": "active_speaker_debug.mp4" if rendered_path else None,
            "work_dir": os.path.relpath(work_dir, out_dir),
        },
    })

    if verbose:
        print("\n" + "=" * 70)
        print("stage profile:")
        print(stages.summary())
        print(f"  run dir: {out_dir}")
        if results_path:
            print(f"  report:  {results_path}")
        if rendered_path:
            print(f"  debug:   {rendered_path}")
        print("=" * 70)

    # Handed straight to the renderer, so a combined run never reloads JSON.
    return {
        "tracks": tracks,
        "scores": all_scores,
        "fps": float(fps),
        "width": scan["width"],
        "height": scan["height"],
        "n_frames": total_frames,
        "scene_cuts": cuts,
        "persons_per_frame": persons_per_frame,
        "video": video25,
        "audio": audio_path,
        "work_dir": work_dir,
        "out_dir": out_dir,
        "target_aspect": [aspect_w, aspect_h],
        "stage_seconds": {k: round(v, 2) for k, v in stages.times.items()},
        "verbose": verbose,
    }


if __name__ == "__main__":
    main()
