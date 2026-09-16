"""Plan a stable speaker-following crop and render the complete video."""
import json
import os

import cv2
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from .media import FrameSink, probe_video

# ----------------------------------------------------------------------
# Source identifiers used in the plan (kept stable across versions).
# ----------------------------------------------------------------------
SOURCE_SPEAKER = "speaker"            # following the active speaker
SOURCE_HOLD = "hold"                  # current subject on screen but silent
SOURCE_LARGEST = "largest_face"       # nobody speaking; biggest visible face
SOURCE_PERSON = "person"              # no face at all; biggest person box
SOURCE_CENTER = "center"              # nobody on screen; default to mid-frame


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _smooth_scores(scores, window):
    """
    Centred mean-smoothing with shrinking edges.

    Same result as the original per-frame Python loop, computed with a prefix
    sum so it is linear in the track length rather than length x window.

    NaN marks a frame that was never scored, and averaging over it must not
    poison its neighbours: a track can be scored for most of its length and
    NaN-padded where the audio ran short. Only finite values contribute, and a
    window with nothing finite in it stays NaN.
    """
    arr = np.asarray(scores, dtype=np.float32)
    n = arr.shape[0]
    if n == 0 or window <= 1:
        return arr
    half = window // 2
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    totals = np.concatenate(([0.0], np.cumsum(filled, dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    lo = np.maximum(np.arange(n) - half, 0)
    hi = np.minimum(np.arange(n) + half + 1, n)
    total = totals[hi] - totals[lo]
    count = counts[hi] - counts[lo]
    out = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    return out.astype(np.float32)


def _smooth_bboxes(bboxes, kernel):
    """
    Median-filter each bbox coord over time to reduce jitter.

    `mode="nearest"` replicates the edges; scipy.signal.medfilt's implicit zero
    padding biased the first and last few boxes of every track toward 0.
    """
    arr = np.asarray(bboxes, dtype=np.float32)
    n = arr.shape[0]
    if n < 3 or kernel < 3:
        return arr
    k = min(kernel, n if n % 2 else n - 1)
    if k < 3:
        return arr
    out = np.empty_like(arr)
    for j in range(4):
        out[:, j] = median_filter(arr[:, j], size=k, mode="nearest")
    return out


def _crop_dims(video_width, video_height, aspect_w, aspect_h):
    """Largest exact-aspect crop with even dimensions for H.264/yuv420p."""
    from math import gcd
    common = gcd(aspect_w, aspect_h)
    w, h = aspect_w // common, aspect_h // common
    scale = min(video_width // w, video_height // h)
    scale -= scale % 2
    if scale < 2:
        raise ValueError("video is too small for the target aspect ratio")
    return w * scale, h * scale


def _is_speaking(score, threshold):
    """NaN (an unscored track) is never speaking, and never raises a warning."""
    return score == score and score >= threshold


# ----------------------------------------------------------------------
# Stage 1: per-frame intent (which track + centre the camera should target)
# ----------------------------------------------------------------------
def _per_frame_intent(
    tracks,
    scores,
    n_frames,
    video_width,
    video_height,
    speaking_threshold,
    score_smooth_window,
    bbox_smooth_kernel,
    switch_min_dwell_frames,
    scene_cuts=None,
    speaker_margin=0.0,
    persons_per_frame=None,
):
    """
    Returns four arrays of length n_frames:
        target_cx  : float32, the bbox center-x to follow this frame
        target_cy  : float32, the bbox center-y to follow this frame
        track_id   : int64, the track being followed (-1 if none)
        source     : list[str], one of SOURCE_* (why we picked the target)
    """
    # Build a sparse index frame -> list[(track_id, smoothed_score, cx, cy, area)]
    face_at = [[] for _ in range(n_frames)]
    for tidx, (track, raw_score) in enumerate(zip(tracks, scores)):
        frames = np.asarray(track["frame"], dtype=np.int64)
        bboxes = _smooth_bboxes(track["bbox"], kernel=bbox_smooth_kernel)
        smoothed = _smooth_scores(raw_score, window=score_smooth_window)
        m = min(len(frames), len(bboxes), len(smoothed))
        for i in range(m):
            f = int(frames[i])
            if 0 <= f < n_frames:
                x1, y1, x2, y2 = bboxes[i]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                face_at[f].append(
                    (tidx, float(smoothed[i]), float(cx), float(cy), float(area))
                )

    target_cx = np.full(n_frames, np.nan, dtype=np.float32)
    target_cy = np.full(n_frames, np.nan, dtype=np.float32)
    track_id = np.full(n_frames, -1, dtype=np.int64)
    source = [SOURCE_CENTER] * n_frames

    current_track = -1
    pending_track = -1
    pending_count = 0

    cut_set = set(scene_cuts or [])
    for f in range(n_frames):
        if f in cut_set:
            current_track = pending_track = -1
            pending_count = 0
        faces = face_at[f]

        top_speaker = None       # (tid, score, cx, cy)
        runner_up = None
        face_for_current = None
        biggest_face = None
        biggest_area = -1.0
        for tid, sc, cx, cy, area in faces:
            if _is_speaking(sc, speaking_threshold):
                if top_speaker is None or sc > top_speaker[1]:
                    runner_up = top_speaker[1] if top_speaker is not None else runner_up
                    top_speaker = (tid, sc, cx, cy)
                elif runner_up is None or sc > runner_up:
                    runner_up = sc
            if tid == current_track:
                face_for_current = (tid, sc, cx, cy)
            if area > biggest_area:
                biggest_area = area
                biggest_face = (tid, sc, cx, cy)

        # A lead too narrow to trust is treated as nobody clearly speaking.
        if (top_speaker is not None and runner_up is not None
                and speaker_margin > 0.0
                and top_speaker[1] - runner_up < speaker_margin
                and top_speaker[0] != current_track):
            top_speaker = None

        def _set(entry, label, tid=None):
            target_cx[f] = entry[2]
            target_cy[f] = entry[3]
            track_id[f] = entry[0] if tid is None else tid
            source[f] = label

        if top_speaker is not None:
            tid = top_speaker[0]
            if tid == current_track or current_track == -1:
                current_track = tid
                pending_track, pending_count = -1, 0
                _set(top_speaker, SOURCE_SPEAKER)
            else:
                # Different speaker -- require sustained dominance before switching.
                if tid == pending_track:
                    pending_count += 1
                else:
                    pending_track, pending_count = tid, 1

                if pending_count >= switch_min_dwell_frames:
                    current_track = pending_track
                    pending_track, pending_count = -1, 0
                    _set(top_speaker, SOURCE_SPEAKER, tid=current_track)
                elif face_for_current is not None:
                    _set(face_for_current, SOURCE_HOLD, tid=current_track)
                elif f > 0 and f not in cut_set and np.isfinite(target_cx[f - 1]):
                    target_cx[f] = target_cx[f - 1]
                    target_cy[f] = target_cy[f - 1]
                    track_id[f] = track_id[f - 1]
                    source[f] = SOURCE_HOLD
                else:
                    # No previous state to hold -- accept the speaker.
                    current_track = tid
                    _set(top_speaker, SOURCE_SPEAKER)
        else:
            # No active speaker.
            person = None
            if not faces and persons_per_frame is not None and f < len(persons_per_frame):
                person = _largest_person(persons_per_frame[f])

            if face_for_current is not None:
                _set(face_for_current, SOURCE_HOLD, tid=current_track)
            elif biggest_face is not None:
                _set(biggest_face, SOURCE_LARGEST)
            elif person is not None:
                # Nobody's face is visible but a body is. Framing the person is
                # a great deal better than holding a stale centre or defaulting
                # to the middle of the frame, which is where ~40% of frames of
                # cut-heavy footage end up.
                target_cx[f], target_cy[f] = person
                track_id[f] = -1
                source[f] = SOURCE_PERSON
            elif f > 0 and f not in cut_set and np.isfinite(target_cx[f - 1]):
                target_cx[f] = target_cx[f - 1]
                target_cy[f] = target_cy[f - 1]
                track_id[f] = track_id[f - 1]
                # Holding nobody is just the centered fallback, not a real subject hold.
                source[f] = SOURCE_HOLD if track_id[f] != -1 else SOURCE_CENTER
            else:
                target_cx[f] = video_width / 2.0
                target_cy[f] = video_height / 2.0
                track_id[f] = -1
                source[f] = SOURCE_CENTER

    return target_cx, target_cy, track_id, source


def _largest_person(persons):
    """Centre to frame for the biggest person box: horizontally centred, head high."""
    best = None
    best_area = 0.0
    for person in persons or ():
        x1, y1, x2, y2 = person["bbox"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    if best is None:
        return None
    x1, y1, x2, y2 = best
    # Aim a third of the way down the body, which is roughly head/shoulders.
    return 0.5 * (x1 + x2), y1 + (y2 - y1) / 3.0


# ----------------------------------------------------------------------
# Stage 2: smooth the trajectory & build the plan
# ----------------------------------------------------------------------
def _segment_anchors(target, track_id, scene_cuts, n_frames):
    """
    Collapse the per-frame target signal into a piecewise-constant anchor.

    A 'segment' is a contiguous run of frames sharing the same track_id
    AND lying inside the same scene (a scene_cuts entry breaks any
    segment that would span it). The anchor for a segment is the
    *median* of the target within that segment -- this rejects per-frame
    bbox jitter, head sway, and brief detection wobble. The result is a
    signal that is perfectly flat within a shot and only changes when
    the subject changes or a cut occurs.

    Returns:
        anchor    : float32[n_frames]   -- piecewise-constant signal
        boundaries: sorted list[int]    -- segment start frames (first is 0)
    """
    cut_set = set(int(c) for c in scene_cuts if 0 < int(c) < n_frames)
    boundaries = [0]
    for f in range(1, n_frames):
        if track_id[f] != track_id[f - 1] or f in cut_set:
            boundaries.append(f)
    boundaries.append(n_frames)

    anchor = np.empty(n_frames, dtype=np.float32)
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg = target[s:e]
        finite = seg[np.isfinite(seg)]
        value = float(np.median(finite)) if finite.size else float(target[s])
        anchor[s:e] = value

    return anchor, boundaries[:-1]


def _follow_anchors(target, track_id, scene_cuts, n_frames, crop_span,
                    deadzone=0.12, response=0.08):
    """
    Damped follow with a deadzone, as an alternative to a locked crop.

    The camera holds still while the subject stays within `deadzone` of the
    crop span, and eases toward them when they leave it. This keeps a shot
    framed when someone walks or leans out of a locked crop, without
    reintroducing the per-frame shake that the locked anchor exists to remove.
    Boundaries are still hard cuts: nothing is smoothed across them.
    """
    anchor, boundaries = _segment_anchors(target, track_id, scene_cuts, n_frames)
    limit = deadzone * crop_span
    bounds = list(boundaries) + [n_frames]
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        position = float(anchor[s])
        for f in range(s, e):
            goal = target[f]
            if not np.isfinite(goal):
                goal = position
            error = goal - position
            if abs(error) > limit:
                position += (error - np.sign(error) * limit) * response
            out[f] = position
    return out


def build_reframe_plan(
    tracks,
    scores,
    n_frames,
    video_width,
    video_height,
    fps=25,
    target_aspect_w=9,
    target_aspect_h=16,
    speaking_threshold=0.0,
    score_smooth_window=11,         # ~0.44 s @ 25 fps
    bbox_smooth_kernel=13,          # ~0.52 s; matches visualize.py default
    switch_min_dwell_seconds=0.4,   # required dominance before camera switches
    scene_cuts=None,                # list[int] frame indices; empty if unknown
    speaker_margin=0.0,             # lead required over the runner-up
    persons_per_frame=None,         # fallback subject when no face is visible
    motion="lock",                  # "lock" (piecewise-constant) or "follow"
):
    if n_frames <= 0 or fps <= 0 or video_width <= 0 or video_height <= 0:
        raise ValueError("video must have positive dimensions, duration and fps")
    if target_aspect_w <= 0 or target_aspect_h <= 0:
        raise ValueError("aspect ratio must be positive")
    if video_width / video_height < target_aspect_w / target_aspect_h:
        raise ValueError("input is narrower than target; this reframer requires a horizontal crop")
    if motion not in ("lock", "follow"):
        raise ValueError("motion must be 'lock' or 'follow'")
    crop_w, crop_h = _crop_dims(
        video_width, video_height, target_aspect_w, target_aspect_h
    )
    half_w, half_h = crop_w / 2.0, crop_h / 2.0
    min_cx, max_cx = half_w, video_width - half_w
    min_cy, max_cy = half_h, video_height - half_h
    switch_min_dwell_frames = max(1, int(round(switch_min_dwell_seconds * fps)))
    scene_cuts = list(scene_cuts) if scene_cuts is not None else []

    target_cx, target_cy, track_id, source = _per_frame_intent(
        tracks=tracks,
        scores=scores,
        n_frames=n_frames,
        video_width=video_width,
        video_height=video_height,
        speaking_threshold=speaking_threshold,
        score_smooth_window=score_smooth_window,
        bbox_smooth_kernel=bbox_smooth_kernel,
        switch_min_dwell_frames=switch_min_dwell_frames,
        scene_cuts=scene_cuts,
        speaker_margin=speaker_margin,
        persons_per_frame=persons_per_frame,
    )
    target_cx = np.clip(target_cx, min_cx, max_cx)
    target_cy = np.clip(target_cy, min_cy, max_cy)

    # Anchor per (track x scene) segment. Kills micro-shake within a shot,
    # and -- because we never smooth across boundaries -- every subject
    # change or scene cut becomes an instantaneous hard cut in the output.
    if motion == "follow":
        anchor_cx = _follow_anchors(target_cx, track_id, scene_cuts, n_frames, crop_w)
        anchor_cy = _follow_anchors(target_cy, track_id, scene_cuts, n_frames, crop_h)
    else:
        anchor_cx, _ = _segment_anchors(target_cx, track_id, scene_cuts, n_frames)
        anchor_cy, _ = _segment_anchors(target_cy, track_id, scene_cuts, n_frames)
    anchor_cx = np.clip(anchor_cx, min_cx, max_cx)
    anchor_cy = np.clip(anchor_cy, min_cy, max_cy)

    crop_x1 = np.round(anchor_cx - half_w).astype(np.int32)
    crop_x1 = np.clip(crop_x1, 0, int(video_width) - crop_w)
    # Vertical placement was previously fixed at row 0, which clipped faces
    # whenever the exact-aspect crop was shorter than the source frame.
    crop_y1 = np.round(anchor_cy - half_h).astype(np.int32)
    crop_y1 = np.clip(crop_y1, 0, int(video_height) - crop_h)

    return {
        "fps": float(fps),
        "video_width": int(video_width),
        "video_height": int(video_height),
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "target_aspect": [int(target_aspect_w), int(target_aspect_h)],
        "n_frames": int(n_frames),
        "params": {
            "speaking_threshold": float(speaking_threshold),
            "score_smooth_window": int(score_smooth_window),
            "bbox_smooth_kernel": int(bbox_smooth_kernel),
            "switch_min_dwell_seconds": float(switch_min_dwell_seconds),
            "speaker_margin": float(speaker_margin),
            "motion": motion,
        },
        "scene_cuts": [int(c) for c in scene_cuts],
        "crop_x1": crop_x1,         # numpy; serialized below
        "crop_y1": crop_y1,
        "track": track_id,
        "source": source,
        "center_x": anchor_cx,
        "center_y": anchor_cy,
    }


# ----------------------------------------------------------------------
# Plan I/O -- compact JSON with both per-frame arrays and an RLE summary
# ----------------------------------------------------------------------
def _segments_from(track_id, source):
    """Run-length encode (track, source) into human-readable segments."""
    n = len(source)
    segments = []
    if n == 0:
        return segments
    start = 0
    cur = (int(track_id[0]), source[0])
    for i in range(1, n):
        nxt = (int(track_id[i]), source[i])
        if nxt != cur:
            segments.append({
                "start_frame": start, "end_frame": i - 1,
                "track": cur[0], "source": cur[1],
            })
            start = i
            cur = nxt
    segments.append({
        "start_frame": start, "end_frame": n - 1,
        "track": cur[0], "source": cur[1],
    })
    return segments


def serialize_plan(plan):
    """Convert the in-memory plan to a JSON-friendly dict."""
    fps = plan["fps"]
    track = plan["track"]
    source = plan["source"]

    segments = _segments_from(track, source)
    for s in segments:
        s["start_time_s"] = round(s["start_frame"] / fps, 3)
        s["end_time_s"] = round((s["end_frame"] + 1) / fps, 3)
        s["duration_s"] = round(s["end_time_s"] - s["start_time_s"], 3)

    return {
        "fps": fps,
        "video_width": plan["video_width"],
        "video_height": plan["video_height"],
        "crop_width": plan["crop_width"],
        "crop_height": plan["crop_height"],
        "target_aspect": plan["target_aspect"],
        "n_frames": plan["n_frames"],
        "params": plan["params"],
        "scene_cuts": plan.get("scene_cuts", []),
        "segments": segments,
        "per_frame": {
            "crop_x1": plan["crop_x1"].tolist(),
            "crop_y1": plan["crop_y1"].tolist(),
            "track": track.tolist(),
            "source": list(source),
        },
        "_note": (
            "For each frame f, crop the source to [crop_x1[f] : crop_x1[f]+crop_width] "
            "horizontally and [crop_y1[f] : crop_y1[f]+crop_height] vertically. "
            "center_x = crop_x1 + crop_width/2. The camera position is "
            "piecewise-constant under motion=lock: every change in `track` or any "
            "frame in scene_cuts is a hard cut (no pans, no smoothing across "
            "boundaries)."
        ),
    }


def render_video(input_video_path, audio_path, plan, out_path, native_fps=False,
                 output_height=None, encoder="auto", progress=True):
    """
    Crop every planned frame and encode once.

    The previous renderer wrote an `mp4v` temp file and then re-encoded it to
    H.264, which cost two encodes and two generations of lossy compression on
    the deliverable. Frames now go straight into a single H.264 encode over a
    pipe.

    native_fps reads the original source instead of the 25 fps analysis copy,
    so a 60 fps input stays 60 fps and never picks up the intermediate's
    generation loss. The crop plan is piecewise-constant in time, so it maps
    onto any output frame rate.
    """
    fps = float(plan["fps"])
    crop_w, crop_h = int(plan["crop_width"]), int(plan["crop_height"])
    n_frames = int(plan["n_frames"])
    crop_x1 = np.asarray(plan["crop_x1"], dtype=np.int64)
    crop_y1 = np.asarray(plan.get("crop_y1", np.zeros(n_frames, dtype=np.int64)),
                         dtype=np.int64)

    info = probe_video(input_video_path)
    if (info["width"], info["height"]) != (plan["video_width"], plan["video_height"]):
        raise ValueError("video dimensions differ from analysis")
    out_fps = info["fps"] if (native_fps and info["fps"] > 0) else fps
    max_out = int(np.ceil(n_frames / fps * out_fps - 1e-9))

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_video_path}")

    out_size = None
    if output_height:
        scaled_w = int(round(crop_w * (output_height / crop_h)))
        out_size = (scaled_w - scaled_w % 2, int(output_height) - int(output_height) % 2)

    sink = FrameSink(
        out_path, crop_w, crop_h, out_fps,
        audio_path=audio_path,
        audio_duration=n_frames / fps,
        out_size=out_size,
        encoder=encoder,
        quality="final",
    )
    written = 0
    ratio = fps / out_fps
    try:
        # disable=None lets tqdm suppress itself when stderr is not a terminal,
        # so piped logs do not fill up with redrawn progress bars.
        bar = tqdm(total=max_out, desc="reframe", unit="frame",
                   disable=None if progress else True)
        while written < max_out:
            ok, frame = cap.read()
            if not ok:
                break
            f = min(n_frames - 1, int(written * ratio))
            x1, y1 = int(crop_x1[f]), int(crop_y1[f])
            sink.write(frame[y1:y1 + crop_h, x1:x1 + crop_w])
            written += 1
            bar.update(1)
        bar.close()
        if written == 0:
            raise RuntimeError(f"no frames read from {input_video_path}")
        if written < max_out and not native_fps:
            raise RuntimeError(f"video ended early: {written}/{max_out} frames")
    finally:
        cap.release()
        sink.close()
    return str(out_path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _load_artifacts(save_path):
    tracks_payload = _read_json(os.path.join(save_path, "tracks.json"))
    scores_payload = _read_json(os.path.join(save_path, "scores.json"))
    fd_meta = _read_json(os.path.join(save_path, "face_detections.json"))

    tracks_raw = tracks_payload.get("tracks", []) if isinstance(tracks_payload, dict) else tracks_payload
    if isinstance(scores_payload, dict):
        score_items = scores_payload.get("scores", [])
        scores_raw = [item.get("values", item) for item in score_items]
    else:
        scores_raw = scores_payload

    cuts = []
    cuts_path = os.path.join(save_path, "cuts.json")
    scenes_path = os.path.join(save_path, "scenes.json")
    if os.path.exists(cuts_path):
        cuts_payload = _read_json(cuts_path)
        cuts = cuts_payload.get("cuts", []) if isinstance(cuts_payload, dict) else cuts_payload
    elif os.path.exists(scenes_path):
        scenes_payload = _read_json(scenes_path)
        cuts = scenes_payload.get("cuts", [])

    tracks = [
        {"frame": np.asarray(t["frame"], dtype=np.int64),
         "bbox": np.asarray(t["bbox"], dtype=np.float32)}
        for t in tracks_raw
    ]
    # `null` in scores.json means the track was deliberately not scored.
    scores = [
        np.asarray([np.nan if v is None else v for v in s], dtype=np.float32)
        for s in scores_raw
    ]

    return {
        "tracks": tracks,
        "scores": scores,
        "fps": float(fd_meta["fps"]),
        "width": int(fd_meta["width"]),
        "height": int(fd_meta["height"]),
        "n_frames": int(fd_meta["total_frames"]),
        "scene_cuts": [int(c) for c in cuts],
        "persons_per_frame": fd_meta.get("persons"),
    }
