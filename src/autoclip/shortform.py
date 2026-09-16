"""Plan a stable speaker-following crop and render the complete video."""
import json
import os
import subprocess
import cv2
import numpy as np
from scipy import signal
from tqdm import tqdm

# ----------------------------------------------------------------------
# Source identifiers used in the plan (kept stable across versions).
# ----------------------------------------------------------------------
SOURCE_SPEAKER = "speaker"            # following the active speaker
SOURCE_HOLD = "hold"                  # current subject on screen but silent
SOURCE_LARGEST = "largest_face"       # nobody speaking; biggest visible face
SOURCE_CENTER = "center"              # nobody on screen; default to mid-frame


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _smooth_scores(scores, window):
    """Centered mean-smoothing per track (matches existing pipeline)."""
    arr = np.asarray(scores, dtype=np.float32)
    n = arr.shape[0]
    if n == 0 or window <= 1:
        return arr
    half = window // 2
    out = np.empty_like(arr)
    for i in range(n):
        out[i] = arr[max(i - half, 0): min(i + half + 1, n)].mean()
    return out


def _smooth_bboxes(bboxes, kernel):
    """Median-filter each bbox coord over time to reduce jitter."""
    arr = np.asarray(bboxes, dtype=np.float32)
    if arr.shape[0] < kernel:
        return arr
    out = np.empty_like(arr)
    for j in range(4):
        out[:, j] = signal.medfilt(arr[:, j], kernel_size=kernel)
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


# ----------------------------------------------------------------------
# Stage 1: per-frame intent (which track + cx the camera should target)
# ----------------------------------------------------------------------
def _per_frame_intent(
    tracks,
    scores,
    n_frames,
    video_width,
    speaking_threshold,
    score_smooth_window,
    bbox_smooth_kernel,
    switch_min_dwell_frames,
    scene_cuts=None,
):
    """
    Returns three arrays of length n_frames:
        target_cx  : float32, the bbox center-x to follow this frame
        track_id   : int64, the track being followed (-1 if none)
        source     : list[str], one of SOURCE_* (why we picked target_cx)
    """
    # Build a sparse index frame -> list[(track_id, smoothed_score, cx, area)]
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
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                face_at[f].append((tidx, float(smoothed[i]), float(cx), float(area)))

    target_cx = np.full(n_frames, np.nan, dtype=np.float32)
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

        top_speaker = None       # (tid, score, cx)
        face_for_current = None
        biggest_face = None
        biggest_area = -1.0
        for tid, sc, cx, area in faces:
            if sc >= speaking_threshold and (top_speaker is None or sc > top_speaker[1]):
                top_speaker = (tid, sc, cx)
            if tid == current_track:
                face_for_current = (tid, sc, cx)
            if area > biggest_area:
                biggest_area = area
                biggest_face = (tid, sc, cx)

        if top_speaker is not None:
            tid, _, cx = top_speaker
            if tid == current_track or current_track == -1:
                current_track = tid
                pending_track, pending_count = -1, 0
                target_cx[f] = cx
                track_id[f] = tid
                source[f] = SOURCE_SPEAKER
            else:
                # Different speaker -- require sustained dominance before switching.
                if tid == pending_track:
                    pending_count += 1
                else:
                    pending_track, pending_count = tid, 1

                if pending_count >= switch_min_dwell_frames:
                    current_track = pending_track
                    pending_track, pending_count = -1, 0
                    target_cx[f] = cx
                    track_id[f] = current_track
                    source[f] = SOURCE_SPEAKER
                else:
                    # Stay with current subject (visible face if any, else hold last)
                    if face_for_current is not None:
                        target_cx[f] = face_for_current[2]
                        track_id[f] = current_track
                        source[f] = SOURCE_HOLD
                    elif f > 0 and f not in cut_set and np.isfinite(target_cx[f - 1]):
                        target_cx[f] = target_cx[f - 1]
                        track_id[f] = track_id[f - 1]
                        source[f] = SOURCE_HOLD
                    else:
                        # No previous state to hold -- accept the speaker.
                        current_track = tid
                        target_cx[f] = cx
                        track_id[f] = tid
                        source[f] = SOURCE_SPEAKER
        else:
            # No active speaker.
            if face_for_current is not None:
                target_cx[f] = face_for_current[2]
                track_id[f] = current_track
                source[f] = SOURCE_HOLD
            elif biggest_face is not None:
                target_cx[f] = biggest_face[2]
                track_id[f] = biggest_face[0]
                source[f] = SOURCE_LARGEST
            elif f > 0 and f not in cut_set and np.isfinite(target_cx[f - 1]):
                target_cx[f] = target_cx[f - 1]
                track_id[f] = track_id[f - 1]
                # Holding nobody is just the centered fallback, not a real subject hold.
                source[f] = SOURCE_HOLD if track_id[f] != -1 else SOURCE_CENTER
            else:
                target_cx[f] = video_width / 2.0
                track_id[f] = -1
                source[f] = SOURCE_CENTER

    return target_cx, track_id, source


# ----------------------------------------------------------------------
# Stage 2: smooth the trajectory & build the plan
# ----------------------------------------------------------------------
def _segment_anchors(target_cx, track_id, scene_cuts, n_frames):
    """
    Collapse the per-frame target signal into a piecewise-constant anchor.

    A 'segment' is a contiguous run of frames sharing the same track_id
    AND lying inside the same scene (a scene_cuts entry breaks any
    segment that would span it). The anchor for a segment is the
    *median* of target_cx within that segment -- this rejects per-frame
    bbox jitter, head sway, and brief detection wobble. The result is a
    signal that is perfectly flat within a shot and only changes when
    the subject changes or a cut occurs.

    Returns:
        anchor_cx : float32[n_frames]   -- piecewise-constant signal
        boundaries: sorted list[int]    -- segment start frames (first is 0)
    """
    cut_set = set(int(c) for c in scene_cuts if 0 < int(c) < n_frames)
    boundaries = [0]
    for f in range(1, n_frames):
        if track_id[f] != track_id[f - 1] or f in cut_set:
            boundaries.append(f)
    boundaries.append(n_frames)

    anchor_cx = np.empty(n_frames, dtype=np.float32)
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg = target_cx[s:e]
        finite = seg[np.isfinite(seg)]
        anchor = float(np.median(finite)) if finite.size else float(target_cx[s])
        anchor_cx[s:e] = anchor

    return anchor_cx, boundaries[:-1]


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
):
    if n_frames <= 0 or fps <= 0 or video_width <= 0 or video_height <= 0:
        raise ValueError("video must have positive dimensions, duration and fps")
    if target_aspect_w <= 0 or target_aspect_h <= 0:
        raise ValueError("aspect ratio must be positive")
    if video_width / video_height < target_aspect_w / target_aspect_h:
        raise ValueError("input is narrower than target; this reframer requires a horizontal crop")
    crop_w, crop_h = _crop_dims(
        video_width, video_height, target_aspect_w, target_aspect_h
    )
    half_w = crop_w / 2.0
    min_cx = half_w
    max_cx = video_width - half_w
    switch_min_dwell_frames = max(1, int(round(switch_min_dwell_seconds * fps)))
    scene_cuts = list(scene_cuts) if scene_cuts is not None else []

    target_cx, track_id, source = _per_frame_intent(
        tracks=tracks,
        scores=scores,
        n_frames=n_frames,
        video_width=video_width,
        speaking_threshold=speaking_threshold,
        score_smooth_window=score_smooth_window,
        bbox_smooth_kernel=bbox_smooth_kernel,
        switch_min_dwell_frames=switch_min_dwell_frames,
        scene_cuts=scene_cuts,
    )
    target_cx = np.clip(target_cx, min_cx, max_cx)

    # Anchor per (track x scene) segment. Kills micro-shake within a shot,
    # and -- because we never smooth across boundaries -- every subject
    # change or scene cut becomes an instantaneous hard cut in the output.
    anchor_cx, _ = _segment_anchors(target_cx, track_id, scene_cuts, n_frames)
    anchor_cx = np.clip(anchor_cx, min_cx, max_cx)

    crop_x1 = np.round(anchor_cx - half_w).astype(np.int32)
    crop_x1 = np.clip(crop_x1, 0, int(video_width) - crop_w)

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
        },
        "scene_cuts": [int(c) for c in scene_cuts],
        "crop_x1": crop_x1,         # numpy; serialized below
        "track": track_id,
        "source": source,
        "center_x": anchor_cx,
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
    crop_x1 = plan["crop_x1"]
    crop_w = plan["crop_width"]

    segments = _segments_from(track, source)
    for s in segments:
        s["start_time_s"] = round(s["start_frame"] / fps, 3)
        s["end_time_s"] = round((s["end_frame"] + 1) / fps, 3)
        s["duration_s"] = round(s["end_time_s"] - s["start_time_s"], 3)

    return {
        "fps": fps,
        "video_width": plan["video_width"],
        "video_height": plan["video_height"],
        "crop_width": crop_w,
        "crop_height": plan["crop_height"],
        "target_aspect": plan["target_aspect"],
        "n_frames": plan["n_frames"],
        "params": plan["params"],
        "scene_cuts": plan.get("scene_cuts", []),
        "segments": segments,
        "per_frame": {
            "crop_x1": crop_x1.tolist(),
            "track": track.tolist(),
            "source": list(source),
        },
        "_note": (
            "For each frame f, crop the source to [crop_x1[f] : crop_x1[f]+crop_width] "
            "horizontally, using rows [0:crop_height]. center_x = crop_x1 + crop_width/2. "
            "The camera position is piecewise-constant: every change in `track` or any "
            "frame in scene_cuts is a hard cut (no pans, no smoothing across boundaries)."
        ),
    }


def render_video(input_video_path, audio_path, plan, out_path):
    """Stream all planned frames, then encode H.264 and mux AAC audio."""
    fps = float(plan["fps"])
    crop_w, crop_h = int(plan["crop_width"]), int(plan["crop_height"])
    n_frames = int(plan["n_frames"])
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_video_path}")
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (plan["video_width"], plan["video_height"]):
        cap.release()
        raise ValueError("video dimensions differ from analysis")
    tmp_path = str(out_path) + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open writer for {tmp_path}")
    written = 0
    try:
        for f in tqdm(range(n_frames), desc="reframe", unit="frame"):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended early: {written}/{n_frames} frames")
            x1 = int(plan["crop_x1"][f])
            # Height may lose a small bottom strip to obtain an exact even 9:16 crop.
            writer.write(frame[:crop_h, x1:x1 + crop_w])
            written += 1
    finally:
        cap.release()
        writer.release()
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path, "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-af", "apad", "-t", str(n_frames / fps),
        "-movflags", "+faststart", "-loglevel", "error", str(out_path),
    ], check=True)
    os.remove(tmp_path)
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
    scores = [np.asarray(s, dtype=np.float32) for s in scores_raw]

    return {
        "tracks": tracks,
        "scores": scores,
        "fps": float(fd_meta["fps"]),
        "width": int(fd_meta["width"]),
        "height": int(fd_meta["height"]),
        "n_frames": int(fd_meta["total_frames"]),
        "scene_cuts": [int(c) for c in cuts],
    }


