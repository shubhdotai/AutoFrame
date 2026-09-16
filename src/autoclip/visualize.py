"""
Render the active-speaker bounding boxes on top of the source video.

For each frame, every tracked face is drawn:
    - GREEN, thick   = currently speaking (smoothed score >= threshold)
    - RED,   thin    = currently not speaking

Streams the input video via cv2.VideoCapture so it works on long files
(no full-video frame buffer in RAM). Audio is re-muxed in via ffmpeg at
the end so the output mp4 has sound synced with the boxes.

Renders at 25 fps using the temporary preprocessed MP4. Bbox coordinates are
computed at 25 fps so this keeps everything aligned.
"""

import os

import cv2
import numpy as np
from scipy import signal
from tqdm import tqdm

from .media import mux_audio


# BGR colors (OpenCV)
_COLOR_SPEAKING = (0, 220, 0)      # green
_COLOR_SILENT = (32, 32, 220)      # red
_COLOR_TEXT = (255, 255, 255)      # white
_COLOR_FACE = (0, 220, 0)          # green
_COLOR_PERSON = (220, 140, 0)      # orange-blue (BGR)


def _smooth(arr, window):
    arr = np.asarray(arr, dtype=np.float32)
    n = arr.shape[0]
    if n == 0 or window <= 1:
        return arr
    half = window // 2
    return np.array([
        float(np.mean(arr[max(i - half, 0): min(i + half + 1, n)]))
        for i in range(n)
    ], dtype=np.float32)


def _smooth_bbox(bboxes, kernel=13):
    """Median-filter each bbox coord over time to reduce visible jitter."""
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.shape[0] < kernel:
        return bboxes
    out = np.empty_like(bboxes)
    for j in range(4):
        out[:, j] = signal.medfilt(bboxes[:, j], kernel_size=kernel)
    return out


def render_active_speaker_video(
    input_video_path,
    audio_path,
    tracks,
    scores,
    out_path,
    threshold=0.0,
    smoothing_window=5,
    bbox_smooth_kernel=13,
    fps=25,
    show_label=True,
):
    """
    input_video_path : 25-fps preprocessed video
    audio_path       : 16 kHz audio to mux back into the output
    tracks           : list of dicts with 'frame' (np.ndarray) and 'bbox' (Nx4 np.ndarray)
    scores           : list of np.ndarray, per-frame class-1 speaking-logit scores
                       (len(scores[i]) == len(tracks[i]['frame']))
    out_path         : final .mp4 path
    threshold        : score >= threshold ⇒ speaking
    smoothing_window : odd window for mean-smoothing scores before thresholding
    bbox_smooth_kernel: odd kernel for median-smoothing bboxes (visual stability)
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- Build per-frame face list (small, in memory) ----
    # frame_faces[fidx] = list of {track, bbox (x1,y1,x2,y2), score, speaking}
    frame_faces = [[] for _ in range(n_frames)]

    for tidx, (track, raw_score) in enumerate(zip(tracks, scores)):
        frames = track["frame"]
        bboxes = _smooth_bbox(track["bbox"], kernel=bbox_smooth_kernel)
        sm = _smooth(raw_score, window=smoothing_window)
        n = min(len(frames), len(bboxes), len(sm))
        for i in range(n):
            f = int(frames[i])
            if 0 <= f < n_frames:
                frame_faces[f].append({
                    "track": tidx,
                    "bbox": bboxes[i],
                    "score": float(sm[i]),
                    "speaking": float(sm[i]) >= threshold,
                })

    # ---- Stream encode ----
    tmp_path = out_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open writer for {tmp_path}")

    for fidx in tqdm(range(n_frames), desc="render", unit="f"):
        ret, frame = cap.read()
        if not ret:
            break
        for face in frame_faces[fidx]:
            x1, y1, x2, y2 = (int(round(v)) for v in face["bbox"])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(width - 1, x2); y2 = min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            speaking = face["speaking"]
            color = _COLOR_SPEAKING if speaking else _COLOR_SILENT
            thickness = 4 if speaking else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if show_label:
                label = f"#{face['track']}  {face['score']:+.2f}"
                if speaking:
                    label = "SPEAKING  " + label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(y1 - 8, th + 8)
                cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 4), color, -1)
                cv2.putText(frame, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, _COLOR_TEXT, 2)
        writer.write(frame)

    cap.release()
    writer.release()

    try:
        mux_audio(tmp_path, audio_path, out_path)
    except Exception:
        # Fall back: just rename the silent video
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return out_path


def render_raw_detections_video(
    input_video_path,
    audio_path,
    detections_per_frame,
    out_path,
    persons_per_frame=None,
    fps=25,
    label_prefix="face",
    show_score=True,
):
    """
    Draws every raw face/person detection box on top of the source video.

    detections_per_frame : list[list[{"bbox":[x1,y1,x2,y2], "conf":float}]]
                           length == total frames; empty entries for non-sampled frames
    persons_per_frame    : optional list with same shape, drawn in a different color
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    has_persons = persons_per_frame is not None

    tmp_path = out_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open writer for {tmp_path}")

    # Cache so non-sampled frames keep showing the most recent boxes (otherwise
    # the overlay would flicker on/off every other frame).
    last_faces = []
    last_persons = []

    def _draw_box(frame, x1, y1, x2, y2, color, label, thickness):
        x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
        x2 = min(width - 1, int(round(x2))); y2 = min(height - 1, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - 6, th + 6)
            cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
            cv2.putText(frame, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, _COLOR_TEXT, 1)

    for fidx in tqdm(range(n_frames), desc="render-raw", unit="f"):
        ret, frame = cap.read()
        if not ret:
            break

        faces = detections_per_frame[fidx] if fidx < len(detections_per_frame) else []
        if faces:
            last_faces = faces
        draw_faces = faces if faces else last_faces

        if has_persons:
            persons = persons_per_frame[fidx] if fidx < len(persons_per_frame) else []
            if persons:
                last_persons = persons
            draw_persons = persons if persons else last_persons
            for det in draw_persons:
                x1, y1, x2, y2 = det["bbox"]
                lab = f"person {det['conf']:.2f}" if show_score else "person"
                _draw_box(frame, x1, y1, x2, y2, _COLOR_PERSON, lab, 2)

        for det in draw_faces:
            x1, y1, x2, y2 = det["bbox"]
            lab = f"{label_prefix} {det['conf']:.2f}" if show_score else label_prefix
            _draw_box(frame, x1, y1, x2, y2, _COLOR_FACE, lab, 2)

        writer.write(frame)

    cap.release()
    writer.release()

    try:
        mux_audio(tmp_path, audio_path, out_path)
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return out_path
