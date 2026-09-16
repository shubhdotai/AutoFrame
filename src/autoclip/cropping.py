"""
Per-track face crops, produced in one sequential pass over the video.

The original extracted each track separately: seek to the track start, decode,
write a 224x224 `mp4v` file, shell out to FFmpeg twice (once to slice a WAV,
once to mux AAC into the MP4 that inference never reads), then decode that MP4
again inside the model. On a 147 s clip with 97 tracks that measured ~34 s of
FFmpeg spawns and keyframe seeks against ~2 s of actual network compute, and it
fed the model lossily re-encoded pixels.

Now the video is decoded once, in order, and every track live at a given frame
takes its crop from that one decoded frame. Crops go straight to the consumer
as arrays.
"""

import numpy as np
from scipy.ndimage import median_filter

import cv2

# Matches the original 13-tap smoothing, but with edge replication instead of
# scipy.signal.medfilt's implicit zero padding. Zero padding pulled the first
# and last ~6 centres of every track toward the origin, shifting the crop box
# left and up and shrinking it at both ends of every track.
_SMOOTH_KERNEL = 13

# Neutral grey used where a crop runs past the frame edge, matching upstream.
_PAD_VALUE = 110


def smooth_track_geometry(bboxes, kernel=_SMOOTH_KERNEL):
    """Return (cx, cy, half_size) arrays, median-smoothed over time."""
    bboxes = np.asarray(bboxes, dtype=float)
    size = np.maximum(bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    if len(size) >= 2:
        k = min(kernel, len(size) if len(size) % 2 else len(size) - 1)
        if k >= 3:
            size = median_filter(size, size=k, mode="nearest")
            cy = median_filter(cy, size=k, mode="nearest")
            cx = median_filter(cx, size=k, mode="nearest")
    return cx, cy, size


def crop_face(frame, cx, cy, half, crop_scale=0.40, out_size=224,
              center_half=False, grayscale=False):
    """
    Cut the model's face box out of a full frame.

    Geometry matches the upstream crop: `half` above centre, `half * (1 + 2cs)`
    below, `half * (1 + cs)` either side. Only the requested rectangle is
    touched -- the original padded the entire frame on every track on every
    frame, which is O(frame) work to produce an O(crop) result.

    center_half=True returns the central 50% of that box, which is the region
    the visual encoder actually consumes; grayscale=True converts before the
    resize, which is what the encoder wants and is cheaper than converting a
    full-colour crop afterwards.
    """
    y0 = cy - half
    y1 = cy + half * (1 + 2 * crop_scale)
    x0 = cx - half * (1 + crop_scale)
    x1 = cx + half * (1 + crop_scale)
    if center_half:
        qy, qx = (y1 - y0) / 4.0, (x1 - x0) / 4.0
        y0, y1, x0, x1 = y0 + qy, y1 - qy, x0 + qx, x1 - qx

    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Empty face crop; refusing to shift audio/video alignment")

    h, w = frame.shape[:2]
    sy0, sy1 = max(y0, 0), min(y1, h)
    sx0, sx1 = max(x0, 0), min(x1, w)
    if sy1 <= sy0 or sx1 <= sx0:
        face = np.full((y1 - y0, x1 - x0, frame.shape[2]), _PAD_VALUE, dtype=frame.dtype)
    else:
        patch = frame[sy0:sy1, sx0:sx1]
        top, bottom = sy0 - y0, y1 - sy1
        left, right = sx0 - x0, x1 - sx1
        if top or bottom or left or right:
            patch = cv2.copyMakeBorder(
                patch, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * frame.shape[2],
            )
        face = patch

    if grayscale and face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if out_size is None:
        return face
    if face.shape[0] == out_size and face.shape[1] == out_size:
        return face
    interp = cv2.INTER_AREA if face.shape[0] > out_size else cv2.INTER_LINEAR
    return cv2.resize(face, (out_size, out_size), interpolation=interp)


class TrackCropper:
    """
    Sequential crop pass.

    `on_open(track_idx)`, `on_frame(track_idx, local_idx, crop)` and
    `on_close(track_idx)` are called in frame order as the video is decoded.
    """

    def __init__(self, tracks, crop_scale=0.40, out_size=224,
                 smooth_kernel=_SMOOTH_KERNEL, selected=None, center_half=False,
                 grayscale=False):
        self.tracks = tracks
        self.crop_scale = float(crop_scale)
        self.out_size = out_size
        self.center_half = bool(center_half)
        self.grayscale = bool(grayscale)
        self.selected = set(range(len(tracks))) if selected is None else set(selected)

        self.geometry = []
        self.starts = []
        self.ends = []
        for track in tracks:
            frames = np.asarray(track["frame"], dtype=np.int64)
            self.geometry.append(smooth_track_geometry(track["bbox"], smooth_kernel))
            self.starts.append(int(frames[0]) if len(frames) else 0)
            self.ends.append(int(frames[-1]) if len(frames) else -1)

        # frame -> tracks starting there, so activation is O(1) per frame.
        self.opening = {}
        for idx in sorted(self.selected, key=lambda i: self.starts[i]):
            if self.ends[idx] >= self.starts[idx]:
                self.opening.setdefault(self.starts[idx], []).append(idx)

    def run(self, video_path, on_open, on_frame, on_close, total_frames=None,
            progress=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        active = []
        frame_idx = 0
        try:
            while True:
                if total_frames is not None and frame_idx >= total_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                for idx in self.opening.get(frame_idx, ()):
                    on_open(idx)
                    active.append(idx)

                if active:
                    still = []
                    for idx in active:
                        local = frame_idx - self.starts[idx]
                        cx, cy, half = self.geometry[idx]
                        if local < len(half):
                            on_frame(idx, local, crop_face(
                                frame, cx[local], cy[local], half[local],
                                crop_scale=self.crop_scale, out_size=self.out_size,
                                center_half=self.center_half,
                                grayscale=self.grayscale,
                            ))
                        if frame_idx >= self.ends[idx]:
                            on_close(idx)
                        else:
                            still.append(idx)
                    active = still

                frame_idx += 1
                if progress is not None:
                    progress(frame_idx)
        finally:
            cap.release()
            for idx in active:
                on_close(idx)
        return frame_idx


def crop_face_track(
    source_video_path,
    source_audio_path,
    track,
    out_prefix,
    fps=25,
    crop_scale=0.40,
    n_threads=4,
    out_size=224,
):
    """
    Write one track's face video and audio to disk.

    Retained for diagnostics and for tools that want inspectable per-track
    media. The analysis pipeline does not call this: it streams crops straight
    into the encoder instead. Audio is sliced from the already-decoded WAV with
    numpy rather than by spawning FFmpeg, which measured 143 ms per track.
    """
    import os
    from scipy.io import wavfile

    from .media import FrameSink

    frames = np.asarray(track["frame"], dtype=np.int64)
    cx, cy, half = smooth_track_geometry(track["bbox"])
    out_mp4 = out_prefix + ".mp4"
    out_wav = out_prefix + ".wav"

    sr, audio = wavfile.read(source_audio_path)
    lo = int(round(float(frames[0]) / fps * sr))
    hi = int(round(float(frames[-1] + 1) / fps * sr))
    wavfile.write(out_wav, sr, audio[max(lo, 0):max(hi, 0)])

    cropper = TrackCropper([track], crop_scale=crop_scale, out_size=out_size)
    sink = FrameSink(out_mp4, out_size, out_size, fps, quality="analysis")
    written = 0

    def on_frame(_idx, _local, crop):
        nonlocal written
        sink.write(crop)
        written += 1

    try:
        cropper.run(source_video_path, lambda i: None, on_frame, lambda i: None,
                    total_frames=int(frames[-1]) + 1)
    finally:
        sink.close()

    if written != len(frames):
        raise RuntimeError(f"Incomplete track crop: {written}/{len(frames)} frames")
    if not os.path.exists(out_mp4):
        raise RuntimeError(f"crop encode produced no output: {out_mp4}")
    return out_mp4, out_wav
