"""Per-track face video + audio extraction for model inference."""

import os

import cv2
import numpy as np
from scipy import signal

from .media import run_ffmpeg


def crop_face_track(
    source_video_path,
    source_audio_path,
    track,
    out_prefix,
    fps=25,
    crop_scale=0.40,
    n_threads=4,
):
    """
    track: dict with 'frame' (np.ndarray of contiguous frame indices) and
           'bbox' (np.ndarray Nx4).
    out_prefix: path prefix; produces {out_prefix}.mp4 (video+audio muxed) and
                {out_prefix}.wav.
    Returns (mp4_path, wav_path).
    """
    # Center / size per frame, smoothed with a 13-tap median filter
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max(det[3] - det[1], det[2] - det[0]) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(np.asarray(dets["s"], dtype=float), kernel_size=13)
    dets["x"] = signal.medfilt(np.asarray(dets["x"], dtype=float), kernel_size=13)
    dets["y"] = signal.medfilt(np.asarray(dets["y"], dtype=float), kernel_size=13)

    tmp_mp4 = out_prefix + "_tmp.mp4"
    out_mp4 = out_prefix + ".mp4"
    out_wav = out_prefix + ".wav"

    cap = cv2.VideoCapture(source_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(track["frame"][0]))
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (224, 224))

    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open track video reader/writer")
    expected_n = len(track["frame"])
    written = 0
    for i in range(expected_n):
        ret, image = cap.read()
        if not ret:
            break

        cs = crop_scale
        bs = dets["s"][i]
        bsi = int(bs * (1 + 2 * cs))
        padded = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            mode="constant",
            constant_values=110,
        )
        my = dets["y"][i] + bsi
        mx = dets["x"][i] + bsi
        face = padded[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        if face.size == 0:
            cap.release()
            writer.release()
            raise ValueError("Empty face crop; refusing to shift audio/video alignment")
        writer.write(cv2.resize(face, (224, 224)))
        written += 1

    cap.release()
    writer.release()

    if written != expected_n:
        raise RuntimeError(f"Incomplete track crop: {written}/{expected_n} frames")

    # Slice audio for this track
    audio_start = float(track["frame"][0]) / fps
    audio_end = float(track["frame"][-1] + 1) / fps
    run_ffmpeg([
        "-i", source_audio_path,
        "-ac", "1",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-threads", str(n_threads),
        "-ss", f"{audio_start:.3f}",
        "-to", f"{audio_end:.3f}",
        out_wav,
    ])

    # Mux video + audio
    run_ffmpeg([
        "-i", tmp_mp4,
        "-i", out_wav,
        "-threads", str(n_threads),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_mp4,
    ])
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)

    return out_mp4, out_wav
