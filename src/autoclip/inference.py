"""
Optimized LR-ASD inference for long videos.

Key optimizations vs the original `evaluate_network` in Columbia_test.py:

1. STREAMING READER. Face video frames are read one at a time and pushed
   straight into the encoder window. Peak RAM is bounded by
   `encoder_window_frames * 112 * 112` (~5 MB for float32 input alone) instead of the entire ~1.1 GB
   per-track tensor.

2. ENCODER-OUTPUT CACHE. The visual + audio encoders run ONCE over the full
   track in non-overlapping windows, producing per-frame 128-d embeddings
   (~46 MB for 60 minutes). The detector + classifier then replay over those
   cached embeddings for each duration in `duration_set`. Original code
   re-runs the encoders for each unique duration.

3. REDUCED durationSet. Default is (2, 4, 6) instead of six unique durations.

4. REAL BATCHING. Detector chunks are stacked into a real batch
   (`detector_batch` chunks at a time) so the GPU gets utilized.

The score returned is the raw `lossAV` class-1 speaking logit,
matching what the original Columbia_test.py uses for visualization. Higher =
more confidently speaking.
"""

import math
import time

import cv2
import numpy as np
import torch
from scipy.io import wavfile
import python_speech_features


from .ASD import ASD


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path, device="mps"):
    """Instantiate ASD on the requested device, load weights, set eval()."""
    s = ASD(device=device)
    s.loadParameters(weights_path)
    s.eval()
    return s


# ---------------------------------------------------------------------------
# Streaming pre-processing
# ---------------------------------------------------------------------------

def _read_face_window(cap, n):
    """Read up to n frames; return (T, 112, 112) float32 array or None."""
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray[56:168, 56:168]  # center crop to 112x112
        frames.append(gray)
    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Encoder pass — runs ONCE per track, caches per-frame embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(s, video_path, audio_path, device="mps", encoder_window_frames=100):
    """
    Run the audio + visual encoders over the entire track in non-overlapping
    windows of `encoder_window_frames` (default 100 frames = 4 s).

    Returns (audio_embed, visual_embed), both torch tensors on `device` of
    shape (T, 128). Each row is the per-frame embedding from the corresponding
    encoder.
    """
    # ---- audio: full-track MFCC up front (it's small: ~18 MB / 60 min) ----
    if encoder_window_frames <= 0 or encoder_window_frames % 4:
        raise ValueError("encoder_window_frames must be a positive multiple of 4")
    sr, audio = wavfile.read(audio_path)
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("ASD requires 16 kHz mono audio")
    mfcc = python_speech_features.mfcc(
        audio, 16000, numcep=13, winlen=0.025, winstep=0.010
    ).astype(np.float32)
    # 100 audio frames per second; 25 video fps -> 4 audio frames per video frame.
    cap = cv2.VideoCapture(video_path)
    n_video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_video = min(n_video_total, mfcc.shape[0] // 4)

    # Trim audio to a clean multiple of 4 video frames
    mfcc = mfcc[: n_video * 4]
    audio_t = torch.from_numpy(mfcc).to(device)  # (n_video*4, 13)

    visual_embeds = []
    audio_embeds = []
    frames_done = 0

    while frames_done < n_video:
        remaining = n_video - frames_done
        take = min(encoder_window_frames, remaining)
        v_np = _read_face_window(cap, take)
        if v_np is None:
            break

        v_t = torch.from_numpy(v_np).unsqueeze(0).to(device)  # (1, T, 112, 112)
        a_slice = audio_t[frames_done * 4 : (frames_done + v_t.shape[1]) * 4]
        a_t = a_slice.unsqueeze(0)  # (1, T_a, 13)

        ev = s.model.forward_visual_frontend(v_t)   # (1, T, 128)
        ea = s.model.forward_audio_frontend(a_t)    # (1, ~T, 128) — pooled to video fps

        T_min = min(ev.shape[1], ea.shape[1])
        visual_embeds.append(ev[0, :T_min])
        audio_embeds.append(ea[0, :T_min])
        frames_done += v_t.shape[1]

    cap.release()

    if not visual_embeds:
        return None, None

    visual_embed = torch.cat(visual_embeds, dim=0)   # (T, 128)
    audio_embed = torch.cat(audio_embeds, dim=0)
    T = min(visual_embed.shape[0], audio_embed.shape[0])
    return audio_embed[:T], visual_embed[:T]


# ---------------------------------------------------------------------------
# Detector replay — cheap, runs once per duration in duration_set
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_detector(
    s,
    audio_embed,
    visual_embed,
    duration_set=(2, 4, 6),
    detector_batch=64,
):
    """
    Replay the fusion + detector + classifier over the cached embeddings at
    multiple chunk durations and average the resulting per-frame scores.

    Returns: np.ndarray (T,) of speaking logit scores.
    """
    if not duration_set or any(d <= 0 for d in duration_set) or detector_batch <= 0:
        raise ValueError("durations and detector_batch must be positive")
    T = audio_embed.shape[0]
    C = audio_embed.shape[1]
    if T == 0:
        return np.array([], dtype=np.float32)

    all_scores = []

    for duration in duration_set:
        chunk = max(1, duration * 25)
        n_chunks = math.ceil(T / chunk)

        out_scores = np.empty(n_chunks * chunk, dtype=np.float32)
        for b0 in range(0, n_chunks, detector_batch):
            b1 = min(b0 + detector_batch, n_chunks)
            a_batch = audio_embed.new_zeros(b1 - b0, chunk, C)
            v_batch = visual_embed.new_zeros(b1 - b0, chunk, C)
            for i in range(b0, b1):
                start = i * chunk
                end = min(start + chunk, T)
                a_batch[i - b0, :end-start] = audio_embed[start:end]
                v_batch[i - b0, :end-start] = visual_embed[start:end]
            out = s.model.forward_audio_visual_backend(a_batch, v_batch)
            scores = s.lossAV.forward(out, labels=None)
            out_scores[b0 * chunk : b1 * chunk] = np.asarray(scores, dtype=np.float32)

        # Reshape, drop padding tail, append
        per_frame = out_scores.reshape(n_chunks, chunk).reshape(-1)[:T]
        all_scores.append(per_frame)

    return np.mean(np.stack(all_scores, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_asd_on_track(
    s,
    video_path,
    audio_path,
    duration_set=(2, 4, 6),
    encoder_window_frames=100,
    detector_batch=64,
    device="mps",
    progress_label=None,
):
    """
    End-to-end inference for one cropped track. Returns a 1-D numpy array of
    per-frame speaking-logit scores (length matches the track's video frame
    count after audio/video alignment).
    """
    t0 = time.perf_counter()
    audio_embed, visual_embed = compute_embeddings(
        s,
        video_path=video_path,
        audio_path=audio_path,
        device=device,
        encoder_window_frames=encoder_window_frames,
    )
    if audio_embed is None or audio_embed.shape[0] == 0:
        return np.array([], dtype=np.float32)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = score_detector(
        s,
        audio_embed,
        visual_embed,
        duration_set=duration_set,
        detector_batch=detector_batch,
    )
    t_det = time.perf_counter() - t0

    if progress_label is not None:
        T = scores.shape[0]
        print(
            f"[asd] {progress_label}: T={T} frames "
            f"({T/25:.1f}s) | encoder {t_enc:.1f}s | "
            f"detector ×{len(duration_set)} {t_det:.1f}s"
        )

    return scores
