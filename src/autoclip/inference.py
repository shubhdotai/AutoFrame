"""
Optimized LR-ASD inference for long videos.

Design notes, and how this differs from `evaluate_network` in the original
Columbia_test.py:

1. NO MEDIA ROUND-TRIP. Face crops are pushed in as uint8 arrays straight from
   the decode pass. The original wrote one MP4 per track and read it back;
   that cost an encode, a decode, and -- because the intermediate was `mp4v`
   -- fed the network lossily compressed pixels.

2. STREAMING, BATCHED ENCODER. Windows from many tracks are stacked into one
   real batch. At batch size 1 the visual encoder is latency-bound on short
   tracks, which is most tracks once scene cuts chop them up.

3. EXACT WINDOW MARGINS. The encoders are convolutional over time with a
   combined receptive-field radius of 9 frames. Non-overlapping windows
   therefore corrupt ~9 frames at every boundary. Windows now carry a margin
   of `_MARGIN` frames on each side which is computed and discarded, so the
   result matches a single unbroken pass over the track.

4. ENCODER-OUTPUT CACHE. Encoders run once per track; the detector replays
   over the cached 128-d embeddings for each duration.

5. NO ZERO-PADDED CHUNKS. The detector's last chunk used to be zero-filled to
   a whole `duration * 25` frames -- on short tracks that meant a bidirectional
   GRU reading mostly zeros (79% of the sequence for a 6 s duration against a
   median 31-frame track). Durations are now clamped to the track length and
   the tail chunk slides back to fit, so the GRU only ever sees real frames.

The score returned is the raw `lossAV` class-1 speaking logit, matching what
the original uses for visualization. Higher = more confidently speaking.
"""

import time

import cv2
import numpy as np
import torch
from scipy.io import wavfile
import python_speech_features


from .ASD import ASD

# Combined temporal receptive-field radius of the three encoder blocks
# (kernels 5 and 3 per block, three blocks: 2+1 + 2+1 + 2+1).
_MARGIN = 9

# Audio frames per video frame at 25 fps / 100 Hz MFCC.
_AUDIO_PER_VIDEO = 4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path, device="mps"):
    """Instantiate ASD on the requested device, load weights, set eval()."""
    s = ASD(device=device)
    s.loadParameters(weights_path)
    s.eval()
    return s


def compute_mfcc(audio_path):
    """Full-track MFCC at 100 Hz. Small enough to hold whole: ~18 MB/hour."""
    sr, audio = wavfile.read(audio_path)
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("ASD requires 16 kHz mono audio")
    return python_speech_features.mfcc(
        audio, 16000, numcep=13, winlen=0.025, winstep=0.010
    ).astype(np.float32)


def prepare_face(frame_bgr, out_size=112):
    """
    BGR crop -> the grayscale 112x112 tensor the visual encoder expects.

    The original wrote 224x224 crops and then used only `[56:168, 56:168]`,
    so three quarters of every written pixel was discarded after a full
    encode/decode round-trip. Taking the centre half directly is the same
    region at a quarter of the pixel work.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    y0, x0 = h // 4, w // 4
    centre = gray[y0:y0 + (h - 2 * y0), x0:x0 + (w - 2 * x0)]
    if centre.shape[:2] != (out_size, out_size):
        centre = cv2.resize(centre, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return centre


# ---------------------------------------------------------------------------
# Streaming batched encoder
# ---------------------------------------------------------------------------

class _TrackState:
    __slots__ = ("key", "n_frames", "mfcc", "buf", "buf_start", "emitted",
                 "visual", "audio", "closed", "received")

    def __init__(self, key, n_frames, mfcc):
        self.key = key
        self.n_frames = int(n_frames)
        self.mfcc = mfcc
        self.buf = []
        self.buf_start = 0
        self.emitted = 0
        self.received = 0
        self.closed = False
        self.visual = np.zeros((self.n_frames, 128), dtype=np.float32)
        self.audio = np.zeros((self.n_frames, 128), dtype=np.float32)


class EmbeddingEngine:
    """
    Push face crops in, get per-frame audio/visual embeddings out.

    Windows from every open track share one batch, so the GPU sees real work
    even when each individual track is only a second long.
    """

    def __init__(self, model, device="mps", window=100, batch=16, progress=False):
        if window <= 0 or window % 4:
            raise ValueError("window must be a positive multiple of 4")
        self.model = model
        self.device = device
        self.window = int(window)
        self.batch = max(1, int(batch))
        self.progress = progress
        self._tracks = {}
        self._pending = []          # (state, out_start, out_len, frames, left_pad)
        self.frames_encoded = 0
        self.windows_run = 0

    # -- track lifecycle ---------------------------------------------------

    def open_track(self, key, n_frames, mfcc):
        """mfcc: (n_frames * 4, 13) float32 slice aligned to this track."""
        if key in self._tracks:
            raise KeyError(f"track {key!r} is already open")
        state = _TrackState(key, n_frames, mfcc)
        self._tracks[key] = state
        return state

    def push(self, key, face112):
        state = self._tracks[key]
        if state.received >= state.n_frames:
            return
        state.buf.append(face112)
        state.received += 1
        # Emit as soon as the right-hand margin is available.
        while state.emitted + self.window + _MARGIN <= state.buf_start + len(state.buf):
            self._emit(state, self.window)

    def close_track(self, key):
        state = self._tracks[key]
        state.closed = True
        while state.emitted < state.received:
            self._emit(state, min(self.window, state.received - state.emitted))
        state.buf = []

    # -- window construction ----------------------------------------------

    def _emit(self, state, count):
        start = state.emitted
        lo = max(0, start - _MARGIN)
        hi = min(state.received, start + count + _MARGIN)
        frames = state.buf[lo - state.buf_start: hi - state.buf_start]
        self._pending.append((state, start, count, np.stack(frames, axis=0), start - lo))
        state.emitted = start + count

        # Drop buffered frames that no future window can reach.
        keep_from = max(0, state.emitted - _MARGIN)
        if keep_from > state.buf_start:
            state.buf = state.buf[keep_from - state.buf_start:]
            state.buf_start = keep_from

        if len(self._pending) >= self.batch:
            self._run_batch()

    @torch.no_grad()
    def _run_batch(self):
        """
        Encode every pending window, batching only windows of equal length.

        Padding a short window up to the batch's longest is not equivalent to
        running it alone: `forward_visual_frontend` subtracts a mean and
        divides by a std, so a zero-filled tail enters the convolutions as
        -2.465 rather than as the zero that conv padding would supply. Grouping
        by length keeps batched results identical to unbatched ones.
        """
        if not self._pending:
            return
        pending, self._pending = self._pending, []

        buckets = {}
        for item in pending:
            buckets.setdefault(item[3].shape[0], []).append(item)

        for length, group in buckets.items():
            v_batch = np.empty((len(group), length, 112, 112), dtype=np.float32)
            a_batch = np.zeros((len(group), length * _AUDIO_PER_VIDEO, 13), dtype=np.float32)
            for i, (state, out_start, _count, frames, left_pad) in enumerate(group):
                v_batch[i] = frames
                a_lo = (out_start - left_pad) * _AUDIO_PER_VIDEO
                a_slice = state.mfcc[a_lo: a_lo + length * _AUDIO_PER_VIDEO]
                a_batch[i, :a_slice.shape[0]] = a_slice

            v_t = torch.from_numpy(v_batch).to(self.device)
            a_t = torch.from_numpy(a_batch).to(self.device)
            ev = self.model.model.forward_visual_frontend(v_t)   # (B, T, 128)
            ea = self.model.model.forward_audio_frontend(a_t)    # (B, T, 128)
            ev_np = ev.float().cpu().numpy()
            ea_np = ea.float().cpu().numpy()

            for i, (state, out_start, count, _frames, left_pad) in enumerate(group):
                take = min(count, state.n_frames - out_start,
                           ev_np.shape[1] - left_pad, ea_np.shape[1] - left_pad)
                if take <= 0:
                    continue
                state.visual[out_start:out_start + take] = ev_np[i, left_pad:left_pad + take]
                state.audio[out_start:out_start + take] = ea_np[i, left_pad:left_pad + take]
                self.frames_encoded += take
            self.windows_run += len(group)

    # -- results -----------------------------------------------------------

    def finish(self):
        """Flush and return {key: (audio_embed, visual_embed)} as numpy arrays."""
        for key in list(self._tracks):
            if not self._tracks[key].closed:
                self.close_track(key)
        self._run_batch()
        out = {}
        for key, state in self._tracks.items():
            n = min(state.received, state.n_frames)
            out[key] = (state.audio[:n], state.visual[:n])
        return out


# ---------------------------------------------------------------------------
# Detector replay
# ---------------------------------------------------------------------------

def resolve_durations(duration_set, n_frames, fps=25):
    """
    Chunk lengths in frames, clamped to what the track can actually supply.

    A duration longer than the track used to become zero padding rather than
    context, which is worse than useless inside a bidirectional GRU.
    """
    if n_frames <= 0:
        return []
    chunks = sorted({min(max(1, int(round(d * fps))), int(n_frames))
                     for d in duration_set if d > 0})
    return chunks


@torch.no_grad()
def score_detector(
    s,
    audio_embed,
    visual_embed,
    duration_set=(2, 4, 6),
    detector_batch=64,
    fps=25,
):
    """
    Replay the fusion + detector + classifier over the cached embeddings at
    multiple chunk durations and average the resulting per-frame scores.

    Returns: np.ndarray (T,) of speaking logit scores.
    """
    if not duration_set or any(d <= 0 for d in duration_set) or detector_batch <= 0:
        raise ValueError("durations and detector_batch must be positive")

    device = next(s.parameters()).device
    if not torch.is_tensor(audio_embed):
        audio_embed = torch.from_numpy(np.asarray(audio_embed, dtype=np.float32))
    if not torch.is_tensor(visual_embed):
        visual_embed = torch.from_numpy(np.asarray(visual_embed, dtype=np.float32))
    audio_embed = audio_embed.to(device)
    visual_embed = visual_embed.to(device)

    T = int(audio_embed.shape[0])
    if T == 0:
        return np.array([], dtype=np.float32)

    all_scores = []
    for chunk in resolve_durations(duration_set, T, fps=fps):
        # Whole-length windows only: the tail window slides back to fit rather
        # than being zero-filled, so every frame is scored in real context.
        starts = list(range(0, T, chunk))
        if starts[-1] + chunk > T:
            starts[-1] = max(0, T - chunk)
        starts = sorted(set(starts))

        per_frame = torch.empty(T, dtype=torch.float32, device=device)
        for b0 in range(0, len(starts), detector_batch):
            batch_starts = starts[b0:b0 + detector_batch]
            index = torch.as_tensor(batch_starts, device=device)[:, None] + \
                torch.arange(chunk, device=device)[None, :]
            a_batch = audio_embed[index]        # (B, chunk, 128)
            v_batch = visual_embed[index]
            out = s.model.forward_audio_visual_backend(a_batch, v_batch)
            # Call the head directly: lossAV.forward would sync to numpy on
            # every batch, stalling the device once per iteration.
            logits = s.lossAV.FC(out.reshape(-1, out.shape[-1]))[:, 1]
            logits = logits.reshape(len(batch_starts), chunk)
            for i, start in enumerate(batch_starts):
                per_frame[start:start + chunk] = logits[i]
        all_scores.append(per_frame)

    if not all_scores:
        return np.array([], dtype=np.float32)
    return torch.stack(all_scores, dim=0).mean(dim=0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience entry point for a single track backed by files
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
    End-to-end inference for one cropped track stored as media files.

    The main pipeline avoids these files entirely; this remains for tools that
    already have per-track MP4/WAV pairs on disk.
    """
    t0 = time.perf_counter()
    mfcc = compute_mfcc(audio_path)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(prepare_face(frame))
    cap.release()

    n = min(len(frames), mfcc.shape[0] // _AUDIO_PER_VIDEO)
    if n == 0:
        return np.array([], dtype=np.float32)

    engine = EmbeddingEngine(s, device=device, window=encoder_window_frames, batch=1)
    engine.open_track(0, n, mfcc[: n * _AUDIO_PER_VIDEO])
    for frame in frames[:n]:
        engine.push(0, frame)
    engine.close_track(0)
    audio_embed, visual_embed = engine.finish()[0]
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = score_detector(
        s, audio_embed, visual_embed,
        duration_set=duration_set, detector_batch=detector_batch,
    )
    t_det = time.perf_counter() - t0

    if progress_label is not None:
        print(f"[asd] {progress_label}: T={scores.shape[0]} frames "
              f"({scores.shape[0] / 25:.1f}s) | encoder {t_enc:.1f}s | "
              f"detector x{len(resolve_durations(duration_set, n))} {t_det:.1f}s")
    return scores
