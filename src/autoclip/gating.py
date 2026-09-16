"""
Decide which tracks actually need active-speaker scoring.

The reframer's only use for an ASD score is to choose between faces that are
on screen at the same time. Read `shortform._per_frame_intent`: when a single
track is live in a frame, the speaker branch, the hold branch and the
largest-face branch all resolve to the same centre, so the score cannot move
the crop. The same is true when several faces are close enough together that
every candidate yields the same crop rectangle -- with a 9:16 window over a
16:9 frame the crop is more than half the width, so that happens often.

Measured on the validation clip: 463 of 3674 frames have two or more faces,
and only those frames can be influenced at all. Scoring every track regardless
spends the overwhelming majority of the ASD budget on frames whose output is
already determined.

Tracks that are skipped get NaN scores rather than zeros. NaN compares false
against any threshold, so a skipped track is never reported as speaking, and
it is still fully available to the crop as a visible subject.
"""

import numpy as np


def contested_frames(tracks, n_frames, video_width, crop_width, min_shift_px=8):
    """
    Boolean mask of frames where the choice of speaker moves the crop.

    A frame is contested when at least two tracks are live and the crop
    rectangles implied by their centres differ by more than `min_shift_px`.
    """
    mask = np.zeros(int(n_frames), dtype=bool)
    if crop_width >= video_width or len(tracks) < 2:
        return mask

    half = crop_width / 2.0
    max_x1 = float(video_width - crop_width)
    lo = np.full(int(n_frames), np.inf, dtype=np.float64)
    hi = np.full(int(n_frames), -np.inf, dtype=np.float64)
    count = np.zeros(int(n_frames), dtype=np.int32)

    for track in tracks:
        frames = np.asarray(track["frame"], dtype=np.int64)
        bboxes = np.asarray(track["bbox"], dtype=np.float64)
        n = min(len(frames), len(bboxes))
        if n == 0:
            continue
        frames = frames[:n]
        keep = (frames >= 0) & (frames < n_frames)
        if not keep.any():
            continue
        frames = frames[keep]
        cx = 0.5 * (bboxes[:n, 0] + bboxes[:n, 2])[keep]
        x1 = np.clip(np.round(np.clip(cx, half, video_width - half) - half), 0, max_x1)
        np.minimum.at(lo, frames, x1)
        np.maximum.at(hi, frames, x1)
        np.add.at(count, frames, 1)

    np.greater(hi - lo, float(min_shift_px), out=mask, where=count >= 2)
    return mask


def speech_mask(audio, sample_rate, n_frames, fps=25, floor_db=-45.0):
    """
    Per-frame mask of frames with any audible signal.

    LR-ASD will happily emit a positive logit over silence. A track that is
    entirely below the floor contains no speech to detect, so scoring it is
    both wasted work and a source of false positives. The floor is deliberately
    conservative: it is a silence test, not a voice-activity classifier, so it
    never suppresses quiet speech.
    """
    n_frames = int(n_frames)
    if n_frames <= 0 or audio is None or len(audio) == 0:
        return np.ones(max(n_frames, 0), dtype=bool)

    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    hop = max(1, int(round(sample_rate / float(fps))))
    usable = min(n_frames, len(audio) // hop)
    mask = np.ones(n_frames, dtype=bool)
    if usable <= 0:
        return mask

    block = audio[:usable * hop].reshape(usable, hop)
    rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-12)
    mask[:usable] = db >= float(floor_db)
    return mask


def select_tracks_for_asd(
    tracks,
    n_frames,
    video_width,
    crop_width,
    min_shift_px=8,
    speech=None,
    require_contested=True,
):
    """
    Return (selected_indices, reasons) where reasons[i] explains any skip.

    A track is scored when it overlaps a contested frame that also carries
    audible audio. Both conditions are necessary for the score to change the
    rendered result.
    """
    n_frames = int(n_frames)
    contested = (
        contested_frames(tracks, n_frames, video_width, crop_width, min_shift_px)
        if require_contested
        else np.ones(n_frames, dtype=bool)
    )

    selected = []
    reasons = {}
    for idx, track in enumerate(tracks):
        frames = np.asarray(track["frame"], dtype=np.int64)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        if len(frames) == 0:
            reasons[idx] = "empty"
            continue
        if not contested[frames].any():
            reasons[idx] = "uncontested"
            continue
        if speech is not None and not speech[frames].any():
            reasons[idx] = "silent"
            continue
        selected.append(idx)
    return selected, reasons
