"""
Generate a results.txt report in the same format as the original repository's
visualization step (see Columbia_test.py:295). One section per face track,
listing speaking segments above a configurable threshold.

Tracks the pipeline chose not to score (see `gating`) carry NaN scores. They
are reported as `scored: false` with no speaking segments rather than being
silently treated as scoring zero, which at the default threshold of 0.0 would
have claimed every skipped track was speaking for its whole length.
"""

import json
import os

import numpy as np

from .shortform import _smooth_scores


def _segments_from_mask(mask, frames, fps):
    """Contiguous runs of True in `mask`, expressed in frame indices and times."""
    segments = []
    if mask.size == 0:
        return segments
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        sf = int(frames[start])
        ef = int(frames[end - 1])
        segments.append({
            "start_frame": sf,
            "end_frame": ef,
            "start_time_s": round(sf / fps, 3),
            "end_time_s": round((ef + 1) / fps, 3),
        })
    return segments


def build_report(
    video_path,
    tracks,
    scores,
    fps=25,
    threshold=0.0,
    smoothing_window=5,
    frame_width=None,
):
    video_name = os.path.basename(video_path)
    report = {
        "video": video_name,
        "threshold": float(threshold),
        "smoothing_window": int(smoothing_window),
        "track_count": len(tracks),
        "tracks": [],
    }

    for tidx, (track, raw_score) in enumerate(zip(tracks, scores)):
        frames = np.asarray(track["frame"], dtype=np.int64)
        raw_score = np.asarray(raw_score, dtype=np.float32)
        if len(frames) == 0 or len(raw_score) == 0:
            continue
        n = min(len(frames), len(raw_score))
        frames = frames[:n]
        raw_score = raw_score[:n]
        scored = bool(np.isfinite(raw_score).any())

        smoothed = _smooth_scores(raw_score, smoothing_window)
        bboxes = np.asarray(track["bbox"][:n], dtype=float)

        avg_bbox = bboxes.mean(axis=0)
        first_bbox = bboxes[0]
        avg_cx = (avg_bbox[0] + avg_bbox[2]) / 2
        avg_cy = (avg_bbox[1] + avg_bbox[3]) / 2
        position = None
        if frame_width is not None:
            position = "left" if avg_cx < frame_width / 2 else "right"

        speaking = np.isfinite(smoothed) & (smoothed >= threshold)
        # nanmean/nanmax so a NaN-padded tail cannot turn the whole summary --
        # and with it results.json -- into a non-finite value JSON cannot hold.
        finite = smoothed[np.isfinite(smoothed)]
        report["tracks"].append({
            "track_id": tidx,
            "scored": scored,
            "position": position,
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_time_s": round(int(frames[0]) / fps, 3),
            "end_time_s": round((int(frames[-1]) + 1) / fps, 3),
            "frame_count": int(n),
            "avg_score": float(finite.mean()) if finite.size else None,
            "max_score": float(finite.max()) if finite.size else None,
            "speaking_frames": int(speaking.sum()),
            "avg_center": [float(avg_cx), float(avg_cy)],
            "first_bbox": [float(v) for v in first_bbox],
            "avg_bbox": [float(v) for v in avg_bbox],
            "speaking_segments": _segments_from_mask(speaking, frames, fps),
        })

    return report


def write_report(
    out_path,
    video_path,
    tracks,
    scores,
    fps=25,
    threshold=0.0,
    smoothing_window=5,
    frame_width=None,
    json_path=None,
):
    """Write a text report and optionally the same data as JSON."""
    report = build_report(
        video_path=video_path,
        tracks=tracks,
        scores=scores,
        fps=fps,
        threshold=threshold,
        smoothing_window=smoothing_window,
        frame_width=frame_width,
    )
    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)

    unscored = sum(1 for t in report["tracks"] if not t["scored"])
    with open(out_path, "w") as f:
        f.write("LR-ASD score report (optimized pipeline)\n")
        f.write(f"video: {report['video']}\n")
        f.write(f"threshold: {threshold:.2f} (score >= threshold means speaking)\n")
        f.write(f"tracks: {report['track_count']}")
        if unscored:
            f.write(f" ({unscored} not scored: the speaker choice could not "
                    f"change the crop there)")
        f.write("\n\n")

        for item in report["tracks"]:
            position = item["position"] or "?"
            if item["scored"]:
                scores_text = (f"avg_score {item['avg_score']:.2f}, "
                               f"max_score {item['max_score']:.2f}, ")
            else:
                scores_text = "not scored, "
            f.write(
                f"track {item['track_id']:03d}: position {position}, "
                f"frames {item['start_frame']}-{item['end_frame']}, "
                f"time {item['start_time_s']:.2f}s-{item['end_time_s']:.2f}s, "
                f"{scores_text}"
                f"speaking_frames {item['speaking_frames']}/{item['frame_count']}\n"
            )
            avg_cx, avg_cy = item["avg_center"]
            f.write(f"  avg_center: x {avg_cx:.1f}, y {avg_cy:.1f}\n")
            first_bbox = item["first_bbox"]
            f.write(
                f"  first_bbox: x1 {first_bbox[0]:.1f}, y1 {first_bbox[1]:.1f}, "
                f"x2 {first_bbox[2]:.1f}, y2 {first_bbox[3]:.1f}\n"
            )
            avg_bbox = item["avg_bbox"]
            f.write(
                f"  avg_bbox: x1 {avg_bbox[0]:.1f}, y1 {avg_bbox[1]:.1f}, "
                f"x2 {avg_bbox[2]:.1f}, y2 {avg_bbox[3]:.1f}\n"
            )

            for segment in item["speaking_segments"]:
                f.write(
                    f"  speaking_segment: frames "
                    f"{segment['start_frame']}-{segment['end_frame']}, "
                    f"time {segment['start_time_s']:.2f}s-"
                    f"{segment['end_time_s']:.2f}s\n"
                )
            f.write("\n")
    return report
