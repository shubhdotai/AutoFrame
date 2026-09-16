"""
Generate a results.txt report in the same format as the original repository's
visualization step (see Columbia_test.py:295). One section per face track,
listing speaking segments above a configurable threshold.
"""

import json
import os

import numpy as np


def build_report(
    video_path,
    tracks,
    scores,
    fps=25,
    threshold=0.0,
    smoothing_window=5,
    frame_width=None,
):
    half = smoothing_window // 2
    video_name = os.path.basename(video_path)
    report = {
        "video": video_name,
        "threshold": float(threshold),
        "smoothing_window": int(smoothing_window),
        "track_count": len(tracks),
        "tracks": [],
    }

    for tidx, (track, raw_score) in enumerate(zip(tracks, scores)):
        frames = track["frame"].tolist()
        if len(frames) == 0 or len(raw_score) == 0:
            continue
        n = min(len(frames), len(raw_score))
        frames = frames[:n]
        raw_score = np.asarray(raw_score[:n])

        smoothed = np.array([
            float(np.mean(raw_score[max(i - half, 0): min(i + half + 1, n)]))
            for i in range(n)
        ])
        bboxes = np.array(track["bbox"][:n])

        avg_bbox = bboxes.mean(axis=0)
        first_bbox = bboxes[0]
        avg_cx = (avg_bbox[0] + avg_bbox[2]) / 2
        avg_cy = (avg_bbox[1] + avg_bbox[3]) / 2
        position = None
        if frame_width is not None:
            position = "left" if avg_cx < frame_width / 2 else "right"

        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        segments = []
        in_seg = False
        seg_start = 0
        for i, sc in enumerate(smoothed):
            if sc >= threshold and not in_seg:
                in_seg = True
                seg_start = i
            elif sc < threshold and in_seg:
                in_seg = False
                sf = int(frames[seg_start])
                ef = int(frames[i - 1])
                segments.append({
                    "start_frame": sf,
                    "end_frame": ef,
                    "start_time_s": round(sf / fps, 3),
                    "end_time_s": round((ef + 1) / fps, 3),
                })
        if in_seg:
            sf = int(frames[seg_start])
            ef = int(frames[n - 1])
            segments.append({
                "start_frame": sf,
                "end_frame": ef,
                "start_time_s": round(sf / fps, 3),
                "end_time_s": round((ef + 1) / fps, 3),
            })

        report["tracks"].append({
            "track_id": tidx,
            "position": position,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": round(start_frame / fps, 3),
            "end_time_s": round((end_frame + 1) / fps, 3),
            "frame_count": n,
            "avg_score": float(np.mean(smoothed)),
            "max_score": float(np.max(smoothed)),
            "speaking_frames": int(np.sum(smoothed >= threshold)),
            "avg_center": [float(avg_cx), float(avg_cy)],
            "first_bbox": [float(v) for v in first_bbox],
            "avg_bbox": [float(v) for v in avg_bbox],
            "speaking_segments": segments,
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
            json.dump(report, f, indent=2)

    with open(out_path, "w") as f:
        f.write("LR-ASD score report (optimized pipeline)\n")
        f.write(f"video: {report['video']}\n")
        f.write(f"threshold: {threshold:.2f} (score >= threshold means speaking)\n")
        f.write(f"tracks: {report['track_count']}\n\n")

        for item in report["tracks"]:
            position = item["position"] or "?"
            f.write(
                f"track {item['track_id']:03d}: position {position}, "
                f"frames {item['start_frame']}-{item['end_frame']}, "
                f"time {item['start_time_s']:.2f}s-{item['end_time_s']:.2f}s, "
                f"avg_score {item['avg_score']:.2f}, "
                f"max_score {item['max_score']:.2f}, "
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
