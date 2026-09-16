"""Scene-aware face tracking with optimal per-frame assignment.

Three changes from the original greedy tracker.

Matching solves the whole frame at once instead of accepting the first box
that clears the IoU threshold. That first-match rule caused identity swaps
when two faces crossed: the wrong face inherits the track, and every
downstream stage -- ASD scoring, speaker selection, the crop -- then follows
the wrong person.

Matching is done against each track's *predicted* position rather than its
last seen one. Optimal assignment alone does not save a crossing, because at
the moment two faces overlap their boxes are nearly identical and both
pairings score the same. A constant-velocity prediction separates them again:
the track moving left is compared against where it was heading.

And tracks are built in a single forward pass per scene rather than by
rescanning the scene once per track, so cost is linear in detections instead
of quadratic.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter + 1e-6)


def _iou_matrix(track_boxes, det_boxes):
    """Vectorized IoU between two sets of xyxy boxes."""
    t = np.asarray(track_boxes, dtype=float)[:, None, :]
    d = np.asarray(det_boxes, dtype=float)[None, :, :]
    x1 = np.maximum(t[..., 0], d[..., 0])
    y1 = np.maximum(t[..., 1], d[..., 1])
    x2 = np.minimum(t[..., 2], d[..., 2])
    y2 = np.minimum(t[..., 3], d[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    return inter / (area_t + area_d - inter + 1e-6)


def _predict(track, frame_idx, max_extrapolation=12):
    """
    Where this track's box should be at `frame_idx`, at constant velocity.

    Velocity comes from the two most recent detections. Extrapolation is
    capped so a long gap cannot fling the predicted box across the frame.
    """
    box = track["box"]
    prev = track.get("prev")
    if prev is None:
        return box
    prev_box, prev_frame = prev
    span = track["last_frame"] - prev_frame
    if span <= 0:
        return box
    step = min(frame_idx - track["last_frame"], max_extrapolation)
    if step <= 0:
        return box
    scale = step / float(span)
    return [b + (b - p) * scale for b, p in zip(box, prev_box)]


def _scene_ranges(total_frames, scene_cuts=None):
    cuts = sorted({int(c) for c in (scene_cuts or []) if 0 < int(c) < total_frames})
    starts = [0, *cuts]
    ends = [*cuts, total_frames]
    return [(s, e) for s, e in zip(starts, ends) if e > s]


def _densify(track, min_face_size):
    """Interpolate a sparse detection list onto every frame it spans."""
    frame_nums = np.array([f["frame"] for f in track])
    bboxes = np.array([np.asarray(f["bbox"], dtype=float) for f in track])

    frame_i = np.arange(frame_nums[0], frame_nums[-1] + 1)
    if len(frame_nums) == 1:
        bboxes_i = np.repeat(bboxes, len(frame_i), axis=0)
    else:
        bboxes_i = np.stack(
            [interp1d(frame_nums, bboxes[:, j])(frame_i) for j in range(4)],
            axis=1,
        )

    avg_size = max(
        np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
        np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]),
    )
    if avg_size <= min_face_size:
        return None
    mean_conf = float(np.mean([f.get("conf", 1.0) for f in track]))
    return {"frame": frame_i, "bbox": bboxes_i, "mean_conf": mean_conf}


def _track_faces_in_range(
    detections_per_frame,
    start_frame,
    end_frame,
    num_failed_det=10,
    min_track=10,
    min_face_size=1,
    iou_threshold=0.5,
    min_confidence=0.0,
):
    open_tracks = []      # list of {"dets": [...], "last_frame": int, "box": [...]}
    finished = []

    for frame_idx in range(start_frame, end_frame):
        dets = [
            d for d in detections_per_frame[frame_idx]
            if float(d.get("conf", 1.0)) >= min_confidence
        ]

        # Retire tracks that have gone unseen for too long.
        still_open = []
        for tr in open_tracks:
            if frame_idx - tr["last_frame"] > num_failed_det:
                finished.append(tr)
            else:
                still_open.append(tr)
        open_tracks = still_open

        if not dets:
            continue

        if open_tracks:
            costs = _iou_matrix([_predict(tr, frame_idx) for tr in open_tracks],
                                [d["bbox"] for d in dets])
            # Hungarian on -IoU: the globally best pairing for this frame,
            # not whichever pair happens to be examined first.
            rows, cols = linear_sum_assignment(-costs)
            matched_dets = set()
            for r, c in zip(rows, cols):
                if costs[r, c] <= iou_threshold:
                    continue
                tr = open_tracks[r]
                tr["dets"].append(dets[c])
                tr["prev"] = (tr["box"], tr["last_frame"])
                tr["last_frame"] = frame_idx
                tr["box"] = dets[c]["bbox"]
                matched_dets.add(c)
            unmatched = [d for i, d in enumerate(dets) if i not in matched_dets]
        else:
            unmatched = dets

        for det in unmatched:
            open_tracks.append({
                "dets": [det], "last_frame": frame_idx, "box": det["bbox"],
                "prev": None,
            })

    finished.extend(open_tracks)

    tracks = []
    for tr in finished:
        if len(tr["dets"]) <= min_track:
            continue
        dense = _densify(tr["dets"], min_face_size)
        if dense is not None:
            tracks.append(dense)

    tracks.sort(key=lambda t: int(t["frame"][0]))
    return tracks


def track_faces(
    detections_per_frame,
    num_failed_det=10,
    min_track=10,
    min_face_size=1,
    iou_threshold=0.5,
    scene_cuts=None,
    min_confidence=0.0,
):
    """
    detections_per_frame: list[list[face_dict]]; each face has frame, bbox, conf.
    scene_cuts: optional first-frame indices of new scenes.
    min_confidence: drop detections the detector was unsure about, before they
        can seed a track. Low-confidence false positives otherwise become
        tracks, get scored, and can steal the crop.

    Returns tracks with contiguous frame arrays and interpolated bboxes.
    Tracking resets at every scene cut so identities never bridge unrelated
    shots. The input list is not modified.
    """
    total_frames = len(detections_per_frame)
    tracks = []
    for start_frame, end_frame in _scene_ranges(total_frames, scene_cuts):
        scene_tracks = _track_faces_in_range(
            detections_per_frame=detections_per_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            num_failed_det=num_failed_det,
            min_track=min_track,
            min_face_size=min_face_size,
            iou_threshold=iou_threshold,
            min_confidence=min_confidence,
        )
        for track in scene_tracks:
            track["scene_start_frame"] = int(start_frame)
            track["scene_end_frame"] = int(end_frame - 1)
        tracks.extend(scene_tracks)
    return tracks
