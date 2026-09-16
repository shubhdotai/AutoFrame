"""Scene-aware IoU face tracking with bbox interpolation."""

import numpy as np
from scipy.interpolate import interp1d


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter + 1e-6)


def _scene_ranges(total_frames, scene_cuts=None):
    cuts = sorted({int(c) for c in (scene_cuts or []) if 0 < int(c) < total_frames})
    starts = [0, *cuts]
    ends = [*cuts, total_frames]
    return [(s, e) for s, e in zip(starts, ends) if e > s]


def _track_faces_in_range(
    detections_per_frame,
    start_frame,
    end_frame,
    num_failed_det=10,
    min_track=10,
    min_face_size=1,
    iou_threshold=0.5,
):
    scene_faces = [list(faces) for faces in detections_per_frame[start_frame:end_frame]]
    tracks = []

    while True:
        track = []
        for frame_faces in scene_faces:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                    break
                elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                    if _iou(face["bbox"], track[-1]["bbox"]) > iou_threshold:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                else:
                    break

        if not track:
            break

        if len(track) > min_track:
            frame_nums = np.array([f["frame"] for f in track])
            bboxes = np.array([np.asarray(f["bbox"], dtype=float) for f in track])

            frame_i = np.arange(frame_nums[0], frame_nums[-1] + 1)
            bboxes_i = np.stack(
                [interp1d(frame_nums, bboxes[:, j])(frame_i) for j in range(4)],
                axis=1,
            )

            avg_size = max(
                np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]),
            )
            if avg_size > min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks


def track_faces(
    detections_per_frame,
    num_failed_det=10,
    min_track=10,
    min_face_size=1,
    iou_threshold=0.5,
    scene_cuts=None,
):
    """
    detections_per_frame: list[list[face_dict]]; each face has frame, bbox, conf.
    scene_cuts: optional first-frame indices of new scenes.

    Returns tracks with contiguous frame arrays and interpolated bboxes. Tracking
    is reset at every scene cut so identities never bridge unrelated shots.
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
        )
        for track in scene_tracks:
            track["scene_start_frame"] = int(start_frame)
            track["scene_end_frame"] = int(end_frame - 1)
        tracks.extend(scene_tracks)
    return tracks
