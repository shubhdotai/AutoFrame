"""Scene detection helpers."""

import os


def detect_scenes(video_path, threshold=27.0, min_scene_len=15, progress=True):
    """
    Detect scene boundaries with PySceneDetect.

    Returns (scenes, cuts), where cuts are the first frame indices of each
    scene after scene 0.
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is required for scene-aware tracking. Install it with "
            "`python -m pip install scenedetect`."
        ) from exc

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video=video, show_progress=progress)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for idx, (start, end) in enumerate(scene_list):
        start_frame = int(start.get_frames())
        end_exclusive = int(end.get_frames())
        if end_exclusive <= start_frame:
            continue
        scenes.append({
            "index": idx,
            "start_frame": start_frame,
            "end_frame": end_exclusive - 1,
            "start_time_s": round(float(start.get_seconds()), 3),
            "end_time_s": round(float(end.get_seconds()), 3),
        })

    if not scenes:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        scenes = [{
            "index": 0,
            "start_frame": 0,
            "end_frame": max(0, total - 1),
            "start_time_s": 0.0,
            "end_time_s": round(total / fps, 3) if fps else 0.0,
        }]

    cuts = [scene["start_frame"] for scene in scenes[1:]]
    return scenes, cuts


def scene_json(video_path, scenes, cuts, threshold, min_scene_len):
    return {
        "source": os.path.basename(video_path),
        "detector": "PySceneDetect ContentDetector",
        "threshold": float(threshold),
        "min_scene_len": int(min_scene_len),
        "cuts": [int(c) for c in cuts],
        "scenes": scenes,
    }

