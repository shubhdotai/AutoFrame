"""Scene detection with PySceneDetect's AdaptiveDetector."""

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector


class SceneDetector:
    def __init__(self, min_scene_len=15):
        self.min_scene_len = min_scene_len

    def detect(self, video_path):
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(min_scene_len=self.min_scene_len))
        sm.detect_scenes(video=video, show_progress=True)
        scene_list = sm.get_scene_list()

        scenes = []
        for idx, (start, end) in enumerate(scene_list):
            s, e = int(start.get_frames()), int(end.get_frames())
            if e > s:
                scenes.append({"index": idx, "start_frame": s, "end_frame": e - 1})

        if not scenes:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            scenes = [{"index": 0, "start_frame": 0, "end_frame": max(0, total - 1)}]

        cuts = [s["start_frame"] for s in scenes[1:]]
        return scenes, cuts
