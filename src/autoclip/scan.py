"""
One decode pass that produces both scene cuts and face detections.

The pipeline used to walk the normalized video twice: once for PySceneDetect
and once for face detection. Both stages need the same consecutive frames, so
they now share a single `cv2.VideoCapture` loop.

The loop is also a three-stage pipeline. Decoding runs ~1150 fps and detection
runs at a comparable rate, so running them in lockstep wastes roughly half the
wall clock. Scene detection and face detection each get a worker thread with a
bounded queue; the reader only blocks when a consumer falls behind, which caps
memory at a couple of in-flight batches.
"""

import queue
import threading
import time

import cv2

# PySceneDetect's own default: frames are downscaled to ~256 px wide before
# the content measure runs. Matching it keeps threshold semantics unchanged.
_SCENE_MIN_WIDTH = 256

# Cap on frames held in flight for detection, so 4K inputs do not balloon.
_MAX_INFLIGHT_BYTES = 192 << 20


def _build_detector(backend, width, height, **kwargs):
    """Pass through only the keys a given backend understands."""
    if backend == "apple_vision":
        from .face_detection import build_detector
        allowed = {"num_workers", "detect_size", "min_confidence",
                   "min_face_size", "batch_size"}
    elif backend == "yolo_face_person":
        from .face_detection_yolo import build_detector
        allowed = {"weights_path", "device", "conf", "iou", "imgsz",
                   "batch_size", "min_confidence", "min_face_size", "progress"}
    else:
        raise ValueError(f"unknown face detector backend: {backend}")
    return build_detector(width, height, **{k: v for k, v in kwargs.items() if k in allowed})


class _SceneTracker:
    """Incremental wrapper around a PySceneDetect detector."""

    def __init__(self, width, threshold=27.0, min_scene_len=15, mode="content"):
        try:
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
        except ImportError as exc:
            raise RuntimeError(
                "PySceneDetect is required for scene-aware tracking. Install it with "
                "`python -m pip install scenedetect`."
            ) from exc

        if mode == "adaptive":
            # Adaptive compares each frame against a rolling window, which is
            # far less trigger-happy on fast motion and camera flashes than a
            # fixed threshold. Over-segmentation is expensive downstream: it
            # chops tracks short, which is exactly where ASD is weakest.
            self._detector = AdaptiveDetector(
                adaptive_threshold=3.0, min_scene_len=min_scene_len
            )
        elif mode == "content":
            self._detector = ContentDetector(
                threshold=threshold, min_scene_len=min_scene_len
            )
        else:
            raise ValueError(f"unknown scene detector mode: {mode}")

        self.factor = max(1.0, width / float(_SCENE_MIN_WIDTH))
        self.cuts = []
        self._last_frame = 0

    def shrink(self, frame):
        if self.factor <= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, round(frame.shape[1] / self.factor)),
             max(1, round(frame.shape[0] / self.factor))),
            interpolation=cv2.INTER_LINEAR,
        )

    def process(self, frame_num, small_frame):
        self._last_frame = frame_num
        self.cuts.extend(self._detector.process_frame(frame_num, small_frame))

    def finish(self):
        self.cuts.extend(self._detector.post_process(self._last_frame))
        return sorted({int(c) for c in self.cuts if c > 0})


class _Worker(threading.Thread):
    """Single consumer thread over a bounded queue, preserving submit order."""

    def __init__(self, handler, maxsize):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=maxsize)
        self._handler = handler
        self.error = None

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if self.error is None:
                try:
                    self._handler(item)
                except BaseException as exc:  # re-raised on the reader thread
                    self.error = exc

    def submit(self, item):
        if self.error is not None:
            raise self.error
        self.queue.put(item)

    def finish(self):
        self.queue.put(None)
        self.join()
        if self.error is not None:
            raise self.error


def scenes_from_cuts(cuts, total_frames, fps):
    scenes = []
    starts = [0, *cuts]
    ends = [*cuts, total_frames]
    index = 0
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        scenes.append({
            "index": index,
            "start_frame": int(start),
            "end_frame": int(end - 1),
            "start_time_s": round(start / fps, 3) if fps else 0.0,
            "end_time_s": round(end / fps, 3) if fps else 0.0,
        })
        index += 1
    return scenes


def scan_video(
    video_path,
    backend="apple_vision",
    sample_every_n=2,
    batch_size=32,
    progress=True,
    detect_scenes=True,
    scene_threshold=27.0,
    min_scene_len=15,
    scene_mode="content",
    expected_frames=None,
    fps=None,
    **detector_kwargs
):
    """
    Returns:
        {
          "fps", "width", "height", "total_frames", "sample_every_n",
          "detections_per_frame": list[list[face]],   # length == total_frames
          "persons_per_frame":    list[list[person]] | None,
          "cuts":   list[int],   # first frame of each scene after scene 0
          "scenes": list[dict],
        }
    Frames that were not sampled carry an empty detection list.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    hint = int(expected_frames or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = _build_detector(backend, width, height, batch_size=batch_size,
                               progress=progress, **detector_kwargs)
    detect_batch = max(1, int(getattr(detector, "batch_size", batch_size)))
    frame_bytes = max(1, width * height * 3)
    inflight = max(1, _MAX_INFLIGHT_BYTES // (frame_bytes * detect_batch))

    detections_per_frame = []
    persons_per_frame = [] if getattr(detector, "provides_persons", False) else None

    def on_detect(item):
        frames, indices = item
        faces, persons = detector.detect(frames, indices)
        for offset, idx in enumerate(indices):
            detections_per_frame[idx] = faces[offset]
            if persons_per_frame is not None and persons is not None:
                persons_per_frame[idx] = persons[offset]

    scene_tracker = (
        _SceneTracker(width, scene_threshold, min_scene_len, scene_mode)
        if detect_scenes else None
    )

    def on_scene(item):
        scene_tracker.process(item[0], item[1])

    detect_worker = _Worker(on_detect, maxsize=inflight)
    detect_worker.start()
    scene_worker = None
    if scene_tracker is not None:
        scene_worker = _Worker(on_scene, maxsize=64)
        scene_worker.start()

    if progress:
        duration = hint / src_fps / 60 if src_fps and hint else 0
        print(f"[scan] {width}x{height} @ {src_fps:.2f} fps | ~{hint} frames "
              f"({duration:.1f} min)")
        print(f"[scan] detector={backend} | sampling every {sample_every_n} frame(s) | "
              f"batch={detect_batch} | scenes={scene_mode if detect_scenes else 'off'}")

    pending_frames, pending_indices = [], []
    t_start = time.perf_counter()
    last_report = t_start
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections_per_frame.append([])
            if persons_per_frame is not None:
                persons_per_frame.append([])

            if scene_worker is not None:
                # Downscale on the reader thread: it is a few hundred
                # microseconds and it keeps the queued frames small.
                scene_worker.submit((frame_idx, scene_tracker.shrink(frame)))

            if frame_idx % sample_every_n == 0:
                pending_frames.append(frame)
                pending_indices.append(frame_idx)
                if len(pending_frames) >= detect_batch:
                    detect_worker.submit((pending_frames, pending_indices))
                    pending_frames, pending_indices = [], []

            frame_idx += 1
            if progress and time.perf_counter() - last_report > 5.0:
                last_report = time.perf_counter()
                elapsed = last_report - t_start
                rate = frame_idx / max(elapsed, 1e-6)
                total = max(hint, frame_idx)
                eta = (total - frame_idx) / max(rate, 1e-6) / 60
                print(f"  {frame_idx}/{total} ({frame_idx / max(total, 1) * 100:.1f}%) | "
                      f"{rate:.1f} fps | ETA {eta:.1f} min")

        if pending_frames:
            detect_worker.submit((pending_frames, pending_indices))
    finally:
        cap.release()
        try:
            detect_worker.finish()
            if scene_worker is not None:
                scene_worker.finish()
        finally:
            detector.close()

    total_frames = frame_idx
    cuts = scene_tracker.finish() if scene_tracker is not None else []
    cuts = [c for c in cuts if 0 < c < total_frames]
    scenes = scenes_from_cuts(cuts, total_frames, src_fps)

    if progress:
        elapsed = time.perf_counter() - t_start
        n_faces = sum(len(d) for d in detections_per_frame)
        print(f"[scan] done in {elapsed / 60:.2f} min | {total_frames} frames | "
              f"{n_faces} face detections | {len(cuts)} cuts")

    return {
        "fps": src_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "sample_every_n": int(sample_every_n),
        "detections_per_frame": detections_per_frame,
        "persons_per_frame": persons_per_frame,
        "cuts": cuts,
        "scenes": scenes,
    }
