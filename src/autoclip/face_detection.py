"""
Apple Vision face detection (VNDetectFaceRectanglesRequest).

Streams the input video in chunks, samples 1-in-N frames, runs detection in a
thread pool, and returns a per-frame list of face dicts (empty for non-sampled
frames). Drop-in replacement for the S3FD `inference_video` step in the
original Columbia_test.py.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2

# pyobjc imports are deferred so this module is importable on non-macOS dev
# machines. The actual detection call below requires them.
_thread_local = threading.local()


def _get_request():
    import Vision  # noqa: F401  (deferred macOS dependency)
    if not hasattr(_thread_local, "request"):
        _thread_local.request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    return _thread_local.request


def _detect_one_frame(frame_idx, frame, width, height):
    import Vision  # noqa: F401
    from Foundation import NSData

    success, buf = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Could not encode frame for Apple Vision")
    data = buf.tobytes()
    ns_data = NSData.dataWithBytes_length_(data, len(data))

    request = _get_request()
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Apple Vision face detection failed: {error}")

    faces = []
    for obs in (request.results() or []):
        bb = obs.boundingBox()
        x = bb.origin.x * width
        y_bottom = bb.origin.y * height
        bw = bb.size.width * width
        bh = bb.size.height * height
        y_top = height - y_bottom - bh
        x1, y1 = int(x), int(y_top)
        x2, y2 = int(x + bw), int(y_top + bh)
        faces.append({
            "frame": frame_idx,
            "bbox": [x1, y1, x2, y2],
            "conf": float(obs.confidence()),
        })
    return frame_idx, faces


def detect_faces_in_video(
    video_path,
    sample_every_n=3,
    num_workers=8,
    chunk_size=32,
    progress=True,
):
    """
    Returns a dict:
        {
          "fps": float,
          "width": int, "height": int,
          "total_frames": int,
          "sample_every_n": int,
          "detections_per_frame": list[list[face_dict]]   # length == total_frames
        }
    Frames not sampled have an empty list.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if progress:
        dur_min = total_frames / fps / 60 if fps else 0
        print(f"[face_detection] {width}x{height} @ {fps:.2f} fps | "
              f"{total_frames} frames ({dur_min:.1f} min)")
        print(f"[face_detection] sampling every {sample_every_n} frame(s) | "
              f"workers={num_workers}")

    detections_per_frame = [[] for _ in range(total_frames)]
    t_start = time.perf_counter()
    frame_idx = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            chunk_frames = []
            chunk_start = frame_idx
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
                frame_idx += 1
            if not chunk_frames:
                break

            futures = []
            for i, f in enumerate(chunk_frames):
                gi = chunk_start + i
                if gi % sample_every_n == 0:
                    futures.append(executor.submit(_detect_one_frame, gi, f, width, height))

            for fut in futures:
                idx, faces = fut.result()
                detections_per_frame[idx] = faces

            if progress:
                elapsed = time.perf_counter() - t_start
                pct = frame_idx / total_frames * 100
                fps_a = frame_idx / max(elapsed, 1e-6)
                eta_min = (total_frames - frame_idx) / max(fps_a, 1e-6) / 60
                print(f"  {frame_idx}/{total_frames} ({pct:.1f}%) | "
                      f"{fps_a:.1f} fps | ETA {eta_min:.1f} min")

    cap.release()
    elapsed = time.perf_counter() - t_start
    if progress:
        n_sampled = len(range(0, frame_idx, sample_every_n))
        n_faces = sum(len(d) for d in detections_per_frame)
        print(f"[face_detection] done in {elapsed/60:.1f} min | "
              f"{n_sampled} sampled frames | {n_faces} face detections")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "sample_every_n": sample_every_n,
        "detections_per_frame": detections_per_frame,
    }
