"""
Apple Vision face detection (VNDetectFaceRectanglesRequest).

Frames reach Vision as a wrapped CVPixelBuffer rather than an encoded image.
The previous implementation PNG-encoded every sampled frame, which cost more
than the detection itself: measured on 1080x1080 frames, PNG round-trip was
66.5 ms/frame versus 7.0 ms/frame for a 720 px pixel buffer, with the same
faces found. Detection also runs on a downscaled copy, because Vision returns
normalized coordinates and those map back to full resolution for free.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_thread_local = threading.local()

# Long-edge resolution fed to Vision. Face recall was identical at 720 and
# 1080 on the validation clip; below ~540 small background faces start to drop.
DEFAULT_DETECT_SIZE = 720


def _get_request():
    import Vision  # deferred macOS dependency
    if not hasattr(_thread_local, "request"):
        _thread_local.request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    return _thread_local.request


def _quartz():
    try:
        import Quartz
        if hasattr(Quartz, "CVPixelBufferCreateWithBytes"):
            return Quartz
    except ImportError:
        pass
    return None


def _handler_for(frame):
    """
    Wrap a BGR frame for Vision, preferring a zero-copy pixel buffer.

    CVPixelBufferCreateWithBytes does not copy, so the BGRA array must outlive
    the request; it is returned alongside the handler and held by the caller.
    """
    import Vision
    quartz = _quartz()
    if quartz is not None:
        bgra = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
        height, width = bgra.shape[:2]
        status, pixel_buffer = quartz.CVPixelBufferCreateWithBytes(
            None, width, height, quartz.kCVPixelFormatType_32BGRA,
            bgra, width * 4, None, None, None, None,
        )
        if status == 0 and pixel_buffer is not None:
            handler = Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(
                pixel_buffer, None
            )
            return handler, bgra

    # Fallback for pyobjc builds without the CoreVideo bindings. JPEG rather
    # than PNG: same detections, a fraction of the encode cost.
    from Foundation import NSData
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode frame for Apple Vision")
    data = buf.tobytes()
    ns_data = NSData.dataWithBytes_length_(data, len(data))
    return Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None), ns_data


class VisionFaceDetector:
    """Thread-pooled Apple Vision detector. Returns (faces, persons=None)."""

    name = "apple_vision"
    provides_persons = False

    def __init__(
        self,
        width,
        height,
        num_workers=8,
        detect_size=DEFAULT_DETECT_SIZE,
        min_confidence=0.0,
        min_face_size=0,
        batch_size=32,
    ):
        self.width = int(width)
        self.height = int(height)
        self.batch_size = int(batch_size)
        self.min_confidence = float(min_confidence)
        self.min_face_size = float(min_face_size)

        longest = max(self.width, self.height)
        if detect_size and 0 < detect_size < longest:
            self.scale = detect_size / float(longest)
            self.detect_size = (
                max(1, int(round(self.width * self.scale))),
                max(1, int(round(self.height * self.scale))),
            )
        else:
            self.scale = 1.0
            self.detect_size = (self.width, self.height)

        self._pool = ThreadPoolExecutor(max_workers=int(num_workers))

    def _detect_one(self, frame_idx, frame):
        if self.detect_size != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, self.detect_size, interpolation=cv2.INTER_AREA)

        handler, _keepalive = _handler_for(frame)
        request = _get_request()
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Apple Vision face detection failed: {error}")

        faces = []
        for obs in (request.results() or []):
            confidence = float(obs.confidence())
            if confidence < self.min_confidence:
                continue
            # Vision reports normalized, bottom-left-origin boxes, so they map
            # straight onto full-resolution pixels regardless of detect_size.
            box = obs.boundingBox()
            x = box.origin.x * self.width
            box_w = box.size.width * self.width
            box_h = box.size.height * self.height
            y_top = self.height - (box.origin.y * self.height) - box_h
            if max(box_w, box_h) < self.min_face_size:
                continue
            faces.append({
                "frame": int(frame_idx),
                "bbox": [int(x), int(y_top), int(x + box_w), int(y_top + box_h)],
                "conf": confidence,
            })
        return faces

    def detect(self, frames, indices):
        futures = [
            self._pool.submit(self._detect_one, idx, frame)
            for idx, frame in zip(indices, frames)
        ]
        return [f.result() for f in futures], None

    def close(self):
        self._pool.shutdown(wait=True)


def build_detector(width, height, **kwargs):
    return VisionFaceDetector(width, height, **kwargs)


def detect_faces_in_video(video_path, sample_every_n=3, num_workers=8,
                          chunk_size=32, progress=True, **kwargs):
    """Backwards-compatible single-backend entry point; prefer `scan.scan_video`."""
    from .scan import scan_video
    return scan_video(
        video_path,
        backend="apple_vision",
        sample_every_n=sample_every_n,
        num_workers=num_workers,
        batch_size=chunk_size,
        progress=progress,
        detect_scenes=False,
        **kwargs,
    )
