"""
YOLOv8 face + person detection (iitolstykh/YOLO-Face-Person-Detector).

Exposes the same detector interface as the Apple Vision backend so
`scan.scan_video` can drive either one from a single decode pass. Person boxes
are returned alongside faces and are used by the reframer as a fallback subject
when no face is visible.
"""

import os

import numpy as np

# Class IDs in the iitolstykh/YOLO-Face-Person-Detector checkpoint.
_CLS_PERSON = 0
_CLS_FACE = 1


def _resolve_device(requested):
    if requested == "mps":
        try:
            import torch
            if not torch.backends.mps.is_available():
                print("[face_detection_yolo] MPS not available, falling back to cpu")
                return "cpu"
        except Exception:
            return "cpu"
    return requested


class YoloFacePersonDetector:
    """Batched YOLO detector. Returns (faces, persons)."""

    name = "yolo_face_person"
    provides_persons = True

    def __init__(
        self,
        width,
        height,
        weights_path,
        device="mps",
        conf=0.4,
        iou=0.7,
        imgsz=640,
        batch_size=16,
        min_confidence=0.0,
        min_face_size=0,
        progress=True,
    ):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"missing YOLO weights: {weights_path}")
        from ultralytics import YOLO

        self.width = int(width)
        self.height = int(height)
        self.device = _resolve_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.batch_size = int(batch_size)
        self.min_confidence = float(min_confidence)
        self.min_face_size = float(min_face_size)

        if progress:
            print(f"[face_detection_yolo] loading {os.path.basename(weights_path)}")
        self.model = YOLO(weights_path)
        # Ultralytics moves the model lazily on the first predict call; warm it
        # up here so device errors surface before the main loop.
        self.model.predict(
            np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
            device=self.device, verbose=False, imgsz=self.imgsz,
        )

    def detect(self, frames, indices):
        results = self.model.predict(
            frames, device=self.device, imgsz=self.imgsz,
            conf=self.conf, iou=self.iou, verbose=False,
        )
        faces_out, persons_out = [], []
        for gi, res in zip(indices, results):
            faces, persons = [], []
            boxes = res.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                    rec = {
                        "frame": int(gi),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(cf),
                    }
                    if c == _CLS_FACE:
                        if cf < self.min_confidence:
                            continue
                        if max(x2 - x1, y2 - y1) < self.min_face_size:
                            continue
                        faces.append(rec)
                    elif c == _CLS_PERSON:
                        persons.append(rec)
            faces_out.append(faces)
            persons_out.append(persons)
        return faces_out, persons_out

    def close(self):
        pass


def build_detector(width, height, **kwargs):
    return YoloFacePersonDetector(width, height, **kwargs)


def detect_faces_in_video(video_path, weights_path, sample_every_n=3, batch_size=16,
                          device="mps", conf=0.4, iou=0.7, imgsz=640, progress=True,
                          **kwargs):
    """Backwards-compatible single-backend entry point; prefer `scan.scan_video`."""
    from .scan import scan_video
    return scan_video(
        video_path,
        backend="yolo_face_person",
        sample_every_n=sample_every_n,
        batch_size=batch_size,
        progress=progress,
        detect_scenes=False,
        weights_path=weights_path,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        **kwargs,
    )
