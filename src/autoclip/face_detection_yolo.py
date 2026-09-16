"""
YOLOv8 face + person detection (iitolstykh/YOLO-Face-Person-Detector).

Streams the input video, samples 1-in-N frames, runs batched YOLO inference on
the chosen device (MPS by default on Apple Silicon), and returns per-frame
lists of face and person detections.

Output schema mirrors `face_detection.detect_faces_in_video` so the tracking
pipeline can consume `detections_per_frame` unchanged. Person detections are
returned alongside via `persons_per_frame` for visualization purposes.
"""

import os
import time

import cv2
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


def _load_model(weights_path, device):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    # Warm up on the chosen device. Ultralytics moves the model lazily on the
    # first predict call, so we trigger it here to surface any device errors
    # before the main loop.
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=device, verbose=False, imgsz=640)
    return model


def detect_faces_in_video(
    video_path,
    weights_path,
    sample_every_n=3,
    batch_size=16,
    device="mps",
    conf=0.4,
    iou=0.7,
    imgsz=640,
    progress=True,
):
    """
    Returns:
        {
          "fps": float, "width": int, "height": int,
          "total_frames": int, "sample_every_n": int,
          "detections_per_frame": list[list[face_dict]],
          "persons_per_frame":   list[list[person_dict]],
        }
    Each face/person dict: {"frame": int, "bbox": [x1,y1,x2,y2], "conf": float}
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"missing YOLO weights: {weights_path}")

    device = _resolve_device(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if progress:
        dur_min = total_frames / fps / 60 if fps else 0
        print(f"[face_detection_yolo] {width}x{height} @ {fps:.2f} fps | "
              f"{total_frames} frames ({dur_min:.1f} min)")
        print(f"[face_detection_yolo] sampling every {sample_every_n} frame(s) | "
              f"batch={batch_size} | device={device} | imgsz={imgsz} | "
              f"conf={conf} | iou={iou}")

    print(f"[face_detection_yolo] loading {os.path.basename(weights_path)}")
    model = _load_model(weights_path, device)

    detections_per_frame = [[] for _ in range(total_frames)]
    persons_per_frame = [[] for _ in range(total_frames)]

    t_start = time.perf_counter()
    frame_idx = 0

    pending_frames = []
    pending_indices = []

    def flush(batch_frames, batch_indices):
        if not batch_frames:
            return
        # Ultralytics accepts a list of np.ndarrays (BGR HxWxC) and returns one
        # Result per input.
        results = model.predict(
            batch_frames,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        for gi, res in zip(batch_indices, results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
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
                    detections_per_frame[gi].append(rec)
                elif c == _CLS_PERSON:
                    persons_per_frame[gi].append(rec)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gi = frame_idx
        frame_idx += 1
        if gi % sample_every_n != 0:
            continue
        pending_frames.append(frame)
        pending_indices.append(gi)
        if len(pending_frames) >= batch_size:
            flush(pending_frames, pending_indices)
            pending_frames.clear()
            pending_indices.clear()
            if progress:
                elapsed = time.perf_counter() - t_start
                pct = frame_idx / max(total_frames, 1) * 100
                fps_a = frame_idx / max(elapsed, 1e-6)
                eta_min = (total_frames - frame_idx) / max(fps_a, 1e-6) / 60
                print(f"  {frame_idx}/{total_frames} ({pct:.1f}%) | "
                      f"{fps_a:.1f} fps | ETA {eta_min:.1f} min")

    # Drain
    flush(pending_frames, pending_indices)
    cap.release()

    elapsed = time.perf_counter() - t_start
    if progress:
        n_faces = sum(len(d) for d in detections_per_frame)
        n_persons = sum(len(d) for d in persons_per_frame)
        print(f"[face_detection_yolo] done in {elapsed/60:.1f} min | "
              f"{n_faces} face detections | {n_persons} person detections")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "sample_every_n": sample_every_n,
        "detections_per_frame": detections_per_frame,
        "persons_per_frame": persons_per_frame,
    }
