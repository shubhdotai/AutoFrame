"""Scene-aware face tracking with ByteTrack + cross-scene matching via embeddings.

Within a scene ByteTrack alone gives identity (no embeddings). At end-of-scene
each local track is reconciled against a gallery of faces from PRIOR scenes:

  - Pick SAMPLES_PER_FACE crops per local track via confidence-weighted random
    sampling (higher YOLO conf => more likely to be picked).
  - Batch-embed every chosen crop in a single model forward pass.
  - For each local track, walk the gallery in gid order (face1, face2, ...) and
    compare its embeddings against each known face's references (SAMPLES x
    SAMPLES = 9 sims per known face). The first face whose any pair clears
    MATCH_THRESHOLD wins and matching stops. If none matches, mint a new gid.
  - Gallery commits and seen-counts are applied AFTER the scene, so tracks
    within the same scene never embed-match each other.
  - Gallery is capped at MAX_GALLERY: keep the faces seen in the most scenes
    (recurring characters); evict the rest.

Usage: python pipeline_faces.py path/to/video.mp4 [--out out.mp4]
"""

import argparse
import colorsys
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from face_embedder import FaceEmbedder
from scene_detector import SceneDetector


HERE = Path(__file__).parent
YOLO_WEIGHTS = HERE.parent.parent / "models" / "yolov8x_person_face.pt"
FACE_CKPT = HERE.parent.parent / "models" / "facelivtv2-s.pt"

SKIP_FRAMES = 3            # detect every Nth frame; rest are interpolated
SAMPLES_PER_FACE = 3       # crops stored per face / queried per new track
MATCH_THRESHOLD = 0.5      # cosine similarity for "same identity"
MAX_GALLERY = 10           # cap; least-frequently-seen faces are evicted
_CLS_FACE = 1


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def color_for(gid):
    h = (gid * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def interp(a, b, t):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(4))


def _reset_tracker(model):
    """Clear ByteTrack state so each scene starts fresh."""
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", None) or []:
        t.reset()


def _crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


def track_scene(model, frames, device):
    """Run YOLO + ByteTrack on sampled frames of one scene.

    Returns:
      annos:       {frame_idx -> [(bbox, local_id), ...]}
      occurrences: {local_id  -> [(frame_idx, bbox, conf), ...]}
    """
    _reset_tracker(model)
    annos, occurrences = {}, {}
    targets = sorted({*range(0, len(frames), SKIP_FRAMES), len(frames) - 1})

    for fi in targets:
        res = model.track(frames[fi], device=device, persist=True,
                          tracker="bytetrack.yaml", conf=0.4, iou=0.7,
                          classes=[_CLS_FACE], verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        out = []
        for bb, lid, c in zip(xyxy, ids, confs):
            bbox, lid = tuple(bb), int(lid)
            out.append((bbox, lid))
            occurrences.setdefault(lid, []).append((fi, bbox, float(c)))
        annos[fi] = out

    return annos, occurrences


def _pick_samples(occs, k=SAMPLES_PER_FACE):
    """Confidence-weighted random pick of up to k distinct (frame_idx, bbox) pairs.

    Efraimidis-Spirakis weighted reservoir: key = U^(1/conf); take the top-k keys.
    Higher detection confidence => higher expected key => more likely to be picked.
    """
    keyed = sorted(
        ((random.random() ** (1.0 / max(c, 1e-6)), fi, bb) for fi, bb, c in occs),
        reverse=True,
    )
    return [(fi, bb) for _, fi, bb in keyed[:k]]


def first_match_in_gallery(query_embs, gallery, threshold):
    """Walk gallery in gid order; return the first gid where any (query, ref)
    cosine clears the threshold. None means no match.
    """
    for gid in sorted(gallery):
        for q in query_embs:
            for r in gallery[gid]:
                if FaceEmbedder.cosine(q, r) >= threshold:
                    return gid
    return None


def _prune_gallery(gallery, counts, limit):
    """Cap gallery at `limit` entries; keep faces with the highest scene counts."""
    if len(gallery) <= limit:
        return
    keep = set(sorted(gallery, key=lambda g: counts[g], reverse=True)[:limit])
    for gid in list(gallery):
        if gid not in keep:
            del gallery[gid]
            del counts[gid]


def assign_global_ids(occurrences, frames, embedder, gallery, counts,
                      next_gid, threshold):
    """Reconcile scene-local tracks against the gallery of PRIOR scenes.

    Gallery writes and count bumps are deferred until after every local track
    has been matched, so tracks in the same scene never embed-match each other.
    All chosen crops in the scene are embedded in one batched forward pass.
    """
    # 1) Confidence-weighted pick per local track.
    picks = {lid: _pick_samples(occs) for lid, occs in occurrences.items()}

    # 2) Flatten + batch-embed every chosen crop in one model call.
    flat = [(lid, fi, bb) for lid, ps in picks.items() for fi, bb in ps]
    embs = embedder.embed_crops([_crop(frames[fi], bb) for _, fi, bb in flat])

    # 3) Regroup embeddings back per local track (dropping any None).
    by_lid = {}
    for (lid, _, _), e in zip(flat, embs):
        if e is not None:
            by_lid.setdefault(lid, []).append(e)

    # 4) Match each local track; stage new faces, do NOT write the gallery yet.
    local_to_gid, pending = {}, {}
    for lid in occurrences:
        track_embs = by_lid.get(lid, [])
        gid = first_match_in_gallery(track_embs, gallery, threshold) \
            if track_embs else None
        if gid is None:
            gid = next_gid
            next_gid += 1
            pending[gid] = track_embs
        local_to_gid[lid] = gid
        counts[gid] = counts.get(gid, 0) + 1

    # 5) Commit and cap.
    gallery.update(pending)
    _prune_gallery(gallery, counts, MAX_GALLERY)
    return local_to_gid, next_gid


def relabel(annos, local_to_gid):
    return {fi: [(bbox, local_to_gid[lid]) for bbox, lid in items]
            for fi, items in annos.items()}


def expand(annos, n):
    """Linearly interpolate bboxes for frames between detected ones."""
    detected = sorted(annos.keys())
    if not detected:
        return [[] for _ in range(n)]
    out = [[] for _ in range(n)]
    for d in detected:
        out[d] = annos[d]
    for a, b in zip(detected, detected[1:]):
        amap = {g: bb for bb, g in annos[a]}
        bmap = {g: bb for bb, g in annos[b]}
        for fi in range(a + 1, b):
            t = (fi - a) / (b - a)
            out[fi] = [(interp(amap[g], bmap[g], t) if g in bmap else amap[g], g)
                       for g in amap]
    return out


def draw(frame, annos):
    for bbox, gid in annos:
        x1, y1, x2, y2 = bbox
        color = color_for(gid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Face{gid}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--face-ckpt", type=Path, default=FACE_CKPT)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    args = ap.parse_args()
    out_path = args.out or args.video.with_name(args.video.stem + "_faces.mp4")
    if out_path.is_dir():
        out_path = out_path / (args.video.stem + "_faces.mp4")

    print("[1/3] Detecting scenes...")
    scenes, _ = SceneDetector().detect(str(args.video))
    print(f"      {len(scenes)} scenes")

    print("[2/3] Loading models...")
    device = _device()
    detector = YOLO(str(YOLO_WEIGHTS))
    detector.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                     device=device, verbose=False)
    embedder = FaceEmbedder(str(args.face_ckpt), device=device)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print(f"[3/3] Detect + ByteTrack + embed/match -> {out_path}")
    gallery = {}      # gid -> [SAMPLES_PER_FACE reference embeddings]
    counts = {}       # gid -> number of scenes this face appeared in
    next_gid = 1
    cur = 0
    for sc in scenes:
        frames = []
        while cur <= sc["end_frame"]:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f); cur += 1
        if not frames:
            continue

        local_annos, occurrences = track_scene(detector, frames, device)
        local_to_gid, next_gid = assign_global_ids(
            occurrences, frames, embedder, gallery, counts,
            next_gid, args.threshold)
        annos = relabel(local_annos, local_to_gid)

        for frame, a in zip(frames, expand(annos, len(frames))):
            writer.write(draw(frame, a))

        gids = sorted(set(local_to_gid.values()))
        tag = ", ".join(f"Face{g}" for g in gids) if gids else "(none)"
        print(f"      scene {sc['index']}: {tag}")

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
