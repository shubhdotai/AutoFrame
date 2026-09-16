"""FaceLiVTv2 embedding wrapper for cross-scene face matching."""

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from test_facelvit2 import build_model, _PREPROCESS


def _auto_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class FaceEmbedder:
    def __init__(self, ckpt_path, arch="facelivtv2_s", device=None):
        self.device = device or _auto_device()
        self.model = build_model(arch)
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            for k in ("state_dict", "model", "net"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)

    def embed_crops(self, bgr_crops):
        """Batch-embed a list of BGR face crops in one forward pass.

        Returns a list aligned to `bgr_crops` with `None` at the index of any
        invalid (None / empty) crop.
        """
        valid = [(i, c) for i, c in enumerate(bgr_crops)
                 if c is not None and c.size > 0]
        out = [None] * len(bgr_crops)
        if not valid:
            return out
        tensors = []
        for _, c in valid:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
            tensors.append(_PREPROCESS(Image.fromarray(rgb)))
        x = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            emb = F.normalize(self.model(x), dim=1).cpu()
        for (i, _), e in zip(valid, emb):
            out[i] = e
        return out

    def embed_crop(self, bgr_crop):
        return self.embed_crops([bgr_crop])[0]

    @staticmethod
    def cosine(a, b):
        return float(torch.dot(a, b))
