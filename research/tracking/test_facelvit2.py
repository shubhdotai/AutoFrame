"""
Single-file FaceLiVTv2 face comparison.

No `backbones/` folder, no separate model files — everything inlined here.
Just point at a checkpoint and two face images and run.

Usage:
    python test_facelivt.py face1.jpg face2.jpg --ckpt facelivtv2_s.pt
    python test_facelivt.py face.jpg --ckpt facelivtv2_s.pt          # single image, prints embedding shape
    python test_facelivt.py f1.jpg f2.jpg --ckpt path/to/x.pt --arch facelivtv2_xs

Architecture code adapted from novendrastywn/FaceLiVT (facelivtv2.py).
Original authorship and weights belong to the FaceLiVT authors.

https://huggingface.co/novendrastywn/FaceLiVT/tree/main
https://github.com/novendrastywn/FaceLiVT/tree/main/backbones
"""

import argparse
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Optional: MTCNN for landmark-based alignment (matches training preprocessing)
try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except ImportError:
    _HAS_MTCNN = False

# timm provides trunc_normal_; required by the model code below
from timm.models.vision_transformer import trunc_normal_


# =========================================================================
# FaceLiVTv2 architecture (inlined from backbones/facelivtv2.py)
# =========================================================================

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device
            ).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = act_layer()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x


class RepConv(nn.Module):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inc, ouc, ks, stride, pad, groups=groups)
        self.repconv = nn.Conv2d(inc, ouc, ks // 2, stride, pad // 2, groups=groups)
        self.bn = nn.BatchNorm2d(ouc)

    def forward(self, x):
        return self.bn(self.conv(x) + self.repconv(x))


class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad = 0 if (ks % 2) == 0 else ks // 2
        blocks = math.ceil(ps ** 0.5)
        dims = [inc] + [x.item() for x in ouc // 2 ** torch.arange(blocks - 1, -1, -1)]
        stem = [
            nn.Sequential(
                RepConv(dims[i], dims[i + 1], ks=ks, stride=2, pad=pad),
                act_layer() if i < (blocks - 1) else nn.Identity(),
            )
            for i in range(blocks)
        ]
        self.stem = nn.Sequential(*stem)

    def forward(self, x):
        return self.stem(x)


class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=7, act_layer=nn.ReLU):
        super().__init__()
        pad = 0 if (ks % 2) == 0 else ks // 2
        self.spatial = nn.Sequential(
            RepConv(inc, inc, ks=ks, stride=2, pad=pad, groups=inc),
            Conv2d_BN(inc, ouc, ks=1, stride=1),
        )
        self.channel = Residual(FFN(ouc, ouc * 2, act_layer))

    def forward(self, x):
        return self.channel(self.spatial(x))


class MHSA(nn.Module):
    """Single-head self-attention (FaceLiVTv2 variant)."""

    def __init__(self, dim, resolution, ratio=0.5, act_layer=nn.ReLU):
        super().__init__()
        self.n_head = 1
        self.qk_dim = 32
        self.v_dim = int(dim * ratio)
        self.att_dim = self.n_head * self.v_dim
        self.split_idx = (self.qk_dim, self.qk_dim, self.v_dim)
        self.head_dim = self.qk_dim + self.qk_dim + self.v_dim
        self.scale = self.qk_dim ** -0.5
        proj_dim = 2 * self.qk_dim * self.n_head + self.v_dim * self.n_head
        self.qkv_proj = Conv2d_BN(dim, proj_dim, 1, 1)
        self.out_proj = Conv2d_BN(self.att_dim, dim, 1, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        qkv = self.qkv_proj(x).reshape(B, self.n_head, self.head_dim, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).split(self.split_idx, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(-2, -1).reshape(B, -1, H, W)
        return self.out_proj(out)


class Affine(nn.Module):
    def __init__(self, dim, res=None):
        super().__init__()
        if res is not None:
            self.alpha = nn.Parameter(1e-5 * torch.ones((1, dim, res)))
            self.beta = nn.Parameter(1e-5 * torch.zeros((1, dim, res)))
        else:
            self.alpha = nn.Parameter(1e-5 * torch.ones((1, dim, 1)))
            self.beta = nn.Parameter(1e-5 * torch.zeros((1, dim, 1)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class MultiHeadLinearAttention(nn.Module):
    """Multi-head linear attention used in v2's MHLA blocks."""

    def __init__(self, dim, resolution):
        super().__init__()
        self.n_head = 4
        self.res = resolution ** 2
        self.dim = dim
        self.norm = Affine(dim)
        self.lin = nn.ModuleList([nn.Linear(self.res, self.res) for _ in range(self.n_head)])
        self.ls = nn.Parameter(1e-5 * torch.ones(dim).unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x.reshape(-1, self.dim, self.res))
        chunks = list(x.chunk(self.n_head, dim=1))
        for i in range(self.n_head):
            chunks[i] = self.lin[i](chunks[i])
        x = torch.cat(chunks, dim=1)
        return self.ls * x.reshape(B, C, H, W)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio, resolution, type_, act_layer=nn.ReLU):
        super().__init__()
        if type_ == 'repmix':
            tok = RepConv(dim, dim, ks=3, stride=1, pad=1, groups=dim)
            self.block = nn.Sequential(
                Residual(tok),
                Residual(FFN(dim, dim * mlp_ratio, act_layer)),
            )
        elif type_ == 'mhsa':
            tok = MHSA(dim, resolution, act_layer=act_layer)
            self.block = nn.Sequential(
                Residual(tok),
                Residual(FFN(dim, dim * mlp_ratio, act_layer)),
            )
        elif type_ == 'mhla':
            rep = RepConv(dim, dim, ks=3, stride=1, pad=1, groups=dim)
            mhla = MultiHeadLinearAttention(dim, resolution)
            self.block = nn.Sequential(
                Residual(rep),
                Residual(mhla),
                Residual(FFN(dim, dim * mlp_ratio, act_layer)),
            )
        else:
            raise ValueError(f"Unknown block type: {type_}")

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    def __init__(self, dim, depth, mlp_ratio, resolution, type_, act_layer=nn.ReLU):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(dim=dim, mlp_ratio=mlp_ratio, resolution=resolution,
                  type_=type_, act_layer=act_layer)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)


class FaceLiVTv2(nn.Module):
    def __init__(self, in_chans=3, img_size=112, num_classes=512,
                 dims=(48, 96, 192, 384), depths=(2, 2, 8, 2),
                 type_=("repmix", "repmix", "mhsa", "mhsa"),
                 ks_pe=3, patch_size=4, mlp_ratio=2,
                 act_layer=nn.GELU, distillation=False,
                 final_feature_dim=None, drop_rate=0.0,
                 pre_head="gdconv", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        num_stage = len(depths)
        self.num_stage = num_stage

        img_res = img_size // patch_size
        patch_embedds = [StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer)]
        stages = []
        for i in range(num_stage):
            stages.append(Stage(
                dim=dims[i], depth=depths[i], type_=type_[i],
                resolution=img_res, mlp_ratio=mlp_ratio, act_layer=act_layer,
            ))
            if i < num_stage - 1:
                patch_embedds.append(PatchMerging(dims[i], dims[i + 1], ks=ks_pe, act_layer=act_layer))
                img_res = math.ceil(img_res / 2)

        self.patch_embedds = nn.Sequential(*patch_embedds)
        self.stages = nn.Sequential(*stages)
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        ffd = final_feature_dim if final_feature_dim is not None else dims[-1]
        if pre_head == "gdconv":
            self.pre_head = nn.Sequential(
                Conv2d_BN(dims[-1], ffd) if final_feature_dim is not None else nn.Identity(),
                Conv2d_BN(ffd, ffd, 4, groups=ffd),
            )
        elif pre_head == "gap":
            self.pre_head = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unknown pre_head: {pre_head}")

        self.head = Classfier(ffd, num_classes, distillation)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for i in range(self.num_stage):
            x = self.patch_embedds[i](x)
            x = self.stages[i](x)
        x = self.pre_head(x).flatten(1)
        return self.head(x)


# Variant configurations from the original facelivtv2.py
_VARIANTS = {
    "facelivtv2_xs": dict(dims=[32, 64, 128, 256],  depths=[3, 3, 9, 3]),
    "facelivtv2_s":  dict(dims=[48, 96, 192, 320],  depths=[3, 3, 9, 3]),
    "facelivtv2_m":  dict(dims=[56, 112, 224, 448], depths=[3, 3, 9, 3]),
    "facelivtv2_l":  dict(dims=[64, 128, 256, 512], depths=[3, 3, 9, 3]),
}


def build_model(arch: str) -> FaceLiVTv2:
    if arch not in _VARIANTS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(_VARIANTS)}")
    cfg = _VARIANTS[arch]
    return FaceLiVTv2(
        num_classes=512,
        dims=cfg["dims"],
        depths=cfg["depths"],
        type_=("repmix", "repmix", "mhla", "mhla"),
        final_feature_dim=1284,
        distillation=False,
    )


# =========================================================================
# Face alignment
# =========================================================================

_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class _Aligner:
    """MTCNN if available, otherwise OpenCV Haar cascade. 112x112 RGB output."""

    def __init__(self, device: str):
        self._mtcnn = None
        self._haar = None
        if _HAS_MTCNN:
            mtcnn_device = "cpu" if device == "mps" else device
            self._mtcnn = MTCNN(
                image_size=112, margin=0, post_process=False, device=mtcnn_device
            )
            self.method = "mtcnn"
        else:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.method = "haar"

    def align(self, image_path: Path) -> Image.Image:
        if self._mtcnn is not None:
            return self._mtcnn_align(image_path)
        return self._haar_align(image_path)

    def _mtcnn_align(self, image_path: Path) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        face = self._mtcnn(img)
        if face is None:
            print(f"  [warn] MTCNN found no face in {image_path.name}; center-cropping.",
                  file=sys.stderr)
            return self._center_crop(img, 112)
        face_np = face.permute(1, 2, 0).clamp(0, 255).byte().numpy()
        return Image.fromarray(face_np)

    def _haar_align(self, image_path: Path) -> Image.Image:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            margin = int(0.2 * max(w, h))
            x0 = max(0, x - margin); y0 = max(0, y - margin)
            x1 = min(bgr.shape[1], x + w + margin); y1 = min(bgr.shape[0], y + h + margin)
            crop = bgr[y0:y1, x0:x1]
        else:
            print(f"  [warn] Haar found no face in {image_path.name}; center-cropping.",
                  file=sys.stderr)
            h, w = bgr.shape[:2]; side = min(h, w)
            y0 = (h - side) // 2; x0 = (w - side) // 2
            crop = bgr[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _center_crop(img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2; top = (h - side) // 2
        return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


# =========================================================================
# Inference helpers
# =========================================================================

def load_model(arch: str, ckpt_path: Path, device: str) -> nn.Module:
    model = build_model(arch)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict):
        # Unwrap common checkpoint wrappers
        for key in ("state_dict", "model", "net"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [info] {len(missing)} missing keys (first 3): {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"  [info] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}", file=sys.stderr)

    model.eval().to(device)
    return model


def get_embedding(model: nn.Module, aligner: _Aligner,
                  image_path: Path, device: str) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    aligned = aligner.align(image_path)
    tensor = _PREPROCESS(aligned).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_image_paths(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def load_embeddings_for_paths(model: nn.Module,
                              aligner: _Aligner,
                              image_paths: list[Path],
                              device: str) -> dict[Path, torch.Tensor]:
    embeddings = {}
    for image_path in image_paths:
        print(f"Embedding {image_path.name}...")
        emb = get_embedding(model, aligner, image_path, device)
        print(f"  shape: {tuple(emb.shape)}")
        embeddings[image_path] = emb
    return embeddings


def compare_embeddings(embeddings: dict[Path, torch.Tensor], threshold: float) -> None:
    paths = list(embeddings.keys())
    if len(paths) < 2:
        print("Need at least two images to compare.", file=sys.stderr)
        return

    names = [path.name for path in paths]
    n = len(paths)
    decisions = [["X"] * n for _ in range(n)]
    durations: list[tuple[str, str, float, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            start = time.perf_counter()
            sim = cosine_similarity(embeddings[paths[i]], embeddings[paths[j]])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            decision = "Y" if sim >= threshold else "N"
            decisions[i][j] = decisions[j][i] = decision
            durations.append((names[i], names[j], sim, decision, elapsed_ms))

    col_width = max(max(len(name) for name in names), 5)
    sep = " | "
    header = "".ljust(col_width) + sep + sep.join(name.center(col_width) for name in names)
    line = "-" * len(header)

    print()
    print("Decision matrix (Y = same person, N = different person):")
    print(line)
    print(header)
    print(line)
    for i, name in enumerate(names):
        row = sep.join(decisions[i][j].center(col_width) for j in range(n))
        print(name.ljust(col_width) + sep + row)
    print(line)

    print()
    print("Per-pair similarity and timing:")
    for a_name, b_name, sim, decision, elapsed_ms in durations:
        print(f"{a_name} vs {b_name}: {sim:+.4f} | {'SAME' if decision == 'Y' else 'DIFF'} | {elapsed_ms:.2f} ms")


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="FaceLiVTv2 face comparison (single-file).")
    parser.add_argument("face1", type=Path,
                        help="First face image or a folder containing images for all-pairs matching.")
    parser.add_argument("face2", type=Path, nargs="?", default=None,
                        help="Optional second face image; if given, prints similarity.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to the FaceLiVTv2 .pt checkpoint.")
    parser.add_argument("--arch", default="facelivtv2_s", choices=list(_VARIANTS),
                        help="Model variant (default: facelivtv2_s).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Same-person cosine threshold (default: 0.5).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda:0", "mps"],
                        help="Device (default: auto = CUDA > MPS > CPU).")
    args = parser.parse_args()

    if args.face1.is_dir() and args.face2 is not None:
        raise ValueError("When face1 is a directory, --face2 must not be provided.")

    device = pick_device(args.device)
    aligner = _Aligner(device)

    print(f"Device:    {device}")
    print(f"Aligner:   {aligner.method}"
          + ("" if _HAS_MTCNN else "  (install `facenet-pytorch` for landmark alignment)"))
    print(f"Loading {args.arch} from {args.ckpt}...")
    model = load_model(args.arch, args.ckpt, device)

    if args.face1.is_dir():
        image_paths = list_image_paths(args.face1)
        if not image_paths:
            raise RuntimeError(f"No image files found in folder: {args.face1}")
        print(f"Found {len(image_paths)} images in {args.face1}")
        embeddings = load_embeddings_for_paths(model, aligner, image_paths, device)
        compare_embeddings(embeddings, args.threshold)
        return 0

    print(f"Embedding {args.face1.name}...")
    emb1 = get_embedding(model, aligner, args.face1, device)
    print(f"  shape: {tuple(emb1.shape)}")

    if args.face2 is None:
        return 0

    print(f"Embedding {args.face2.name}...")
    emb2 = get_embedding(model, aligner, args.face2, device)
    print(f"  shape: {tuple(emb2.shape)}")

    sim = cosine_similarity(emb1, emb2)
    is_same = sim >= args.threshold

    print()
    print("=" * 50)
    print(f"  Cosine similarity: {sim:+.4f}")
    print(f"  Threshold:         {args.threshold:+.4f}")
    print(f"  Decision:          {'SAME PERSON ✓' if is_same else 'DIFFERENT PERSON ✗'}")
    print("=" * 50)
    return 0 if is_same else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)