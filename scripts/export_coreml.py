"""Export LR-ASD weights to CoreML mlpackages.

Produces five files in --out-dir:

  visual_encoder.mlpackage      (1, T_v, 112, 112) -> (1, T_v, 128)
  audio_encoder.mlpackage       (1, T_v*4, 13)     -> (1, T_v, 128)
  detector_head_d2.mlpackage    (B,  50, 128) x2   -> (B,  50)
  detector_head_d4.mlpackage    (B, 100, 128) x2   -> (B, 100)
  detector_head_d6.mlpackage    (B, 150, 128) x2   -> (B, 150)

Usage:
  python optimized_asd/tools/export_coreml.py
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct

from autoclip.ASD import ASD


class VisualWrap(nn.Module):
    def __init__(self, asd):
        super().__init__()
        self.model = asd.model

    def forward(self, v):
        return self.model.forward_visual_frontend(v)


class AudioWrap(nn.Module):
    def __init__(self, asd):
        super().__init__()
        self.model = asd.model

    def forward(self, a):
        return self.model.forward_audio_frontend(a)


class BackendWrap(nn.Module):
    def __init__(self, asd):
        super().__init__()
        self.fusion = asd.model.fusion
        self.detector = asd.model.detector
        self.fc = asd.lossAV.FC

    def forward(self, audio_embed, visual_embed):
        x = self.fusion(audio_embed, visual_embed)
        x = self.detector(x)
        logits = self.fc(x)
        return logits[..., 1]


def _convert(traced, inputs, name, out_dir, target):
    print(f"  converting {name} ...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
    )
    out_path = os.path.join(out_dir, f"{name}.mlpackage")
    mlmodel.save(out_path)
    print(f"  saved -> {out_path}")
    return mlmodel


def _max_abs_diff(ref, got):
    return float(np.max(np.abs(ref.astype(np.float32) - got.astype(np.float32))))


def _parity(torch_mod, mlmodel, named_inputs):
    with torch.no_grad():
        ref = torch_mod(*named_inputs.values()).cpu().numpy()
    pred = mlmodel.predict({k: v.numpy() for k, v in named_inputs.items()})
    out_name = next(iter(pred.keys()))
    diff = _max_abs_diff(ref, pred[out_name])
    print(f"  parity: max abs diff = {diff:.4e}  (output: {out_name})")
    if not np.isfinite(diff) or diff > 0.1:
        raise RuntimeError(f"Core ML export parity failed: {diff}")
    return diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",
                   default="models/pretrain_AVA.model")
    p.add_argument("--out-dir",
                   default="models/coreml")
    p.add_argument("--encoder-window", type=int, default=100,
                   help="fixed video frames per encoder call")
    p.add_argument("--max-batch", type=int, default=64,
                   help="upper bound for detector batch RangeDim")
    p.add_argument("--durations", default="2,4,6")
    p.add_argument("--target", default="macOS14",
                   choices=["macOS13", "macOS14", "macOS15"])
    args = p.parse_args()

    target = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
    }[args.target]

    durations = tuple(int(x) for x in args.durations.split(",") if x.strip())

    print(f"loading weights: {args.weights}")
    asd = ASD(device="cpu")
    asd.loadParameters(args.weights)
    asd.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    T_v = args.encoder_window
    T_a = T_v * 4

    print("\n[1/3] visual encoder")
    vmod = VisualWrap(asd).eval()
    v_example = (torch.randn(1, T_v, 112, 112) * 50 + 128).clamp_(0, 255)
    traced = torch.jit.trace(vmod, v_example)
    mlmodel = _convert(
        traced,
        inputs=[ct.TensorType(name="v", shape=(1, T_v, 112, 112), dtype=np.float32)],
        name="visual_encoder",
        out_dir=args.out_dir,
        target=target,
    )
    _parity(vmod, mlmodel, {"v": v_example})

    print("\n[2/3] audio encoder")
    amod = AudioWrap(asd).eval()
    a_example = torch.randn(1, T_a, 13)
    traced = torch.jit.trace(amod, a_example)
    mlmodel = _convert(
        traced,
        inputs=[ct.TensorType(name="a", shape=(1, T_a, 13), dtype=np.float32)],
        name="audio_encoder",
        out_dir=args.out_dir,
        target=target,
    )
    _parity(amod, mlmodel, {"a": a_example})

    print("\n[3/3] detector heads")
    bmod = BackendWrap(asd).eval()
    for d in durations:
        T_chunk = d * 25
        print(f"\n  duration={d}s, T_chunk={T_chunk}")
        a_ex = torch.randn(1, T_chunk, 128)
        v_ex = torch.randn(1, T_chunk, 128)
        traced = torch.jit.trace(bmod, (a_ex, v_ex))
        b_dim = ct.RangeDim(lower_bound=1, upper_bound=args.max_batch, default=1)
        mlmodel = _convert(
            traced,
            inputs=[
                ct.TensorType(name="audio_embed",
                              shape=(b_dim, T_chunk, 128), dtype=np.float32),
                ct.TensorType(name="visual_embed",
                              shape=(b_dim, T_chunk, 128), dtype=np.float32),
            ],
            name=f"detector_head_d{d}",
            out_dir=args.out_dir,
            target=target,
        )
        _parity(bmod, mlmodel,
                {"audio_embed": a_ex, "visual_embed": v_ex})

    print("\ndone.")


if __name__ == "__main__":
    main()
