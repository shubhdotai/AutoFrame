"""Small public CLI; model dependencies are imported only when needed."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description="Reframe a landscape video around its active speaker.")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies and model files")
    doctor.add_argument("--detector", choices=["vision", "yolo"], default="vision")
    doctor.add_argument("--models-dir", type=Path, default=Path("models"))
    for command in ("run", "analyze", "render"):
        q = sub.add_parser(command, help={"run": "Analyze and render", "analyze": "Write speaker artifacts", "render": "Render existing analysis"}[command])
        q.add_argument("video", type=Path)
        q.add_argument("--output", type=Path, default=Path("out"))
        if command != "render":
            q.add_argument("--weights", type=Path, default=Path("models/pretrain_AVA.model"))
            q.add_argument("--detector", choices=["vision", "yolo"], default="vision")
            q.add_argument("--yolo-weights", type=Path, default=Path("models/yolov8x_person_face.pt"))
            q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
            q.add_argument("--sample-every", type=positive_int, default=2)
            q.add_argument("--encoder-window", type=positive_int, default=100)
            q.add_argument("--detector-batch", type=positive_int, default=64)
            q.add_argument("--debug-video", action="store_true")
        q.add_argument("--threshold", type=float, default=0.0)
    return p


def doctor(args):
    checks = [("FFmpeg", shutil.which("ffmpeg") is not None)]
    for name in ["torch", "cv2", "numpy", "scipy", "scenedetect", "python_speech_features", "tqdm"]:
        checks.append((name, importlib.util.find_spec(name) is not None))
    checks.append(("LR-ASD checkpoint", (args.models_dir / "pretrain_AVA.model").is_file()))
    if args.detector == "vision":
        checks.append(("macOS / Apple Vision", sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None))
    else:
        checks.extend([("ultralytics", importlib.util.find_spec("ultralytics") is not None), ("YOLO face/person weights", (args.models_dir / "yolov8x_person_face.pt").is_file())])
    for label, ok in checks:
        print(f"{'OK' if ok else 'MISSING':7} {label}")
    if not all(ok for _, ok in checks):
        raise SystemExit(1)


def analyze(args):
    import torch
    from .run import main as run_analysis

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory is not empty; choose a new directory, or use render to reuse an analysis")
    if not args.weights.is_file():
        raise FileNotFoundError(f"missing {args.weights}; run python scripts/download_models.py")
    if args.detector == "vision" and (sys.platform != "darwin" or importlib.util.find_spec("Vision") is None):
        raise ValueError("Vision requires macOS and pip install -e '.[mac]'; elsewhere use --detector yolo")
    if args.detector == "yolo" and not args.yolo_weights.is_file():
        raise FileNotFoundError(f"missing YOLO face/person checkpoint: {args.yolo_weights}")
    if args.encoder_window % 4:
        raise ValueError("--encoder-window must be a multiple of 4")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    argv = ["--videoPath", str(args.video), "--savePath", str(args.output),
            "--pretrainModel", str(args.weights.resolve()), "--device", device,
            "--faceDetector", "apple_vision" if args.detector == "vision" else "yolo_face_person",
            "--yoloWeights", str(args.yolo_weights.resolve()),
            "--sampleEvery", str(args.sample_every), "--numFailedDet", str(max(10, args.sample_every * 2)),
            "--encoderWindowFrames", str(args.encoder_window), "--detectorBatch", str(args.detector_batch),
            "--threshold", str(args.threshold)]
    if not args.debug_video:
        argv.append("--noVisualize")
    run_analysis(argv)


def render(args):
    from .shortform import _load_artifacts, build_reframe_plan, serialize_plan, render_video

    manifest = json.loads((args.output / "run.json").read_text())
    if Path(manifest["source_video"]).resolve() != args.video:
        raise ValueError("analysis belongs to a different source video")
    art = _load_artifacts(args.output)
    plan = build_reframe_plan(art["tracks"], art["scores"], art["n_frames"], art["width"], art["height"], fps=art["fps"], speaking_threshold=args.threshold, scene_cuts=art["scene_cuts"])
    path = args.output / "vertical.mp4"
    if path.exists():
        raise ValueError("vertical.mp4 already exists; move it before rendering again")
    video = args.output / "_work/video_25fps.mp4"
    audio = args.output / "_work/audio_16khz.wav"
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("render requires the analysis _work media; rerun analyze into a new directory")
    (args.output / "shortform_plan.json").write_text(json.dumps(serialize_plan(plan), indent=2))
    render_video(video, args.video, plan, path)
    print(f"Created {path}")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        doctor(args)
        return
    try:
        args.video = args.video.expanduser().resolve()
        args.output = args.output.expanduser().resolve()
        if not args.video.is_file():
            raise FileNotFoundError(f"video does not exist: {args.video}")
        if not shutil.which("ffmpeg"):
            raise ValueError("FFmpeg is missing; install it and add it to PATH")
        if not math.isfinite(args.threshold):
            raise ValueError("threshold must be finite")
        if args.command in ("run", "analyze"):
            analyze(args)
        if args.command in ("run", "render"):
            render(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        raise SystemExit(f"autoclip: {exc}") from exc
