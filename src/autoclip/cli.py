"""Small public CLI; model dependencies are imported only when needed."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def aspect(value):
    try:
        w, h = (int(v) for v in str(value).split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected W:H, for example 9:16")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("aspect components must be positive")
    return w, h


def parser():
    p = argparse.ArgumentParser(description="Reframe a landscape video around its active speaker.")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies and model files")
    doctor.add_argument("--detector", choices=["vision", "yolo"], default="vision",
                        help="Which detector's requirements to check; 'vision' "
                             "(default) is the macOS Vision framework.")
    doctor.add_argument("--models-dir", type=Path, default=Path("models"))
    for command in ("run", "analyze", "render"):
        q = sub.add_parser(command, help={"run": "Analyze and render", "analyze": "Write speaker artifacts", "render": "Render existing analysis"}[command])
        q.add_argument("video", type=Path)
        q.add_argument("--output", type=Path, default=Path("out"))
        q.add_argument("--aspect", type=aspect, default=(9, 16),
                       help="Target aspect ratio W:H (default 9:16)")
        if command != "render":
            q.add_argument("--weights", type=Path, default=Path("models/pretrain_AVA.model"))
            q.add_argument("--detector", choices=["vision", "yolo"], default="vision",
                           help="Face detector. 'vision' (default) is Apple's "
                                "Vision framework on macOS -- the system "
                                "VNDetectFaceRectanglesRequest, with no weight "
                                "file to download. 'yolo' is the cross-platform "
                                "YOLO face/person checkpoint.")
            q.add_argument("--yolo-weights", type=Path, default=Path("models/yolov8x_person_face.pt"))
            q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
            q.add_argument("--sample-every", type=positive_int, default=2)
            q.add_argument("--detect-size", type=int, default=720,
                           help="Vision: long edge used for detection (0 = native)")
            q.add_argument("--scene-mode", choices=["content", "adaptive"], default="content",
                           help="adaptive resists false cuts on fast motion and flashes")
            q.add_argument("--min-face-conf", type=float, default=0.0,
                           help="Drop detections below this confidence")
            q.add_argument("--min-face-fraction", type=float, default=0.0,
                           help="Drop faces smaller than this fraction of frame height")
            q.add_argument("--encoder-window", type=positive_int, default=100)
            q.add_argument("--encoder-batch", type=positive_int, default=16)
            q.add_argument("--detector-batch", type=positive_int, default=64)
            q.add_argument("--score-all-tracks", action="store_true",
                           help="Score every track, even where the result cannot "
                                "change the crop (slower; complete reports)")
            q.add_argument("--silence-floor-db", type=float, default=-45.0)
            q.add_argument("--video-encoder", default="auto",
                           help="auto, hardware, software, or an FFmpeg encoder name")
            q.add_argument("--debug-video", action="store_true")
        if command == "run":
            q.add_argument("--verbose", "-v", action="store_true",
                           help="Also keep the analysis artifacts: the JSON "
                                "reports, the 25 fps working video and the "
                                "16 kHz audio. Off by default, so a plain run "
                                "leaves only vertical.mp4 behind.")
        if command != "analyze":
            q.add_argument("--motion", choices=["lock", "follow"], default="lock",
                           help="lock holds a fixed crop per shot; follow eases "
                                "toward a subject who leaves a deadzone")
            q.add_argument("--speaker-margin", type=float, default=0.0,
                           help="Lead the top speaker needs over the runner-up")
            q.add_argument("--native-fps", action="store_true",
                           help="Render from the original source at its own frame "
                                "rate instead of the 25 fps analysis copy")
            q.add_argument("--output-height", type=int, default=None,
                           help="Scale the finished crop to this height, e.g. 1920")
        q.add_argument("--threshold", type=float, default=0.0)
    return p


def doctor(args):
    checks = [("FFmpeg", shutil.which("ffmpeg") is not None),
              ("FFprobe", shutil.which("ffprobe") is not None)]
    for name in ["torch", "cv2", "numpy", "scipy", "scenedetect", "python_speech_features", "tqdm"]:
        checks.append((name, importlib.util.find_spec(name) is not None))
    checks.append(("LR-ASD checkpoint", (args.models_dir / "pretrain_AVA.model").is_file()))
    if args.detector == "vision":
        checks.append(("macOS / Apple Vision", sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None))
        checks.append(("CoreVideo pixel buffers (fast path)", importlib.util.find_spec("Quartz") is not None))
    else:
        checks.extend([("ultralytics", importlib.util.find_spec("ultralytics") is not None), ("YOLO face/person weights", (args.models_dir / "yolov8x_person_face.pt").is_file())])
    optional = {"CoreVideo pixel buffers (fast path)"}
    for label, ok in checks:
        print(f"{'OK' if ok else ('SLOW' if label in optional else 'MISSING'):7} {label}")
    if not all(ok for label, ok in checks if label not in optional):
        raise SystemExit(1)


def analyze(args, work_dir=None, verbose=True):
    import torch
    from .run import main as run_analysis

    if verbose and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory is not empty; choose a new directory, or use render to reuse an analysis")
    if not args.weights.is_file():
        raise FileNotFoundError(f"missing {args.weights}; run python scripts/download_models.py")
    if args.detector == "vision" and (sys.platform != "darwin" or importlib.util.find_spec("Vision") is None):
        raise ValueError("Vision requires macOS and pip install -e '.[mac]'; elsewhere use --detector yolo")
    if args.detector == "yolo" and not args.yolo_weights.is_file():
        raise FileNotFoundError(f"missing YOLO face/person checkpoint: {args.yolo_weights}")
    if args.encoder_window % 4:
        raise ValueError("--encoder-window must be a multiple of 4")
    if not 0.0 <= args.min_face_fraction < 1.0:
        raise ValueError("--min-face-fraction must be in [0, 1)")
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
            "--faceDetectSize", str(args.detect_size), "--sceneMode", args.scene_mode,
            "--minFaceConf", str(args.min_face_conf),
            "--minFaceFraction", str(args.min_face_fraction),
            "--encoderWindowFrames", str(args.encoder_window),
            "--encoderBatch", str(args.encoder_batch),
            "--detectorBatch", str(args.detector_batch),
            "--silenceFloorDb", str(args.silence_floor_db),
            "--targetAspect", f"{args.aspect[0]}:{args.aspect[1]}",
            "--encoder", args.video_encoder,
            "--threshold", str(args.threshold)]
    if work_dir is not None:
        argv += ["--workDir", str(work_dir)]
    if verbose:
        argv.append("--verbose")
    if args.score_all_tracks:
        argv.append("--scoreAllTracks")
    if not args.debug_video:
        argv.append("--noVisualize")
    return run_analysis(argv)


def render(args, analysis=None, verbose=True):
    from .shortform import _load_artifacts, build_reframe_plan, serialize_plan, render_video

    if analysis is None:
        manifest = json.loads((args.output / "run.json").read_text())
        if Path(manifest["source_video"]).resolve() != args.video:
            raise ValueError("analysis belongs to a different source video")
        art = _load_artifacts(args.output)
    else:
        art = analysis
    plan = build_reframe_plan(
        art["tracks"], art["scores"], art["n_frames"], art["width"], art["height"],
        fps=art["fps"], target_aspect_w=args.aspect[0], target_aspect_h=args.aspect[1],
        speaking_threshold=args.threshold, scene_cuts=art["scene_cuts"],
        speaker_margin=args.speaker_margin, motion=args.motion,
        persons_per_frame=art.get("persons_per_frame"),
    )
    path = args.output / "vertical.mp4"
    if path.exists():
        raise ValueError("vertical.mp4 already exists; move it before rendering again")
    if analysis is None:
        video = args.output / "_work/video_25fps.mp4"
        audio = args.output / "_work/audio_16khz.wav"
    else:
        video = Path(analysis["video"])
        audio = Path(analysis["audio"])
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("render requires the analysis _work media; rerun analyze into a new directory")
    if verbose:
        (args.output / "shortform_plan.json").write_text(
            json.dumps(serialize_plan(plan), indent=2))
    # Audio always comes from the original source, so the deliverable never
    # inherits the 16 kHz mono waveform that ASD works from.
    source = video
    if args.native_fps:
        from .media import probe_video
        info = probe_video(args.video)
        if (info["width"], info["height"]) == (plan["video_width"], plan["video_height"]):
            source = args.video
        else:
            print("autoclip: source dimensions differ from analysis; "
                  "rendering from the normalized copy instead")
    render_video(source, args.video, plan, path,
                 native_fps=args.native_fps and source == args.video,
                 output_height=args.output_height)
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
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("FFmpeg/FFprobe are missing; install them and add to PATH")
        if not math.isfinite(args.threshold):
            raise ValueError("threshold must be finite")

        if args.command == "analyze":
            # analyze exists to produce artifacts, so it always writes them.
            analyze(args, verbose=True)
            return
        if args.command == "render":
            render(args, verbose=True)
            return

        # `run` analyses and renders in one process. Unless --verbose asks for
        # the artifacts, nothing intermediate is written into the output
        # directory: the analysis is handed to the renderer in memory, and the
        # normalized media lives in a scratch directory that is removed on the
        # way out.
        if args.verbose:
            analysis = analyze(args, verbose=True)
            render(args, analysis=analysis, verbose=True)
            return

        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "vertical.mp4").exists():
            raise ValueError("vertical.mp4 already exists; move it before rendering again")
        scratch = Path(tempfile.mkdtemp(prefix="autoclip-", dir=args.output))
        try:
            analysis = analyze(args, work_dir=scratch, verbose=False)
            render(args, analysis=analysis, verbose=False)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        raise SystemExit(f"autoclip: {exc}") from exc
