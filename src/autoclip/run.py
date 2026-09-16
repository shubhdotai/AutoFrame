"""
End-to-end CLI for the optimized ASD pipeline.

Usage:
    python -m autoclip.run --videoPath path/to/video.mp4

Outputs are written to ./out by default. Preprocessed media and per-track
model crops are kept under ./out/_work so every runtime artifact stays inside
the project output folder.
"""

import argparse
import json
import os
import time

import numpy as np


from autoclip.cropping import crop_face_track
from autoclip.face_detection import detect_faces_in_video as detect_faces_apple_vision
from autoclip.face_detection_yolo import detect_faces_in_video as detect_faces_yolo
from autoclip.media import ensure_dir, preprocess_video
from autoclip.report import write_report
from autoclip.scenes import detect_scenes, scene_json
from autoclip.tracking import track_faces
from autoclip.visualize import (
    render_active_speaker_video,
    render_raw_detections_video,
)


def _parse_durations(value):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _default_weights_path():
    return os.path.abspath("models/pretrain_AVA.model")


def _project_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _tracks_json(tracks, params):
    return {
        "scene_aware": True,
        "params": params,
        "tracks": [
            {
                "track_id": idx,
                "scene_start_frame": int(track.get("scene_start_frame", 0)),
                "scene_end_frame": int(track.get("scene_end_frame", track["frame"][-1])),
                "frame": track["frame"].tolist(),
                "bbox": track["bbox"].tolist(),
            }
            for idx, track in enumerate(tracks)
        ],
    }


def _crop_json(tracks, fps, crop_scale):
    crops = []
    for idx, track in enumerate(tracks):
        frames = track["frame"]
        bboxes = np.asarray(track["bbox"], dtype=float)
        crops.append({
            "track_id": idx,
            "video_path": f"_work/crops/track_{idx:05d}.mp4",
            "audio_path": f"_work/crops/track_{idx:05d}.wav",
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_time_s": round(float(frames[0]) / fps, 3),
            "end_time_s": round(float(frames[-1] + 1) / fps, 3),
            "frame_count": int(len(frames)),
            "crop_size": [224, 224],
            "crop_scale": float(crop_scale),
            "avg_bbox": [float(v) for v in bboxes.mean(axis=0)],
            "persisted_media": True,
        })
    return {
        "note": "Per-track crop MP4/WAV files are model inputs stored under _work/crops.",
        "crops": crops,
    }


def _scores_json(tracks, scores, fps):
    return {
        "fps": float(fps),
        "scores": [
            {
                "track_id": idx,
                "start_frame": int(track["frame"][0]) if len(track["frame"]) else None,
                "values": score.tolist(),
            }
            for idx, (track, score) in enumerate(zip(tracks, scores))
        ],
    }


def _trim_json(report):
    trims = []
    for track in report["tracks"]:
        for segment in track["speaking_segments"]:
            trims.append({"track_id": track["track_id"], **segment})
    return {"trims": trims}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimized LR-ASD pipeline")
    p.add_argument("--videoPath", required=True, help="Input video path")
    p.add_argument(
        "--savePath", "--outDir",
        default="out",
        dest="savePath",
        help="Output run directory (default: out)",
    )
    p.add_argument(
        "--pretrainModel",
        default=_default_weights_path(),
        help="LR-ASD weights path. Defaults to models/pretrain_AVA.model.",
    )

    p.add_argument(
        "--faceDetector",
        default="apple_vision",
        choices=["apple_vision", "yolo_face_person"],
        help="Face detector backend. apple_vision = VNDetectFaceRectanglesRequest. "
             "yolo_face_person = iitolstykh/YOLO-Face-Person-Detector (face + person).",
    )
    p.add_argument("--sampleEvery", type=int, default=2,
                   help="Run face detection on every Nth frame")
    p.add_argument("--numFaceDetWorkers", type=int, default=8,
                   help="apple_vision only: thread pool size")
    p.add_argument("--faceDetChunk", type=int, default=32,
                   help="apple_vision only: streaming chunk size")
    p.add_argument("--yoloWeights",
                   default="models/yolov8x_person_face.pt",
                   help="yolo_face_person only: path to YOLO weights")
    p.add_argument("--yoloBatch", type=int, default=16,
                   help="yolo_face_person only: inference batch size")
    p.add_argument("--yoloImgsz", type=int, default=640,
                   help="yolo_face_person only: input image size")
    p.add_argument("--yoloConf", type=float, default=0.4,
                   help="yolo_face_person only: confidence threshold")
    p.add_argument("--yoloIou", type=float, default=0.7,
                   help="yolo_face_person only: NMS IoU threshold")

    p.add_argument("--sceneThreshold", type=float, default=27.0,
                   help="PySceneDetect ContentDetector threshold")
    p.add_argument("--minSceneLen", type=int, default=15,
                   help="Minimum scene length in frames")

    p.add_argument("--minTrack", type=int, default=10)
    p.add_argument("--numFailedDet", type=int, default=10)
    p.add_argument("--minFaceSize", type=int, default=1)
    p.add_argument("--iouThreshold", type=float, default=0.5)

    p.add_argument("--cropScale", type=float, default=0.40)

    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--durationSet", default="2,4,6")
    p.add_argument("--encoderWindowFrames", type=int, default=100)
    p.add_argument("--detectorBatch", type=int, default=64)

    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--smoothingWindow", type=int, default=5)

    p.add_argument("--noVisualize", action="store_true")
    p.add_argument("--bboxSmoothKernel", type=int, default=13)
    p.add_argument("--ffmpegThreads", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    duration_set = _parse_durations(args.durationSet)
    out_dir = ensure_dir(os.path.abspath(_project_path(args.savePath)))
    weights_path = os.path.abspath(_project_path(args.pretrainModel))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"missing LR-ASD weights: {weights_path}")

    print("=" * 70)
    print(f"input:           {args.videoPath}")
    print(f"output dir:      {out_dir}")
    print(f"weights:         {weights_path}")
    print(f"device:          {args.device}")
    print(f"sample 1-in-N:   {args.sampleEvery}")
    print(f"durationSet:     {duration_set}")
    print("=" * 70)

    work_dir = ensure_dir(os.path.join(out_dir, "_work"))
    crops_dir = ensure_dir(os.path.join(work_dir, "crops"))

    print("\n[1/8] preprocessing video -> 25 fps MP4 + 16 kHz audio")
    t0 = time.perf_counter()
    video25, audio_path = preprocess_video(
        args.videoPath,
        work_dir,
        threads=args.ffmpegThreads,
        fps=25,
    )
    print(f"      took {time.perf_counter()-t0:.1f}s")

    print("\n[2/8] detecting scenes with PySceneDetect")
    t0 = time.perf_counter()
    scenes, cuts = detect_scenes(
        video25,
        threshold=args.sceneThreshold,
        min_scene_len=args.minSceneLen,
        progress=True,
    )
    scenes_data = scene_json(
        video25,
        scenes=scenes,
        cuts=cuts,
        threshold=args.sceneThreshold,
        min_scene_len=args.minSceneLen,
    )
    _write_json(os.path.join(out_dir, "scenes.json"), scenes_data)
    _write_json(os.path.join(out_dir, "cuts.json"), {"cuts": cuts})
    print(f"      {len(scenes)} scenes, {len(cuts)} cuts | {time.perf_counter()-t0:.1f}s")

    print(f"\n[3/8] face detection ({args.faceDetector})")
    t0 = time.perf_counter()
    if args.faceDetector == "apple_vision":
        fd_result = detect_faces_apple_vision(
            video25,
            sample_every_n=args.sampleEvery,
            num_workers=args.numFaceDetWorkers,
            chunk_size=args.faceDetChunk,
        )
    else:
        yolo_weights = os.path.abspath(_project_path(args.yoloWeights))
        fd_result = detect_faces_yolo(
            video25,
            weights_path=yolo_weights,
            sample_every_n=args.sampleEvery,
            batch_size=args.yoloBatch,
            device=args.device,
            conf=args.yoloConf,
            iou=args.yoloIou,
            imgsz=args.yoloImgsz,
        )
    persons_per_frame = fd_result.get("persons_per_frame")
    fd_json = {
        "detector": args.faceDetector,
        "fps": fd_result["fps"],
        "width": fd_result["width"],
        "height": fd_result["height"],
        "total_frames": fd_result["total_frames"],
        "sample_every_n": fd_result["sample_every_n"],
        "detections": fd_result["detections_per_frame"],
    }
    if persons_per_frame is not None:
        fd_json["persons"] = persons_per_frame
    _write_json(os.path.join(out_dir, "face_detections.json"), fd_json)
    print(f"      face detection took {(time.perf_counter()-t0)/60:.2f} min")

    if not args.noVisualize:
        print("      rendering raw detection bbox video")
        t_vis = time.perf_counter()
        raw_det_path = os.path.join(out_dir, "face_detections_debug.mp4")
        render_raw_detections_video(
            input_video_path=video25,
            audio_path=audio_path,
            detections_per_frame=fd_result["detections_per_frame"],
            persons_per_frame=persons_per_frame,
            out_path=raw_det_path,
            fps=25,
        )
        print(f"      bbox video -> {raw_det_path} | {(time.perf_counter()-t_vis)/60:.1f} min")

    print("\n[4/8] scene-aware IoU face tracking")
    t0 = time.perf_counter()
    track_params = {
        "num_failed_det": args.numFailedDet,
        "min_track": args.minTrack,
        "min_face_size": args.minFaceSize,
        "iou_threshold": args.iouThreshold,
    }
    tracks = track_faces(
        fd_result["detections_per_frame"],
        scene_cuts=cuts,
        num_failed_det=args.numFailedDet,
        min_track=args.minTrack,
        min_face_size=args.minFaceSize,
        iou_threshold=args.iouThreshold,
    )
    _write_json(os.path.join(out_dir, "tracks.json"), _tracks_json(tracks, track_params))
    print(f"      {len(tracks)} tracks | {time.perf_counter()-t0:.1f}s")

    print("\n[5/8] creating temporary per-track model crops")
    t0 = time.perf_counter()
    cropped = []
    for idx, track in enumerate(tracks):
        prefix = os.path.join(crops_dir, f"track_{idx:05d}")
        mp4_path, wav_path = crop_face_track(
            video25,
            audio_path,
            track,
            prefix,
            fps=25,
            crop_scale=args.cropScale,
            n_threads=args.ffmpegThreads,
        )
        cropped.append((mp4_path, wav_path))
        print(f"      track {idx:03d}: {len(track['frame'])} frames")
    _write_json(os.path.join(out_dir, "crops.json"), _crop_json(tracks, 25, args.cropScale))
    print(f"      crop prep took {(time.perf_counter()-t0)/60:.2f} min")

    print(f"\n[6/8] LR-ASD inference (device={args.device})")
    from .inference import load_model, run_asd_on_track

    t0 = time.perf_counter()
    model = load_model(weights_path, device=args.device)
    print(f"      model loaded in {time.perf_counter()-t0:.1f}s")

    all_scores = []
    for idx, (mp4_path, wav_path) in enumerate(cropped):
        scores = run_asd_on_track(
            model,
            video_path=mp4_path,
            audio_path=wav_path,
            duration_set=duration_set,
            encoder_window_frames=args.encoderWindowFrames,
            detector_batch=args.detectorBatch,
            device=args.device,
            progress_label=f"track {idx:03d}",
        )
        all_scores.append(scores)
    _write_json(os.path.join(out_dir, "scores.json"), _scores_json(tracks, all_scores, 25))

    print("\n[7/8] writing reports")
    results_path = os.path.join(out_dir, "results.txt")
    report = write_report(
        out_path=results_path,
        json_path=os.path.join(out_dir, "results.json"),
        video_path=args.videoPath,
        tracks=tracks,
        scores=all_scores,
        fps=25,
        threshold=args.threshold,
        smoothing_window=args.smoothingWindow,
        frame_width=fd_result["width"],
    )
    _write_json(os.path.join(out_dir, "trims.json"), _trim_json(report))

    rendered_path = None
    if not args.noVisualize:
        print("\n[8/8] rendering debug active-speaker video")
        t0 = time.perf_counter()
        rendered_path = os.path.join(out_dir, "active_speaker_debug.mp4")
        render_active_speaker_video(
            input_video_path=video25,
            audio_path=audio_path,
            tracks=tracks,
            scores=all_scores,
            out_path=rendered_path,
            threshold=args.threshold,
            smoothing_window=args.smoothingWindow,
            bbox_smooth_kernel=args.bboxSmoothKernel,
            fps=25,
        )
        print(f"      took {(time.perf_counter()-t0)/60:.1f} min")

    _write_json(
        os.path.join(out_dir, "run.json"),
        {
            "source_video": os.path.abspath(args.videoPath),
            "output_dir": out_dir,
            "weights": weights_path,
            "face_detector": args.faceDetector,
            "artifacts": {
                "scenes": "scenes.json",
                "cuts": "cuts.json",
                "face_detections": "face_detections.json",
                "face_detections_debug_video": "face_detections_debug.mp4" if not args.noVisualize else None,
                "tracks": "tracks.json",
                "crops": "crops.json",
                "scores": "scores.json",
                "results": "results.json",
                "trims": "trims.json",
                "debug_video": "active_speaker_debug.mp4" if rendered_path else None,
                "work_dir": "_work",
            },
        },
    )

    print("\n" + "=" * 70)
    print("done.")
    print(f"  run dir: {out_dir}")
    print(f"  report:  {results_path}")
    if rendered_path:
        print(f"  debug:   {rendered_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
