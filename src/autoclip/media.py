"""Shared media helpers for the ASD pipeline."""

import os
import subprocess


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_ffmpeg(args):
    cmd = ["ffmpeg", "-y", *args, "-loglevel", "error"]
    subprocess.run(cmd, check=True)


def preprocess_video(video_path, out_dir, threads=8, fps=25):
    """Create a constant-fps MP4 and 16 kHz mono WAV in out_dir."""
    ensure_dir(out_dir)
    video_out = os.path.join(out_dir, "video_25fps.mp4")
    audio_out = os.path.join(out_dir, "audio_16khz.wav")

    run_ffmpeg([
        "-i", video_path,
        "-r", str(fps),
        "-an",
        "-threads", str(threads),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        video_out,
    ])
    try:
        run_ffmpeg([
            "-i", video_path,
            "-ac", "1",
            "-vn",
            "-threads", str(threads),
            "-ar", "16000",
            audio_out,
        ])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not extract audio from {video_path}. The input video must "
            "contain an audio stream for active-speaker detection."
        ) from exc
    return video_out, audio_out


def mux_audio(video_path, audio_path, out_path):
    run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ])
