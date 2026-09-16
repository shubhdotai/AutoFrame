"""Shared media helpers for the ASD pipeline.

Everything that shells out to FFmpeg lives here. Two rules hold throughout:
decode the source as few times as possible, and never encode an intermediate
that is only going to be decoded again.
"""

import functools
import json
import os
import shutil
import subprocess
import sys

import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_ffmpeg(args):
    cmd = ["ffmpeg", "-y", *args, "-loglevel", "error"]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def ffprobe(video_path):
    """Return the parsed ffprobe JSON for a media file."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video_path),
    ])
    return json.loads(out)


def _fraction(text, default=0.0):
    try:
        num, _, den = str(text).partition("/")
        den = float(den) if den else 1.0
        return float(num) / den if den else default
    except (TypeError, ValueError):
        return default


def probe_video(video_path):
    """
    Authoritative source metadata.

    cv2's CAP_PROP_FRAME_COUNT is a container hint and is wrong often enough
    that frame bookkeeping should not depend on it.
    """
    info = ffprobe(video_path)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {video_path}")

    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    duration = float(video.get("duration") or info.get("format", {}).get("duration") or 0.0)
    n_frames = int(video.get("nb_frames") or 0)
    if n_frames <= 0 and fps > 0 and duration > 0:
        n_frames = int(round(duration * fps))

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(fps),
        "n_frames": int(n_frames),
        "duration": float(duration),
        "has_audio": audio is not None,
        "codec": video.get("codec_name"),
    }


# ---------------------------------------------------------------------------
# Encoder selection
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _available_encoders():
    if not shutil.which("ffmpeg"):
        return frozenset()
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL
        ).decode("utf-8", "replace")
    except subprocess.CalledProcessError:
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])
    return frozenset(names)


def resolve_encoder(preference="auto", quality="analysis"):
    """
    Pick an H.264 encoder for a given job.

    Under `auto` the two jobs want different things. The analysis intermediate
    is throughput-bound and its pixels are only ever read by a detector, so it
    takes the hardware encoder: VideoToolbox costs roughly a sixth of the CPU
    of libx264 on Apple Silicon (measured 17 s vs 104 s on a 147 s clip), and
    the cores it frees are the ones running face detection.

    The deliverable is the opposite. It is encoded once, at crop resolution,
    and it is what the viewer sees, so it takes libx264 -- which at matched
    visual quality produced 2.5 Mbps against VideoToolbox's 4.4 Mbps on the
    same footage. Pass `hardware` explicitly to override.
    """
    encoders = _available_encoders()
    if preference not in ("auto", "hardware", "software"):
        if preference not in encoders:
            raise ValueError(f"encoder {preference!r} is not available in this FFmpeg build")
        return preference
    if preference == "software":
        return "libx264"
    hardware = "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"
    if preference == "hardware":
        if hardware not in encoders:
            raise ValueError(f"{hardware} is not available in this FFmpeg build")
        return hardware
    if quality == "final":
        return "libx264"
    if sys.platform == "darwin" and hardware in encoders:
        return hardware
    return "libx264"


def encoder_args(encoder, quality="analysis"):
    """Quality knobs differ per encoder; keep the mapping in one place."""
    if encoder.endswith("videotoolbox"):
        # -q:v is 1..100 for VideoToolbox, higher is better.
        return ["-c:v", encoder, "-q:v", "70" if quality == "analysis" else "80"]
    if encoder.endswith("nvenc"):
        return ["-c:v", encoder, "-preset", "p4", "-cq", "20" if quality == "analysis" else "19"]
    return [
        "-c:v", encoder, "-preset", "veryfast",
        "-crf", "20" if quality == "analysis" else "18",
    ]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_video(video_path, out_dir, threads=8, fps=25, encoder="auto"):
    """
    Create a constant-fps MP4 and a 16 kHz mono WAV in a single decode pass.

    The previous two-call version decoded the source twice. FFmpeg can emit
    both outputs from one input, so it now decodes once.
    """
    ensure_dir(out_dir)
    video_out = os.path.join(out_dir, f"video_{int(round(fps))}fps.mp4")
    audio_out = os.path.join(out_dir, "audio_16khz.wav")

    info = probe_video(video_path)
    if not info["has_audio"]:
        raise RuntimeError(
            f"Could not extract audio from {video_path}. The input video must "
            "contain an audio stream for active-speaker detection."
        )

    chosen = resolve_encoder(encoder, "analysis")
    run_ffmpeg([
        "-i", str(video_path),
        "-threads", str(threads),
        # video output: constant frame rate, no audio
        "-map", "0:v:0", "-an",
        "-vf", f"fps={fps}",
        *encoder_args(chosen, "analysis"),
        "-pix_fmt", "yuv420p",
        video_out,
        # audio output: 16 kHz mono PCM, no video
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        audio_out,
    ])
    return video_out, audio_out


def mux_audio(video_path, audio_path, out_path):
    run_ffmpeg([
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ])


# ---------------------------------------------------------------------------
# Raw frame sink
# ---------------------------------------------------------------------------

class FrameSink:
    """
    Pipe BGR frames straight into one H.264 encode.

    Writing an `mp4v` temp file and re-encoding it, as the renderer used to,
    costs two encodes and stacks two generations of lossy compression on the
    deliverable. Feeding rawvideo over stdin costs one encode and none.
    """

    def __init__(
        self,
        out_path,
        width,
        height,
        fps,
        audio_path=None,
        audio_duration=None,
        out_size=None,
        encoder="auto",
        quality="final",
        faststart=True,
    ):
        self.width = int(width)
        self.height = int(height)
        chosen = resolve_encoder(encoder, quality)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{float(fps):.6f}",
            "-i", "-",
        ]
        if audio_path is not None:
            cmd += ["-i", str(audio_path)]
        cmd += ["-map", "0:v:0"]
        if audio_path is not None:
            cmd += ["-map", "1:a:0", "-c:a", "aac"]
            if audio_duration is not None:
                # Pad short audio out to the video's length. `apad` generates
                # silence forever, so it is only safe paired with the `-t`
                # below -- without one, FFmpeg never reaches end of stream and
                # the encode hangs.
                cmd += ["-af", "apad"]
            else:
                cmd += ["-shortest"]
        if out_size is not None and tuple(out_size) != (self.width, self.height):
            ow, oh = int(out_size[0]), int(out_size[1])
            cmd += ["-vf", f"scale={ow}:{oh}:flags=lanczos"]
        cmd += [*encoder_args(chosen, quality), "-pix_fmt", "yuv420p"]
        if audio_duration is not None:
            cmd += ["-t", f"{float(audio_duration):.6f}"]
        if faststart:
            cmd += ["-movflags", "+faststart"]
        cmd += ["-loglevel", "error", str(out_path)]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames_written = 0

    def write(self, frame):
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, sink expects "
                f"{self.width}x{self.height}"
            )
        # A crop is a view with a row stride; tobytes() on it would be wrong.
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        code = self.proc.wait()
        self.proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed with exit code {code}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None
            return False
        self.close()
        return False
