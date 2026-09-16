# AutoClip

Turn a 16:9 landscape video into a vertical video that follows the active speaker.
AutoClip detects faces, tracks them within each scene, scores who is speaking
with LR-ASD, and holds a stable crop around the selected person.

The output keeps the full video timeline and original source audio (re-encoded
to AAC). There is no clip selection, transcription, LLM, subtitle or overlay
stage in the normal reframing path.

## Quick start — Apple Silicon Mac

Install Python 3.12 and FFmpeg (for example, `brew install python@3.12 ffmpeg`).
From this repository's root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mac]'
python scripts/download_models.py

autoclip doctor
autoclip run /path/to/video.mp4 --output out/my-video
```

Open `out/my-video/vertical.mp4`. Inputs must contain audio.
The largest even-sized exact 9:16 crop is used (for example, 1080p input
produces 594×1056), with no upscaling. Frame rate is normalized to 25 fps. Use a new output
directory for each analysis; existing runs are never silently overwritten.
Apple Vision uses the detector provided by macOS and needs no face weight file.
The default ASD checkpoint download is about 3.4 MB and is checksum-verified.

This folder is standalone: it can be copied out of the original workspace.
Locally supplied weights in `models/` are ignored by Git; a fresh clone obtains
them using the download command above. Run commands from the repository root,
or provide explicit model paths.

## Linux / Windows / CUDA

Use the optional YOLO face-and-person backend instead of Apple Vision:

```sh
python -m pip install -e '.[yolo]'
python scripts/download_models.py --include-yolo
autoclip doctor --detector yolo
autoclip run /path/to/video.mp4 --detector yolo --output out/my-video
```

Install FFmpeg using your platform's package manager and ensure it is on PATH.
`--device auto` chooses MPS, CUDA, then CPU. Use `--device cpu` explicitly for
CPU runs. CUDA requires a compatible PyTorch installation. These non-Mac paths
are provided but have not been exercised on Linux/Windows/CUDA hardware.
The optional YOLO dependency and weights have separate terms; see
[model documentation](docs/MODELS.md).

## Analyze once, render again

```sh
autoclip analyze /path/to/video.mp4 --output out/analysis
autoclip render /path/to/video.mp4 --output out/analysis
```

Keep `_work/` until you finish rendering: it contains the normalized video,
audio and per-track crops. `--debug-video` adds separate diagnostic face/speaker videos; the final vertical
video has no overlays.
`autoclip --help` and `autoclip run --help` list supported options.

## Project map

| Path | Purpose |
| --- | --- |
| `src/autoclip/` | Supported Python CLI and processing pipeline |
| `src/autoclip/model/` | Original LR-ASD architecture and checkpoint names |
| `models/` | Local weights, download manifest; binaries excluded from Git |
| `scripts/` | Verified-weight downloader and Core ML exporter |
| `research/tracking/` | Separate ByteTrack + face identity experiment |
| `tests/` | Tracking, framing, inference and media regression tests |
| `docs/` | Architecture, model inventory, migration audit and validation record |

## Long videos and limitations

Video frames are decoded in windows, with one encoder pass per face track and
cached embeddings reused for 2/4/6-second detector passes. This avoids loading
a whole video as an image tensor. Metadata, audio, embeddings and frame plans
still grow with duration; the whole pipeline is **not constant-memory**. A
60-minute performance or accuracy guarantee has not been established here.
See [architecture](docs/ARCHITECTURE.md) and [validation](docs/VALIDATION.md).

The supported pipeline uses **PyTorch** (MPS/CUDA/CPU), not MLX. The original workspace also contained a Core ML/Swift variant and MLX
experiments, inventoried in the migration audit but excluded from this focused
release. Mainline tracking is scene-aware IoU. ByteTrack is preserved as
research, not silently substituted into the ASD pipeline.

## Development and attribution

```sh
python -m pip install -e '.[mac,dev]'
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [migration audit](docs/MIGRATION.md),
[model inventory](docs/MODELS.md).
Derived LR-ASD code retains its upstream MIT notice in [LICENSE](LICENSE).
Credit and paper citations are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
