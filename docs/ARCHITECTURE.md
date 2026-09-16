# How AutoClip works

```text
video + audio
  -> FFmpeg: constant 25 fps video, 16 kHz mono audio
  -> PySceneDetect: scene boundaries
  -> Vision or YOLO: sampled face boxes
  -> scene-aware IoU tracker: dense interpolated tracks
  -> 224x224 face crops; grayscale center crop to 112x112
  -> original LR-ASD encoders: audio MFCC + visual embeddings
  -> batched temporal detector at 2, 4, 6 seconds; mean score
  -> speaker selection + dwell time + stable per-shot crop
  -> H.264/AAC vertical MP4
```

## Ownership

`cli.py` owns input validation and public commands. `run.py` orchestrates
analysis. `media.py` invokes FFmpeg with argument lists. `scenes.py` detects
hard scene boundaries. The two `face_detection*` modules return a common
schema. `tracking.py` interpolates detections within each scene. `cropping.py`
prepares aligned per-track media. `inference.py` caches windowed embeddings.
`model/` retains the upstream network architecture, including normalization,
strides and BatchNorm settings. `ASD.py` strictly loads the complete state dict.
`report.py` and `visualize.py` provide diagnostics. `shortform.py` plans and
renders the entire video without overlays or clipping.

## Time and score contracts

All analysis frame indices refer to normalized **25 fps** media, not original
variable-rate frame indices. Audio uses 100 MFCC frames per second and four
MFCC frames per video frame. A short audio tail can make scores shorter than
the track; selection uses their shared prefix. The encoder window must be a
positive multiple of four frames. The default is 100 frames.

The score is **class-1 speaking logit**, not a probability and not the
speaking-minus-silent margin. Default threshold is zero. Changing architecture
or converting precision can affect calibration; accuracy is not inferred from
successful weight loading.

The crop follows active speaker -> visible previous subject -> largest face
-> held previous position or center. Speaker changes need sustained dominance
(default 0.4 seconds). Scene boundaries reset selection state. Within a shot,
the median target center locks the crop to reduce jitter. This deliberately
does not pan to follow large within-shot movement. Person boxes from YOLO are
only diagnostic; they do not drive object-aware reframing.

## Memory and disk

A 100-frame 112x112 float32 input is about 5 MB before network activations.
Two 128-dimensional float32 embedding streams require about 92 MB for one
60-minute track at 25 fps (90,000 frames). Audio PCM, MFCCs, Python detection
objects and per-frame plans consume additional duration-dependent memory.
Detector padding is allocated one mini-batch at a time.

There are several decode passes and persistent per-track crops. Disk usage
can exceed input size substantially when several faces overlap. `_work/` is
required for later renders and can be removed after final exports. There is
no checkpoint/resume mechanism for interrupted analysis yet.

The original Swift variant is inventoried but excluded from this reframing-only
release. Core ML export is kept as an optional development tool, not a second
supported inference backend.

The renderer reads normalized video but muxes audio from the **original source**,
so ASD's 16 kHz mono waveform does not determine final audio quality. Video is
H.264 and audio is AAC. The timeline covers all normalized input frames. If
audio ends early it is padded with silence, not used to shorten the video.
The crop is the largest exact even 9:16 rectangle that fits the source; a small
bottom strip can be removed to satisfy the integer aspect ratio. No upscaling.

## Python run artifacts

| File | Meaning |
| --- | --- |
| `run.json` | Source path, backend, artifact index |
| `scenes.json`, `cuts.json` | Scene metadata and first frame of each new scene |
| `face_detections.json` | Normalized media dimensions, fps and detections |
| `tracks.json` | Dense frame indices, boxes, scene bounds |
| `scores.json` | Class-1 logits aligned with each track start |
| `crops.json` | Per-track model media paths and timing |
| `results.json`, `results.txt`, `trims.json` | Speaking segments and summaries |
| `shortform_plan.json` | Crop and selected subject per frame |
| `_work/` | Normalized media and model crops |
| `vertical.mp4` | Final H.264/AAC export |
