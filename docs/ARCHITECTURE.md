# How AutoClip works

```text
video + audio
  -> FFmpeg, ONE decode: constant 25 fps video + 16 kHz mono audio
  -> ONE decode: PySceneDetect scene cuts AND sampled face boxes, in parallel
  -> scene-aware tracking: predicted-position IoU + optimal assignment
  -> gating: which tracks can actually change the crop
  -> ONE decode: 112x112 face crops streamed into a batched encoder
  -> original LR-ASD encoders: audio MFCC + visual embeddings
  -> batched temporal detector at 2, 4, 6 seconds (clamped); mean score
  -> speaker selection + dwell time + stable per-shot crop
  -> H.264/AAC vertical MP4, encoded once
```

## Face detection backends

The default backend on macOS is **Apple's Vision framework**
(`VNDetectFaceRectanglesRequest`), supplied by the operating system. There is
no face weight file to ship or download; the only checkpoint AutoClip needs is
LR-ASD's. Frames reach Vision as a CVPixelBuffer wrapping the decoded BGRA
bytes, which is both lossless and far cheaper than encoding an image first.

The YOLO face-and-person backend is the cross-platform alternative, and the
only option off macOS. It additionally returns person boxes, which the
reframer uses as a fallback subject when no face is detected.

Both expose the same detector interface -- `detect(frames, indices) ->
(faces, persons)` plus a `batch_size` -- so `scan.scan_video` drives either one
from the same decode loop.

## Ownership

`cli.py` owns input validation and public commands. `run.py` orchestrates
analysis and reports a per-stage time profile. `media.py` invokes FFmpeg with
argument lists and owns encoder selection and the raw-frame sink. `scan.py`
runs the shared decode loop that feeds scene detection and face detection.
`scenes.py` keeps a standalone scene detector. The two `face_detection*`
modules expose a common detector interface. `tracking.py` builds and
interpolates tracks within each scene. `gating.py` decides which tracks are
worth scoring. `cropping.py` produces per-track crops in one sequential pass.
`inference.py` holds the streaming batched encoder and the detector replay.
`model/` retains the upstream network architecture, including normalization,
strides and BatchNorm settings. `ASD.py` strictly loads the complete state
dict. `report.py` and `visualize.py` provide diagnostics. `shortform.py` plans
and renders the entire video without overlays or clipping.

## Decode passes

There are three decodes of the normalized video and one of the source:
preprocessing, the fused scene/face scan, and the crop pass; the renderer reads
either the normalized copy or the original. Earlier versions decoded six times
and encoded the full-length video three times. Nothing is written to disk that
is only going to be read back by the next stage: face crops travel as arrays,
and the renderer pipes raw frames into a single H.264 encode instead of
writing an `mp4v` intermediate and transcoding it.

The scan loop is a three-stage pipeline. Decode, scene detection and face
detection each run on their own thread with bounded queues, because decode and
detection have comparable throughput and running them in lockstep wastes about
half the wall clock. Queue depth is derived from frame size so a 4K input does
not balloon memory.

## Time and score contracts

All analysis frame indices refer to normalized **25 fps** media, not original
variable-rate frame indices. Audio uses 100 MFCC frames per second and four
MFCC frames per video frame. **25 fps is not a tunable**: the checkpoint's
temporal kernels and the four-to-one audio ratio are both trained at it.
Sampling rates that *are* tunable are the face detector's (`--sample-every`),
which tracking interpolates over, and the renderer's output frame rate
(`--native-fps`).

The encoder window must be a positive multiple of four frames; the default is
100. Windows carry a nine-frame margin on each side which is computed and then
discarded, nine being the combined temporal receptive-field radius of the three
encoder blocks. Chunked encoding is therefore exact: it reproduces a single
unbroken pass over the track, and a regression test asserts this at several
window and batch sizes. Batches group windows of equal length, because
`forward_visual_frontend` subtracts a mean and divides by a standard deviation,
so a zero-padded window would enter the convolutions at -2.465 rather than at
the zero that convolution padding supplies.

The score is **class-1 speaking logit**, not a probability and not the
speaking-minus-silent margin. Default threshold is zero. Detector chunk lengths
are clamped to the track length and the tail chunk slides back to fit, so the
bidirectional GRU never reads zero padding; asking for a 6 s window on a 31
frame track previously meant 79% of the sequence was fabricated. Changing
architecture or converting precision can affect calibration; accuracy is not
inferred from successful weight loading.

## What gets scored

A track is scored only when its speaking logit can change the rendered crop.
Read `shortform._per_frame_intent`: when one track is live in a frame, the
speaker, hold and largest-face branches all resolve to the same centre. The
same holds when several faces sit close enough that every candidate produces
the same crop rectangle -- with a 9:16 window over a 16:9 frame that rectangle
is more than half the frame width. Tracks that overlap no contested frame, and
tracks whose audio never rises above the silence floor, are skipped.

Skipped tracks carry **NaN**, not zero. NaN compares false against any
threshold, so a skipped track is never reported as speaking, while remaining
fully available to the crop as a visible subject. `scores.json` records
`scored` and `skip_reason` per track, and `results.txt` marks them `not
scored`. `--score-all-tracks` disables the gate when a complete speaking
report matters more than run time.

## Framing

The crop follows active speaker -> visible previous subject -> largest face ->
largest person box -> held previous position or centre. Speaker changes need
sustained dominance (default 0.4 seconds), and `--speaker-margin` can also
require a lead over the runner-up before the camera moves. Scene boundaries
reset selection state. Within a shot the median target locks the crop, on both
axes, to reduce jitter; `--motion follow` instead eases toward a subject who
leaves a deadzone, for footage where people move within a long take.

Vertical placement is anchored on the subject rather than pinned to row 0,
which matters whenever the exact-aspect crop is shorter than the source frame.
Person boxes from the YOLO backend are a framing fallback for frames with no
detected face; they do not drive object-aware reframing.

## Memory and disk

The encoder holds at most one window plus its margins per open track (about
1.5 MB at the default window) and two 128-dimensional float32 embedding
streams per track, which is about 92 MB for one 60-minute track at 25 fps.
Face crops are never materialized for a whole track and never written to disk.
Audio PCM, MFCCs, Python detection objects and per-frame plans still grow with
duration; the pipeline is **not constant-memory**.

`_work/` holds only the normalized video and audio. It is required for later
renders and can be removed after final exports. There is no checkpoint/resume
mechanism for interrupted analysis yet.

The renderer muxes audio from the **original source**, so ASD's 16 kHz mono
waveform does not determine final audio quality. Video is H.264 and audio is
AAC. Under `auto`, the analysis intermediate uses the hardware encoder for
throughput and the deliverable uses libx264 for bitrate efficiency. The
timeline covers all normalized input frames. If audio ends early it is padded
with silence, not used to shorten the video. The crop is the largest exact even
9:16 rectangle that fits the source; a small strip can be removed to satisfy
the integer aspect ratio. There is no upscaling unless `--output-height` asks
for it.

## What a run writes

`autoclip run` produces only `vertical.mp4` unless `--verbose` is passed. The
analysis is handed to the renderer in memory rather than round-tripped through
JSON, and the normalized media is written to a scratch directory inside the
output folder that is removed when the run finishes. The scratch directory sits
next to the output rather than in the system temp so that a 70 MB working copy
lands on the volume the caller chose.

Under `--verbose` the same run additionally writes every artifact below, keeps
`_work/`, and prints a per-stage timing profile. Rendered video is identical in
both modes. `analyze` and `render` exist specifically to write and to read
artifacts, so they always do; `--verbose` has no meaning for them.

## Python run artifacts

These exist under `--verbose`, and after `autoclip analyze`.

| File | Meaning |
| --- | --- |
| `run.json` | Source path, backend, stage timings, artifact index |
| `scenes.json`, `cuts.json` | Scene metadata and first frame of each new scene |
| `face_detections.json` | Normalized media dimensions, fps, detections, person boxes |
| `tracks.json` | Dense frame indices, boxes, scene bounds, mean confidence |
| `scores.json` | Class-1 logits per track; `null` where a track was not scored |
| `crops.json` | Per-track crop geometry and whether the track was scored |
| `results.json`, `results.txt`, `trims.json` | Speaking segments and summaries |
| `shortform_plan.json` | Crop rectangle and selected subject per frame |
| `_work/` | Normalized video and audio only |
| `vertical.mp4` | Final H.264/AAC export |
