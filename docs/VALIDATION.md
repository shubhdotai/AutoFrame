# Validation record

Validated on 2026-09-16 using Python 3.12.9 on macOS
26.5.2 (arm64). Dependencies were installed into
a temporary environment, not into the original project.

## Verified

- Package installation using the declared `mac,dev` extras.
- Six regression tests: per-frame track uniqueness and scene reset; exact 9:16
  crop bounds; held-subject reset across cuts; speaker-switch dwell; real
  checkpoint forward shapes and detector batch equivalence; actual FFmpeg
  render duration, codec and audio retention (including short audio).
- Both AVA and TalkSet checkpoints load strictly into the corrected architecture.
- The default checkpoint was downloaded into an empty temporary directory and
  matched the pinned SHA-256 hash. Existing-checkpoint reuse also passed.
- `autoclip doctor` passes from the clean repository root.
- Real 1920×1080 input smoke test with locally available video and companion
  audio, combined into a temporary MP4. Both CPU and automatic MPS paths ran
  the full pipeline: 76 sampled frames, 91 face detections, 4 scenes, 4 tracks,
  real encoder/detector inference, and full-video rendering.
- Output: **594×1056, H.264 + AAC, 152 frames at 25 fps, 6.08 seconds**.
  The input normalized timeline was also 152 frames / 6.08 seconds. One output
  frame was visually inspected; no subtitle or diagnostic overlay is present.
- CPU/MPS maximum absolute difference across smoke-test speaker logits:
  **1.37091e-06**. This is one short test, not a general accuracy guarantee.
- Wheel and source distribution build successfully. Wheel import/CLI help were
  checked from a temporary directory outside the source checkout. Distribution
  contents exclude model weights and private media; the source distribution
  includes the downloader and model manifest.
- The retained original model artifacts are size/hash-inventoried. ONNX
  graphs and tensor checkpoints were inspected. Duplicate weights are recorded
  in `MODELS.md`. Inspection does not establish numerical equivalence.

## Findings and remaining limits

The initial sandboxed Apple Vision call could not access the macOS Neural
Engine service. The inherited wrapper ignored the returned error; it now
raises an explicit failure. The successful CPU and MPS smoke tests ran outside
the sandbox with access to the normal macOS service. Run this application in a
normal terminal; a restricted host can block Vision independently of Python.

A 60-minute input was **not** benchmarked. Peak memory, scene-heavy long-video
throughput and speaker-selection accuracy still require measurement on
representative material. The short smoke test is not an ASD accuracy benchmark.

Linux, Windows, CUDA and YOLO inference were not runtime-tested. The optional
YOLO downloader is hash-pinned but was not fetched in this verification.
Legacy native/MLX/LLM workflows are not
included in this reframing-only distribution. Research ByteTrack/FaceLiVT code
is preserved separately and was not promoted into the main pipeline.

CI configuration is supplied but has not been executed on GitHub. Local
regression and packaging results above are the actual evidence.

## Tested direct dependencies

| Package | Version |
| --- | --- |
| torch | 2.14.0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| opencv-python | 4.14.0.94 |
| scenedetect | 0.6.7.1 |
| python-speech-features | 0.6 |
| pyobjc-framework-Vision | 12.2.2 |
| pytest | 9.1.1 |


## Simplified checkpoint release

After restoring the single PyTorch inference path, all 51 pipeline tests passed
with normal macOS video-encoder access. Three additional downloader tests passed
for checksum rejection/temporary-file cleanup, preserving mismatched existing
files, and selecting upstream versus `shubhdotai/autoclip` URLs. Both local
release checkpoints matched the manifest hashes. Model-card YAML and CLI
download syntax were checked. No files were uploaded to Hugging Face, and
downloads from the user's mirror have not been verified before publication.
