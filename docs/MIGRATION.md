# Workspace audit and migration

Original files were left in place. `autoclip-open-source/` is an independent
copy, not a symlink or an overlay. No original Git changes were reset.
The original upstream checkout revision was
`1b6dcd2d8fc2895683de6508ec6294ec47d388ca`; it also contained uncommitted edits.

## What each original area contains

| Original path | Assessment | New home / decision |
| --- | --- | --- |
| Root `ASD.py`, `model/`, `train.py`, `dataLoader.py`, `Columbia_test.py`, `utils/` | LR-ASD research training/evaluation, with local MPS edits | Original architecture in `src/autoclip/model`; training/evaluation remain in original workspace |
| Root `optimized_asd/` | Earlier optimized pipeline; `pyavi/pywork` artifacts | Superseded by newer pipeline; documented rather than duplicated |
| `AutoClip/optimized_asd/` | More complete scene-aware Python pipeline, YOLO option, captioned rendering | Main foundation for `src/autoclip` |
| `AutoClip/optimized_asd/model/` | Rewritten encoders: different normalization, stride, pooling and BN epsilon | Replaced with upstream architecture; identical parameter names are insufficient |
| `AutoClip/optimized_asd/transcript_data.py` | A fixed transcript and clip selection for one video | Excluded; the new renderer processes the whole video |
| `AutoClip/optimized_asd/shortform_auto.py` | Python ASR/LLM orchestration experiment | Excluded per reframing-only scope |
| `AutoClip/optimized_asd/optimized_python/` | YOLO/ONNX benchmarks, face identity and tracking experiments | ByteTrack pipeline and required source retained in `research/tracking`; benchmarks stay original |
| `AutoClip/optimized_asd/tmps/`, `AutoClip/tmp/` | ASR/VLM/LLM scratch scripts, smoke clips, private environment file | Excluded; no environment-file values copied |
| `AutoClip/old/` | Earlier speaker/diarization/cut/depth experiments and vendored depth repos | Excluded from runtime; all model artifacts inventoried |
| `AutoClip/clipwell/` | Next.js landing site; payment/API folders are placeholders | Excluded from the video engine; remains intact in original workspace |
| `demo/`, `out/`, `results.txt`, plan JSON, caches, PDFs | Inputs, generated artifacts, explanatory material | Excluded from distribution |

## Corrections made while packaging

- Strict checkpoint loading; preserve original network forward operations.
- Device is chosen before model allocation; CPU does not allocate on MPS first.
- Imports resolve inside the installed package; no original parent checkout needed.
- Output/model paths are explicit or relative to the caller's working directory.
- Both debug renders are optional; normal runs avoid two unnecessary video exports.
- Renderer reuses normalized analysis media instead of preprocessing it twice.
- No hard-coded transcript, clip selection, subtitles, hooks or LLM dependency.
- H.264/AAC outputs; audio mux failures are errors, never silent success.
- A tracker cannot consume two face detections in the same frame for one track.
- Detector mini-batches do not allocate padding for the whole video.
- Reuse original source audio for final rendering; ASD audio is only model input.
- Exact 9:16 even pixel dimensions for H.264 without upscaling.

## Audit coverage and limits

`source-inventory.json` indexes source/document paths, line counts, and Python
top-level definitions across the original workspace, including vendored depth
code and the landing page. `model-inventory.json` records the retained
model artifacts, byte sizes and SHA-256 checksums. Generated dependency/build
folders were excluded. The pipeline and model call paths were inspected;
this is not a line-by-line correctness certification of every vendored repo.
No historical performance report is treated as a fresh benchmark.
