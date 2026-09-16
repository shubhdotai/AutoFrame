# Contributing

Use Python 3.12 and an isolated virtual environment. Install `.[dev]` (plus
`mac` or `yolo` for your detector) and run `pytest` from the repository root.
Keep fixes small and explain the input that failed and the new behavior.

Keep runtime code under `src/autoclip`; place optional experiments under
`research`. Do not add weights, private videos, transcripts, credentials,
output folders or downloaded dependency trees to source control.

When changing inference, verify checkpoint compatibility with strict loading
and test real tensor shapes. Do not change normalization, temporal padding,
pooling, score semantics or crop conventions without an explicit accuracy
comparison. Windowed inference is not numerically identical to full-sequence
inference. Benchmark long videos with duration, face-track count, resolution,
hardware, peak memory and elapsed time; separate preprocessing from inference.

CI runs model-free tests on Linux and macOS. Model smoke tests run locally
when `models/pretrain_AVA.model` is available. Optional Core ML export is separate from Python CI; see `docs/VALIDATION.md` for actual local results.
