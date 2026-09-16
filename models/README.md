# Local model storage

Run `python scripts/download_models.py` from the repository root for LR-ASD.
Use `--include-yolo` to add the optional face/person detector.
`manifest.json` pins URLs and expected SHA-256 hashes. Existing matching
files are reused; mismatched files are not overwritten.

Binaries are ignored by Git. `legacy_coreml/` contains local original exports
for comparison only; generate corrected exports into `coreml/` using
`scripts/export_coreml.py`. See `../docs/MODELS.md` for every original artifact.
