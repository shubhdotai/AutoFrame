# Model weights

The default checkpoint is `pretrain_AVA.model`. Add `yolov8x_person_face.pt`
only when using `--detector yolo`; macOS Vision needs no detector weight file.

Before the mirror is published, fetch from the original sources:

```sh
python scripts/download_models.py --source upstream --include-yolo
```

After uploading to `shubhdotai/autoclip`:

```sh
python scripts/download_models.py                 # ASD only
python scripts/download_models.py --include-yolo  # ASD and YOLO
```

Run these commands from the project root. Files are stored here, verified
against the SHA-256 hashes in `manifest.json`, and reused if already present.
Use `--list` to list the two models. Model binaries are excluded from Git.
See [the upload guide](../docs/HUGGINGFACE.md) and
[the model card](../docs/HF_MODEL_CARD.md) for provenance and licensing.
