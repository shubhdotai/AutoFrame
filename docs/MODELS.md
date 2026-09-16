# Models used by AutoClip

| File | Role | Source |
| --- | --- | --- |
| `models/pretrain_AVA.model` | Default LR-ASD speaker scoring checkpoint | [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) |
| `models/yolov8x_person_face.pt` | Optional YOLO face/person detector | [YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector) |

ASD runs in PyTorch on MPS, CUDA or CPU. Apple Vision is the default Mac face
detector and uses no downloaded weights. YOLO detects boxes; the application
performs tracking separately. The manifest contains the exact sizes, hashes
and original download URLs of these two files.

See [the combined model card](HF_MODEL_CARD.md) for model descriptions,
limitations, citations and per-file licenses. See [the upload guide](HUGGINGFACE.md)
for publishing both files to `shubhdotai/autoclip` and downloading them again.

The AVA checkpoint loads strictly into the original LR-ASD architecture.
Other historical checkpoints and experiments in the original workspace are
not part of the two-model release. The filtered inventory JSON files retain
background information on the remaining original source/model artifacts;
they are not the download manifest.
