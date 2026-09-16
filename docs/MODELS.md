# Model inventory

The original workspace contains 29 model artifacts including Core ML weight
blobs and an ONNX external tensor sidecar. All paths, byte sizes, full SHA-256
hashes and inspected tensor/graph metadata are in `model-inventory.json`.
Identical hashes below mean exact copies, not merely similar file names.

## Models that matter for reframing

| Model | Role | New package |
| --- | --- | --- |
| `pretrain_AVA.model` | LR-ASD audio/visual speaker scoring; AVA-pretrained checkpoint | Default, checksum-verified download |
| `finetuning_TalkSet.model` | Alternative LR-ASD checkpoint fine-tuned on TalkSet | Optional downloader flag `--include-talkset`, then CLI `--weights` |
| Apple Vision | OS-provided face detector; no standalone weight file | Default Mac detector |
| `yolov8x_person_face.pt` | Two-class face/person detector | Optional YOLO backend; not interchangeable with generic YOLOv8n |
| `facelivtv2-s.pt` | Face embedding/identity model | Research only; provenance unresolved |

The two ASD checkpoints each contain 168 state entries and load strictly into
the original LR-ASD architecture. The inference wrapper has 837,156 trainable
parameters including both heads. Root and nested copies of each checkpoint
are byte-identical. AVA and TalkSet are different checkpoints. The file suffix
`.model` denotes a PyTorch state dictionary here, not an MLX model.

The original nested encoder rewrite retained many parameter names but changed
input normalization, downsampling/pooling and BatchNorm settings. The clean
package preserves the upstream architecture. Strict loading detects missing
or incompatible tensors but does not detect every change in forward behavior.
Original Core ML exports are retained only as ignored local comparison data
in `models/legacy_coreml/`; they are not used by the supported pipeline.

## Core ML exports (original, experimental)

There are five LR-ASD packages, each with a graph and `weight.bin`:

- Visual input `v`: `[1,100,112,112]` grayscale values.
- Audio input `a`: `[1,400,13]` MFCCs.
- Detector d2/d4/d6: paired `[B,50/100/150,128]` embeddings.
- All three detector weight blobs have identical SHA-256; graph shapes differ.

The source export script ties these to the nested architecture; the exact
provenance of the historical binaries is not proven. Do not assume parity.
The optional `scripts/export_coreml.py` now wraps the corrected frontend
operations and strict loader. It is a development tool, not a runtime backend:

```sh
python -m pip install -e '.[export]'
python scripts/export_coreml.py --weights models/pretrain_AVA.model --out-dir models/coreml
```

It performs synthetic output comparisons and rejects max absolute error above
0.1. Export itself was not run in this cleanup. The test environment's Torch
version is newer than the version tested by coremltools, so a compatible export
environment may be required. This does not affect normal Python reframing.

## Detector and depth experiments

- `yolov8n.pt`: generic Ultralytics detector, not the two-class face checkpoint.
- `yolov8x_person_face.onnx`: dynamic batch/spatial graph.
- `*_fp16_static.onnx`: fixed batch 1, 640×640.
- `*_fp16_b8_static.onnx` and `*_fp32_b8_static.onnx`: fixed batch 8, 640×640.
  The inspected FP16 export inputs are still float32; the filename describes
  internal precision, not necessarily the input contract.
- `*_uint8.onnx`: quantization experiment with dynamic input dimensions.
  Parsed successfully; numerical parity and runtime support are unverified.
- `sym_shape_infer_temp.onnx`: intermediate graph with an external `.data`
  sidecar; its small graph file is not a standalone tiny detector.
- `sfd_face.pth`: original S3FD face detector, unused in the new runtime.
- Two `video_depth_anything_vits.pth` copies in the old depth folders are
  byte-identical (351 tensors). They are unrelated to current speaker framing.
- `DepthAnythingV2SmallF16.mlpackage`: older native depth experiment; graph
  and weight blob are inventoried and excluded from the reframer.

Model graphs and tensor dictionaries were inspected without executing their
inference. YOLO `.pt` object archives were inspected statically using pickle
opcodes; no arbitrary object unpickling was used for the audit.

## Provenance and distribution

[LR-ASD upstream](https://github.com/Junhua-Liao/LR-ASD) supplies the ASD model
and MIT notice. The AVA URL is pinned to the original checkout revision and
was freshly downloaded with a matching checksum. The
[YOLO model card](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)
identifies a face/person detector and lists AGPL-3.0. Its download is hash-pinned
but the URL uses `main`; changed upstream content causes a checksum failure.
The optional YOLO download was not freshly verified here. Do not distribute
all experimental weights under the LR-ASD MIT notice.

## Every discovered artifact

Sizes use decimal MB. Paths are relative to the **original workspace**, not
the clean repository. SHA prefixes are for browsing; use full hashes in JSON.

| Original path | MB | SHA-256 prefix |
| --- | ---: | --- |
| `AutoClip/old/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth` | 116.441 | `13379300b739` |
| `AutoClip/old/depth-anything-3/checkpoints/video_depth_anything_vits.pth` | 116.441 | `13379300b739` |
| `AutoClip/old/swift_depth/DepthVideo/models/DepthAnythingV2SmallF16.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.399 | `44ac97a3efcf` |
| `AutoClip/old/swift_depth/DepthVideo/models/DepthAnythingV2SmallF16.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 49.419 | `fa60d9b6a155` |
| `AutoClip/optimized_asd/optimized_python/facelivtv2-s.pt` | 18.968 | `ec659a5f8447` |
| `AutoClip/optimized_asd/optimized_python/sym_shape_infer_temp.onnx` | 0.147 | `642315b20e0a` |
| `AutoClip/optimized_asd/optimized_python/yolov8n.pt` | 6.550 | `f59b3d833e2f` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face.onnx` | 273.441 | `1901c6bd4a09` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face.pt` | 136.716 | `2620f45609a6` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face_fp16_b8_static.onnx` | 136.434 | `be0ab4c747f3` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face_fp16_static.onnx` | 136.434 | `86b3a66de024` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face_fp32_b8_static.onnx` | 272.744 | `ac5202418208` |
| `AutoClip/optimized_asd/optimized_python/yolov8x_person_face_uint8.onnx` | 69.303 | `022f9987bdb1` |
| `AutoClip/optimized_asd/weight/finetuning_TalkSet.model` | 3.426 | `6b4ef53694e8` |
| `AutoClip/optimized_asd/weight/pretrain_AVA.model` | 3.426 | `85e6c77fc981` |
| `AutoClip/optimized_asd/weight/audio_encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.020 | `898ae8a3b8d3` |
| `AutoClip/optimized_asd/weight/audio_encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 0.463 | `63405ae46010` |
| `AutoClip/optimized_asd/weight/detector_head_d2.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.039 | `886ad33de059` |
| `AutoClip/optimized_asd/weight/detector_head_d2.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 0.414 | `77a82a6dbebe` |
| `AutoClip/optimized_asd/weight/detector_head_d4.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.039 | `e9cab750ce00` |
| `AutoClip/optimized_asd/weight/detector_head_d4.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 0.414 | `77a82a6dbebe` |
| `AutoClip/optimized_asd/weight/detector_head_d6.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.039 | `c301643d10e9` |
| `AutoClip/optimized_asd/weight/detector_head_d6.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 0.414 | `77a82a6dbebe` |
| `AutoClip/optimized_asd/weight/visual_encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 0.020 | `7ac0bb518625` |
| `AutoClip/optimized_asd/weight/visual_encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 0.798 | `879f7eaa8b68` |
| `model/faceDetector/s3fd/sfd_face.pth` | 89.844 | `d54a87c2b754` |
| `weight/finetuning_TalkSet.model` | 3.426 | `6b4ef53694e8` |
| `weight/pretrain_AVA.model` | 3.426 | `85e6c77fc981` |
| `AutoClip/optimized_asd/optimized_python/8c1ac272-588d-11f1-a0f3-c2179141218d.data` | 272.484 | `93325a713fa7` |
