# Third-party attribution

The original LR-ASD architecture and derived pipeline code come from
[Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD). Its MIT license
notice (Copyright 2025 Liao Junhua) is preserved verbatim in `LICENSE`.
The repository documents both AVA-pretrained and TalkSet-finetuned checkpoints.

Please cite the work when using its code or models:

- Junhua Liao et al., **A Light Weight Model for Active Speaker Detection**,
  CVPR 2023, pp. 22932–22941.
- Junhua Liao et al., **LR-ASD: Lightweight and Robust Network for Active
  Speaker Detection**, International Journal of Computer Vision, 2025.

Upstream acknowledges [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD).
AutoClip is a derived application, not an official upstream release.

Optional detector: [iitolstykh/YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector),
loaded with Ultralytics. The model card labels it AGPL-3.0; its terms and
Ultralytics' terms are separate from the MIT LR-ASD code. FaceLiVTv2 local
weight provenance is unresolved. No optional model binaries are included in
the source distribution. Native ASR/LLM models also retain their own terms.

The existing local license is retained; it is not a claim that all optional
third-party weights have been relicensed or cleared for redistribution.
