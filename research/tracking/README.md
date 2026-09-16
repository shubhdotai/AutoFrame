# ByteTrack + face identity research

`pipeline_faces.py` is the separate YOLO/ByteTrack experiment found in the
workspace. It detects faces within a scene, resets ByteTrack at scene changes,
and reconciles face identities across scenes with FaceLiVTv2 embeddings.
It is **not** connected to ASD scoring or the supported reframing CLI.

Required local files in `../../models/`:
`yolov8x_person_face.pt` and `facelivtv2-s.pt`. The latter's provenance is not
established, so no downloader or redistribution is provided. The local copy
is Git-ignored. `test_facelvit2.py` contains the model implementation needed
by `face_embedder.py`; despite its historical name it is not a CI test.

Use a separate environment with torch, torchvision, ultralytics, timm,
Pillow, OpenCV, NumPy and scenedetect. From this directory:

```sh
python pipeline_faces.py /path/to/video.mp4 --out /path/to/tracking.mp4
```

This code buffers scene frames in memory. Long uncut scenes may consume large
amounts of RAM. Confidence-weighted sampling is random and the gallery is
capped; identities are heuristic. This experiment has not been runtime-tested
as part of packaging, and its dependencies/weight rights need review before
promoting it into the supported package.
