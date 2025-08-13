# Human Pose Estimation with MoveNet (MultiPose Lightning)

Runs Google MoveNet MultiPose Lightning (TF-Hub) on **images**, **videos**, and **GIFs** with:
- A clean CLI (`pose-run`)
- COCO export (`pose-export-coco` or `python -m movenet_runner.export_coco`)
- Optional overlay output (annotated media)

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run Inference
```bash
pose-run --input data\input.gif --output data\output.gif --kind gif --size 256 --threshold 0.11 --fps 15
```

## COCO Export
```bash
# JSON only
python -m movenet_runner.export_coco --input data\input.gif --kind gif --output data\export.json

# JSON + annotated overlay
python -m movenet_runner.export_coco --input data\input.gif --kind gif --output data\export.json --overlay_out data\overlay.gif --fps 15
```

## Notes:
#### TensorFlow GPU is supported via Linux/WSL2; on native Windows this runs CPU-only.
#### GIF IO uses imageio; videos/images via OpenCV.
#### MultiPose Lightning: up to 6 persons × 17 keypoints.

## License
MIT
