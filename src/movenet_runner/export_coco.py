import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from .model import MoveNetMulti
from .infer_core import pad_resize_to_square
from .io_utils import read_image, read_gif, iter_video, write_gif, get_video_writer, probe_fps
from .draw import draw_person_on_image
from .config import EDGE_COLORS, KEYPOINT_COLOR

# COCO keypoint order (COCO-17)
KEYPOINT_NAMES: List[str] = [
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# COCO "skeleton" pairs are 1-based indices into the keypoints array.
SKELETON: List[List[int]] = [[i + 1, j + 1] for (i, j) in EDGE_COLORS.keys()]


def _kp_to_coco(
    instance: np.ndarray, size: int, scale: float, pad_x: float, pad_y: float, thresh: float
) -> Tuple[List[float], int, float, Tuple[float, float, float, float]]:
    """
    Convert one person's normalized (17,3) (y, x, score) from size x size square back to original image coords.
    Returns:
      keypoints: [x1,y1,v1, x2,y2,v2, ...] (len=51)
      num_keypoints: count with v>0
      mean_score: float
      bbox: (x,y,w,h) from visible keypoints (v>0)
    v = 2 if score>=thresh else 0
    """
    yx = instance[:, :2] * size
    scores = instance[:, 2]

    x_sq = yx[:, 1]
    y_sq = yx[:, 0]
    x = (x_sq - pad_x) / scale
    y = (y_sq - pad_y) / scale

    keypoints: List[float] = []
    vx: List[float] = []
    vy: List[float] = []
    vcount = 0

    for xi, yi, si in zip(x, y, scores):
        v = 2 if si >= thresh else 0
        if v > 0:
            vcount += 1
            vx.append(float(xi))
            vy.append(float(yi))
        keypoints.extend([float(xi), float(yi), int(v)])

    if vcount > 0:
        min_x, max_x = min(vx), max(vx)
        min_y, max_y = min(vy), max(vy)
        bbox = (min_x, min_y, max(1.0, max_x - min_x), max(1.0, max_y - min_y))
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    mean_score = float(np.mean(scores)) if scores.size else 0.0
    return keypoints, vcount, mean_score, bbox


def _export_frame(
    model: MoveNetMulti,
    frame_bgr: np.ndarray,
    image_id: int,
    file_name: str,
    thresh: float,
    size: int,
    draw_overlay: bool = False
) -> Tuple[List[Dict], List[Dict], Optional[np.ndarray]]:
    """
    Run model on a single frame and produce COCO 'images' and 'annotations' entries.
    Optionally also return an annotated frame (BGR) if draw_overlay=True.
    """
    h, w = frame_bgr.shape[:2]
    square_bgr, scale, pad_x, pad_y = pad_resize_to_square(frame_bgr, size)
    square_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    kps_all = model.infer(square_rgb)  # (6, 17, 3)

    images = [{
        "id": image_id,
        "file_name": file_name,
        "width": int(w),
        "height": int(h),
    }]

    annotations: List[Dict] = []
    ann_id = image_id * 100000

    overlay = frame_bgr.copy() if draw_overlay else None

    for inst in kps_all:
        if (inst[:, 2] >= max(0.05, thresh * 0.5)).sum() == 0:
            continue

        # COCO entry
        keypoints, vcount, score, bbox = _kp_to_coco(inst, size, scale, pad_x, pad_y, thresh)
        x, y, bw, bh = bbox
        # clip bbox
        x = float(max(0.0, min(x, w - 1)))
        y = float(max(0.0, min(y, h - 1)))
        bw = float(max(0.0, min(bw, w - x)))
        bh = float(max(0.0, min(bh, h - y)))
        area = float(max(0.0, bw * bh))

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": int(vcount),
            "bbox": [x, y, bw, bh],
            "area": area,
            "score": float(score)
        })
        ann_id += 1

        # Overlay drawing if requested
        if overlay is not None:
            draw_person_on_image(
                overlay,
                inst,            # normalized (17,3)
                size,
                scale,
                pad_x,
                pad_y,
                KEYPOINT_COLOR,
                EDGE_COLORS,
                thresh
            )

    return images, annotations, overlay


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Export MoveNet detections to COCO-style JSON (optional overlay output).")
    ap.add_argument("--input", required=True, help="Path to image/video/gif")
    ap.add_argument("--kind", choices=["image", "video", "gif"], required=True)
    ap.add_argument("--output", required=True, help="Output COCO JSON path")
    ap.add_argument("--size", type=int, default=256, help="Model square input (e.g., 256/192/320)")
    ap.add_argument("--threshold", type=float, default=0.11, help="Keypoint visibility threshold")
    ap.add_argument("--overlay_out", type=str, default="", help="Optional path to save annotated output (PNG/MP4/GIF)")
    ap.add_argument("--fps", type=int, default=15, help="FPS for GIF/video when saving overlay")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model = MoveNetMulti(input_size=args.size)

    images_all: List[Dict] = []
    anns_all: List[Dict] = []

    draw_overlay = bool(args.overlay_out)
    overlay_path = Path(args.overlay_out) if draw_overlay else None

    if args.kind == "image":
        img = read_image(str(in_path))
        imgs, anns, overlay = _export_frame(model, img, image_id=1, file_name=in_path.name,
                                            thresh=args.threshold, size=args.size, draw_overlay=draw_overlay)
        images_all += imgs
        anns_all += anns
        if overlay is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(overlay_path), overlay)
            if not ok:
                raise RuntimeError(f"Could not write overlay image: {overlay_path}")

    elif args.kind == "gif":
        frames = read_gif(str(in_path))
        overlays: List[np.ndarray] = []
        for i, f in enumerate(frames, start=1):
            imgs, anns, overlay = _export_frame(model, f, image_id=i, file_name=f"{in_path.name}#{i}",
                                                thresh=args.threshold, size=args.size, draw_overlay=draw_overlay)
            images_all += imgs
            anns_all += anns
            if overlay is not None:
                overlays.append(overlay)
        if draw_overlay and overlays:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            write_gif(str(overlay_path), overlays, fps=args.fps)

    else:  # video
        frame_idx = 1
        overlay_writer = None
        if draw_overlay:
            # Probe FPS from input; fall back to --fps if probe fails
            in_fps = probe_fps(str(in_path), default_fps=args.fps)
        for f in iter_video(str(in_path)):
            imgs, anns, overlay = _export_frame(model, f, image_id=frame_idx, file_name=f"{in_path.name}#{frame_idx}",
                                                thresh=args.threshold, size=args.size, draw_overlay=draw_overlay)
            images_all += imgs
            anns_all += anns

            if overlay is not None:
                if overlay_writer is None:
                    h, w = overlay.shape[:2]
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    overlay_writer = get_video_writer(str(overlay_path), w, h, in_fps)
                overlay_writer.write(overlay)
            frame_idx += 1
        if overlay_writer is not None:
            overlay_writer.release()

    coco_dict = {
        "info": {"description": "MoveNet MultiPose export", "version": "1.0"},
        "licenses": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON
        }],
        "images": images_all,
        "annotations": anns_all
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"Wrote {out_json} with {len(images_all)} images and {len(anns_all)} annotations.")


if __name__ == "__main__":
    main()
