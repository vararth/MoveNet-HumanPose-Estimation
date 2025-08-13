import cv2
import numpy as np
from typing import Dict, Tuple

def draw_person_on_image(
    frame_bgr: np.ndarray,
    kp_norm: np.ndarray,                 # shape (17, 3) with (y, x, score) in [0..1]
    size: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    color_kp: Tuple[int, int, int],
    edges: Dict[Tuple[int, int], Tuple[int, int, int]],
    thresh: float
):
    """
    Map normalized (y,x) from the padded square back to original image coordinates and draw.
    Mapping:
      x_sq = x_norm * size
      y_sq = y_norm * size
      x_orig = (x_sq - pad_x) / scale
      y_orig = (y_sq - pad_y) / scale
    """
    # Compute original coords
    yx = kp_norm[:, :2] * size  # (17,2) in square pixels
    scores = kp_norm[:, 2]
    x_sq = yx[:, 1]; y_sq = yx[:, 0]
    x_orig = (x_sq - pad_x) / scale
    y_orig = (y_sq - pad_y) / scale

    # Draw keypoints
    for (x, y, s) in zip(x_orig, y_orig, scores):
        if s > thresh:
            cv2.circle(frame_bgr, (int(x), int(y)), 4, color_kp, thickness=-1, lineType=cv2.LINE_AA)

    # Draw edges
    for (p1, p2), col in edges.items():
        x1, y1, s1 = x_orig[p1], y_orig[p1], scores[p1]
        x2, y2, s2 = x_orig[p2], y_orig[p2], scores[p2]
        if s1 > thresh and s2 > thresh:
            cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
