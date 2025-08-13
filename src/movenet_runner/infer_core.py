import cv2
import numpy as np
from typing import Tuple
from .config import WIDTH, HEIGHT, SCORE_THRESH, KEYPOINT_COLOR, EDGE_COLORS
from .draw import draw_person_on_image

def pad_resize_to_square(bgr: np.ndarray, size: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Letterbox-pad an image to a size x size square.
    Returns:
      square_bgr: (size,size,3)
      scale: scale used to fit original into square
      pad_x, pad_y: left/top padding in square coords
    """
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    square = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    square[y0:y0+new_h, x0:x0+new_w] = resized
    return square, scale, pad_x, pad_y

def process_frame(model, frame_bgr: np.ndarray, size: int = WIDTH, thresh: float = SCORE_THRESH) -> np.ndarray:
    """
    Full pipeline on a single frame:
      - Letterbox to square
      - Run MoveNet
      - Map keypoints back to original frame and draw
    Returns the annotated frame with original resolution.
    """
    original = frame_bgr.copy()
    square_bgr, scale, pad_x, pad_y = pad_resize_to_square(frame_bgr, size)

    # Convert BGR→RGB for the model
    square_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)

    kps = model.infer(square_rgb)   # (6, 17, 3); normalized

    # Draw each person
    for instance in kps:
        # If an instance has near-zero scores everywhere, skip
        if (instance[:, 2] > max(0.05, thresh * 0.5)).sum() == 0:
            continue
        draw_person_on_image(
            original, instance, size, scale, pad_x, pad_y,
            KEYPOINT_COLOR, EDGE_COLORS, thresh
        )
    return original

def process_frame_profiled(model, frame_bgr: np.ndarray, size: int = WIDTH, thresh: float = SCORE_THRESH):
    """
    Like process_frame, but also returns how many persons were drawn.
    """
    original = frame_bgr.copy()
    square_bgr, scale, pad_x, pad_y = pad_resize_to_square(frame_bgr, size)
    square_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)

    kps = model.infer(square_rgb)  # (6, 17, 3)
    count = 0
    for instance in kps:
        if (instance[:, 2] > max(0.05, thresh * 0.5)).sum() == 0:
            continue
        count += 1
        draw_person_on_image(
            original, instance, size, scale, pad_x, pad_y,
            KEYPOINT_COLOR, EDGE_COLORS, thresh
        )
    return original, count
