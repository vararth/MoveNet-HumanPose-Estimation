import cv2
import imageio
from pathlib import Path
from typing import Generator, List
import numpy as np


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_gif(path: str) -> List[np.ndarray]:
    # Robust GIF read with imageio (returns list of RGB frames)
    frames_rgb = imageio.mimread(path)
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
    return frames_bgr


def write_gif(path: str, frames_bgr: List[np.ndarray], fps: int = 15) -> None:
    # imageio GIF writer expects 'duration' (seconds per frame), not 'fps'
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    duration = 1.0 / max(fps, 1)
    with imageio.get_writer(path, mode="I", duration=duration, loop=0, format="GIF") as w:
        for fr in frames_rgb:
            w.append_data(fr)


def iter_video(path: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def get_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    # Use char-args form; avoids some stub complaints. Runtime equivalent.
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {path}")
    return writer


def probe_fps(path: str, default_fps: int = 15) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return float(default_fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1e-3 else float(default_fps)
    finally:
        cap.release()
