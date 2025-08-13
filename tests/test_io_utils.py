import numpy as np
from pathlib import Path
from src.movenet_runner.io_utils import write_gif, read_gif

def test_gif_roundtrip(tmp_path: Path):
    frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(5)]
    out = tmp_path / "test.gif"
    write_gif(str(out), frames, fps=10)
    frames2 = read_gif(str(out))
    assert len(frames2) == 5
    assert frames2[0].shape == (32, 48, 3)
