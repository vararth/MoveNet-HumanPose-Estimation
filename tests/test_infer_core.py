import numpy as np
from src.movenet_runner.infer_core import pad_resize_to_square

def test_pad_resize_to_square():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    sq, scale, pad_x, pad_y = pad_resize_to_square(img, 256)
    assert sq.shape == (256, 256, 3)
    assert scale > 0
    assert pad_x >= 0 and pad_y >= 0
