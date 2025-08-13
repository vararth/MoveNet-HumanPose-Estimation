import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Callable, Dict

# MoveNet MultiPose Lightning
_MPV1 = "https://tfhub.dev/google/movenet/multipose/lightning/1"


class MoveNetMulti:
    """
    Wrapper for MoveNet MultiPose Lightning from TF-Hub.

    Input to infer(): square RGB uint8 array [H,W,3] where H=W=input_size.
    Output: keypoints array (6, 17, 3) with (y, x, score) normalized to [0,1].
    """

    def __init__(self, model_url: str = _MPV1, input_size: int = 256):
        self.module = hub.load(model_url)

        # SignatureMap behaves like a dict; access 'serving_default'
        sigs = getattr(self.module, "signatures", None)
        if sigs is None:
            raise RuntimeError(f"Loaded TF-Hub object has no 'signatures' (type={type(self.module)!r}).")

        try:
            # ConcreteFunction: Callable[[tf.Tensor], Dict[str, tf.Tensor]]
            self.fn: Callable[[tf.Tensor], Dict[str, tf.Tensor]] = sigs["serving_default"]  # type: ignore[assignment]
        except Exception:
            try:
                available = list(sigs.keys())  # type: ignore[attr-defined]
            except Exception:
                available = []
            raise RuntimeError(
                f"'serving_default' not found in model signatures. Available: {available}. "
                f"Model URL used: {model_url}"
            )

        self.input_size = input_size

    def infer(self, rgb_square_uint8: np.ndarray) -> np.ndarray:
        # Ensure [1, H, W, 3] int32 as expected by MoveNet
        t = tf.convert_to_tensor(rgb_square_uint8, dtype=tf.uint8)
        t = tf.image.resize_with_pad(t, self.input_size, self.input_size)
        t = tf.cast(t, tf.int32)
        t = tf.expand_dims(t, 0)

        outputs_any = self.fn(t)              # dict-like with "output_0"
        outputs = dict(outputs_any)           # make subscriptable for type checkers
        if "output_0" not in outputs:
            raise KeyError("MoveNet outputs missing 'output_0'")

        out = outputs["output_0"].numpy()     # shape (1, 6, 56)
        # first 51 values = 17 keypoints * (y, x, score)
        kps = out[:, :, :51].reshape((1, 6, 17, 3))
        return kps[0]                          # (6, 17, 3)
