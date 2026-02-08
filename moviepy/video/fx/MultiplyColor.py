from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.video.tools import scratch


@dataclass
class MultiplyColor(Effect):
    """
    Multiplies the clip's colors by the given factor, can be used
    to decrease or increase the clip's brightness (is that the
    right word ?)
    """

    factor: float

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        factor = np.float32(self.factor)

        def _mul(frame: np.ndarray) -> np.ndarray:
            tmp = scratch.get_float32("multiplycolor_tmp", frame.shape)
            np.multiply(frame, factor, out=tmp, casting="unsafe")
            np.minimum(tmp, 255.0, out=tmp)
            return tmp.astype("uint8")

        return clip.image_transform(_mul)
