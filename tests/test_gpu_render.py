import numpy as np
import pytest


def test_gpu_render_composite_smoke(monkeypatch):
    cp = pytest.importorskip("cupy")

    monkeypatch.setenv("MOVIEPY_GPU_RENDER", "1")
    monkeypatch.setenv("MOVIEPY_GPU_AGGRESSIVE", "1")
    monkeypatch.setenv("MOVIEPY_GPU_RENDER_STRICT", "1")

    from moviepy.video.VideoClip import ColorClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.tools import gpu_render

    if not gpu_render.is_available():
        pytest.skip("CuPy installed but CUDA runtime not usable")

    base = ColorClip((64, 48), color=(10, 20, 30)).with_duration(1).with_fps(5)
    top = ColorClip((20, 10), color=(200, 0, 0)).with_duration(1).with_fps(5)
    top = top.with_position((5, 7))

    comp = CompositeVideoClip([base, top], size=(64, 48), bg_color=(0, 0, 0))

    frame_gpu = gpu_render.get_frame_for_export_uint8_gpu(comp, 0)
    # In GPU mode and with a CUDA device, we expect a CuPy array.
    assert hasattr(frame_gpu, "__cuda_array_interface__")

    frame = cp.asnumpy(frame_gpu)
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.shape == (48, 64, 3)
