"""Experimental GPU render path.

Goal: keep intermediate compositing math on the GPU (CuPy) during export,
and only download once per output frame for feeding FFmpeg.

This does NOT change MoviePy's public API by default.
It is used opportunistically by ffmpeg_write_video when enabled.
"""

from __future__ import annotations

import os
import site
from typing import Tuple

import numpy as np

from moviepy.tools import compute_position


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in {"0", "false", "no", "off", ""}


def _strict() -> bool:
    return _env_flag("MOVIEPY_GPU_RENDER_STRICT", "0")


_CUPY_USABLE: bool | None = None


def _windows_cuda_dll_setup() -> None:
    """Best-effort CUDA DLL setup for Windows.

    CuPy may import successfully but fail later when it needs NVRTC DLLs.
    Adding CUDA Toolkit / pip-runtime DLL directories helps DLL resolution.
    """
    if os.name != "nt":
        return

    # Prefer explicit CUDA_PATH.
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")

    # Some installations set versioned variables (e.g. CUDA_PATH_V13_0).
    if not cuda_root:
        versioned = [(k, v) for k, v in os.environ.items() if k.startswith("CUDA_PATH_V") and v]
        if versioned:
            # Pick the highest lexicographic key (good enough for V<major>_<minor>).
            cuda_root = sorted(versioned, key=lambda kv: kv[0])[-1][1]
            os.environ.setdefault("CUDA_PATH", cuda_root)

    candidates: list[str] = []
    if cuda_root:
        candidates.extend(
            [
                os.path.join(cuda_root, "bin"),
                os.path.join(cuda_root, "bin", "x64"),
                os.path.join(cuda_root, "bin", "x86_64"),
            ]
        )

    # Also consider pip-installed NVIDIA runtime layout (if present).
    for sp in site.getsitepackages():
        candidates.extend(
            [
                os.path.join(sp, "nvidia", "cu13", "bin"),
                os.path.join(sp, "nvidia", "cu13", "bin", "x86_64"),
                os.path.join(sp, "nvidia", "cu12", "bin"),
                os.path.join(sp, "nvidia", "cu12", "bin", "x86_64"),
            ]
        )

    for d in candidates:
        try:
            if d and os.path.isdir(d):
                os.add_dll_directory(d)
        except Exception:
            pass


def _cp():
    _windows_cuda_dll_setup()
    import cupy as cp

    return cp


def _cupy_is_usable(cp) -> bool:
    """Return True if CuPy can actually execute kernels.

    Some installations can import CuPy and even see a device, but fail at runtime
    due to missing CUDA/NVRTC DLLs (common on Windows if PATH isn't set up).
    """
    try:
        # Device query
        try:
            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                return False
        except Exception:
            # Be permissive; the real check is executing a kernel.
            pass

        # Minimal kernel compile/launch check.
        a = cp.zeros((1,), dtype=cp.uint8)
        b = a.astype(cp.float32)
        b += 1.0
        cp.cuda.runtime.deviceSynchronize()
        return True
    except Exception:
        return False


def is_available() -> bool:
    global _CUPY_USABLE
    if _CUPY_USABLE is not None:
        return bool(_CUPY_USABLE)
    if _env_flag("MOVIEPY_DISABLE_GPU", "0"):
        _CUPY_USABLE = False
        return False
    try:
        cp = _cp()
        _CUPY_USABLE = bool(_cupy_is_usable(cp))
        return bool(_CUPY_USABLE)
    except Exception:
        _CUPY_USABLE = False
        return False


def is_enabled() -> bool:
    """Whether the experimental GPU export path should be attempted.

    Default: enabled (auto) when available.
    Opt-out: set MOVIEPY_GPU_RENDER=0 or MOVIEPY_DISABLE_GPU=1.
    """
    if _env_flag("MOVIEPY_DISABLE_GPU", "0"):
        return False

    # Aggressive implies enabled.
    if _env_flag("MOVIEPY_GPU_AGGRESSIVE", "0"):
        return True

    # If user explicitly sets the flag, honor it.
    if "MOVIEPY_GPU_RENDER" in os.environ:
        return _env_flag("MOVIEPY_GPU_RENDER", "0")

    # Auto by default.
    return True


def _is_gpu_array(x) -> bool:
    # CuPy (and many CUDA array libs) expose this.
    return hasattr(x, "__cuda_array_interface__")


def _to_cp_u8_rgb(frame):
    cp = _cp()
    f = frame if _is_gpu_array(frame) else cp.asarray(frame)
    if f.ndim == 2:
        f = cp.stack([f, f, f], axis=2)
    if f.ndim == 3 and f.shape[2] >= 3:
        f = f[:, :, :3]
    if f.dtype != cp.uint8:
        f = f.astype(cp.uint8)
    return f


def _to_cp_f32_mask(mask):
    cp = _cp()
    m = mask if _is_gpu_array(mask) else cp.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != cp.float32:
        m = m.astype(cp.float32)
    return m


def _as_u8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.dstack([frame, frame, frame])
    if frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame


def _as_f32_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    return mask


def _coords_for_layer(
    background_size: Tuple[int, int],
    clip_size: Tuple[int, int],
    x_start: int,
    y_start: int,
):
    background_width, background_height = background_size
    clip_width, clip_height = clip_size

    y1_bg = max(y_start, 0)
    y2_bg = min(y_start + clip_height, background_height)
    x1_bg = max(x_start, 0)
    x2_bg = min(x_start + clip_width, background_width)

    y1_clip = max(-y_start, 0)
    y2_clip = y1_clip + (y2_bg - y1_bg)
    x1_clip = max(-x_start, 0)
    x2_clip = x1_clip + (x2_bg - x1_bg)

    return y1_bg, y2_bg, x1_bg, x2_bg, y1_clip, y2_clip, x1_clip, x2_clip


def _composite_rgb_gpu(clip, t):
    """Return a CuPy uint8 RGB frame for a (possibly composite) clip."""
    cp = _cp()

    # CompositeVideoClip path
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    if isinstance(clip, CompositeVideoClip):
        playing = clip.playing_clips(t)

        bg_t = t - clip.bg.start
        bg_u8 = _to_cp_u8_rgb(clip.bg.get_frame(bg_t))
        bg_f = bg_u8.astype(cp.float32)

        bg_mask_f = None
        if clip.bg.mask:
            bgm_t = t - clip.bg.mask.start
            bg_mask_f = _to_cp_f32_mask(clip.bg.mask.get_frame(bgm_t))

        bg_h, bg_w = int(bg_u8.shape[0]), int(bg_u8.shape[1])

        for layer in playing:
            ct = t - layer.start
            fr_u8 = _to_cp_u8_rgb(layer.get_frame(ct))
            fr_f = fr_u8.astype(cp.float32)

            layer_mask_f = None
            if layer.mask is not None:
                layer_mask_f = _to_cp_f32_mask(layer.mask.get_frame(ct))

            pos = layer.pos(ct)
            x_start, y_start = compute_position(
                (int(fr_u8.shape[1]), int(fr_u8.shape[0])),
                (bg_w, bg_h),
                pos,
                layer.relative_pos,
            )

            (
                y1_bg,
                y2_bg,
                x1_bg,
                x2_bg,
                y1_clip,
                y2_clip,
                x1_clip,
                x2_clip,
            ) = _coords_for_layer((bg_w, bg_h), (int(fr_u8.shape[1]), int(fr_u8.shape[0])), x_start, y_start)

            if (y2_bg <= y1_bg) or (x2_bg <= x1_bg):
                continue

            bg_region = bg_f[y1_bg:y2_bg, x1_bg:x2_bg]
            fr_region = fr_f[y1_clip:y2_clip, x1_clip:x2_clip]

            if layer_mask_f is None:
                bg_region[...] = fr_region
                if bg_mask_f is not None:
                    bg_mask_f[y1_bg:y2_bg, x1_bg:x2_bg] = 1.0
                continue

            a = layer_mask_f[y1_clip:y2_clip, x1_clip:x2_clip]

            if bg_mask_f is None:
                # bg = bg + a*(fr - bg)
                cp.subtract(fr_region, bg_region, out=fr_region)
                fr_region *= a[:, :, None]
                bg_region += fr_region
            else:
                # final_a = a + bg_a*(1-a)
                bg_a = bg_mask_f[y1_bg:y2_bg, x1_bg:x2_bg]
                one_minus_a = 1.0 - a
                final_a = a + bg_a * one_minus_a
                safe_a = cp.where(final_a == 0, 1.0, final_a)

                # numerator = fr*a + bg*bg_a*(1-a)
                # reuse fr_region as temp: fr*a
                fr_region *= a[:, :, None]
                bg_region *= (bg_a * one_minus_a)[:, :, None]
                bg_region += fr_region
                bg_region /= safe_a[:, :, None]

                bg_a[...] = final_a

        cp.rint(bg_f, out=bg_f)
        cp.clip(bg_f, 0, 255, out=bg_f)
        return bg_f.astype(cp.uint8)

    # Non-composite: upload current frame.
    return _to_cp_u8_rgb(clip.get_frame(t))


def _composite_mask_gpu(mask_clip, t):
    """Return a CuPy float32 mask frame."""
    cp = _cp()

    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    if isinstance(mask_clip, CompositeVideoClip) and mask_clip.is_mask:
        # Background mask is zeros
        h, w = mask_clip.size[1], mask_clip.size[0]
        b = cp.zeros((h, w), dtype=cp.float32)

        for layer in mask_clip.playing_clips(t):
            ct = t - layer.start
            m = _to_cp_f32_mask(layer.get_frame(ct))

            pos = layer.pos(ct)
            x_start, y_start = compute_position(
                (int(m.shape[1]), int(m.shape[0])),
                (w, h),
                pos,
                layer.relative_pos,
            )

            (
                y1_bg,
                y2_bg,
                x1_bg,
                x2_bg,
                y1_clip,
                y2_clip,
                x1_clip,
                x2_clip,
            ) = _coords_for_layer((w, h), (int(m.shape[1]), int(m.shape[0])), x_start, y_start)

            if (y2_bg <= y1_bg) or (x2_bg <= x1_bg):
                continue

            b_region = b[y1_bg:y2_bg, x1_bg:x2_bg]
            m_region = m[y1_clip:y2_clip, x1_clip:x2_clip]

            # b = m + b*(1-m)
            b_region *= (1.0 - m_region)
            b_region += m_region

        return b

    # Non-composite mask: upload.
    return _to_cp_f32_mask(mask_clip.get_frame(t))


def get_frame_for_export_uint8(clip, t: float):
    """Return a NumPy uint8 frame ready for ffmpeg_writer.

    If clip has a mask, returns RGBA uint8; otherwise RGB uint8.

    Falls back to CPU path on any error.
    """
    if not (is_enabled() and is_available()):
        frame = clip.get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame

    cp = _cp()

    try:
        rgb_u8 = _composite_rgb_gpu(clip, t)
        if clip.mask is None:
            return cp.asnumpy(rgb_u8)

        # Match CPU export path semantics exactly:
        # mask_u8 = (255 * mask_frame).astype(uint8) with NumPy/CuPy overflow
        # behavior for uint8 masks, and stack via dstack semantics.
        mask_frame = clip.mask.get_frame(t)
        m = mask_frame if _is_gpu_array(mask_frame) else cp.asarray(mask_frame)
        m_u8 = m * 255
        if m_u8.dtype != cp.uint8:
            m_u8 = m_u8.astype(cp.uint8)
        if m_u8.ndim == 2:
            m_u8 = m_u8[:, :, None]
        rgba = cp.concatenate([rgb_u8, m_u8], axis=2)
        return cp.asnumpy(rgba)
    except Exception:
        if _strict():
            raise
        # Best-effort: never fail export.
        frame = clip.get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if clip.mask is not None:
            mask = 255 * clip.mask.get_frame(t)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            frame = np.dstack([frame, mask])
        return frame


def get_frame_for_export_uint8_gpu(clip, t: float):
    """Return a CuPy uint8 frame (RGB or RGBA) for ffmpeg export.

    This is the "CuPy-native" experimental export pipeline: all compositing
    work happens on the GPU and the returned frame is a CuPy array. The only
    CPU conversion happens at the final FFmpeg write.

    On any error, returns a NumPy uint8 frame as a safe fallback.
    """
    if not (is_enabled() and is_available()):
        frame = clip.get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if clip.mask is not None:
            mask = 255 * clip.mask.get_frame(t)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            frame = np.dstack([frame, mask])
        return frame

    cp = _cp()
    try:
        rgb_u8 = _composite_rgb_gpu(clip, t)
        if clip.mask is None:
            return rgb_u8

        # Match CPU export path semantics exactly:
        # mask_u8 = (255 * mask_frame).astype(uint8) with CuPy overflow
        # behavior for uint8 masks, and stack via dstack semantics.
        mask_frame = clip.mask.get_frame(t)
        m = mask_frame if _is_gpu_array(mask_frame) else cp.asarray(mask_frame)
        m_u8 = m * 255
        if m_u8.dtype != cp.uint8:
            m_u8 = m_u8.astype(cp.uint8)
        if m_u8.ndim == 2:
            m_u8 = m_u8[:, :, None]
        rgba = cp.concatenate([rgb_u8, m_u8], axis=2)
        return rgba
    except Exception:
        if _strict():
            raise
        frame = clip.get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if clip.mask is not None:
            mask = 255 * clip.mask.get_frame(t)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            frame = np.dstack([frame, mask])
        return frame
