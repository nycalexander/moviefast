"""Optional GPU-native video encoding via NVIDIA NVENC (PyNvCodec).

This module is internal and best-effort:
- It is only attempted when the experimental GPU render path is explicitly enabled.
- It never changes MoviePy's public API.
- It always falls back to the existing FFmpeg stdin writer on any failure.

The goal is to avoid downloading per-frame RGB data back to the CPU for encoding.
We still rely on FFmpeg for container muxing and (optionally) audio.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from moviepy.config import FFMPEG_BINARY
from moviepy.tools import ffmpeg_escape_filename, subprocess_call


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in {"0", "false", "no", "off", ""}


def _disabled_value(val: str) -> bool:
    v = (val or "").strip().lower()
    return v in {"0", "false", "no", "off", "none", "disable", "disabled"}


def gpu_encode_backend() -> str:
    return (os.getenv("MOVIEPY_GPU_ENCODE_BACKEND") or "auto").strip().lower()


def is_enabled() -> bool:
    """Whether GPU-native encoding should be attempted.

    To avoid surprising behavior, this only turns on when GPU render is
    *explicitly* enabled (or aggressive mode is set).
    """
    if _env_flag("MOVIEPY_DISABLE_GPU", "0"):
        return False
    if _env_flag("MOVIEPY_DISABLE_GPU_ENCODE", "0"):
        return False

    backend = gpu_encode_backend()
    if _disabled_value(backend):
        return False

    # Aggressive implies opt-in.
    if _env_flag("MOVIEPY_GPU_AGGRESSIVE", "0"):
        return True

    # Only opt-in when user explicitly enabled GPU render.
    if "MOVIEPY_GPU_RENDER" in os.environ and _env_flag("MOVIEPY_GPU_RENDER", "0"):
        return True

    return False


def _strict() -> bool:
    return _env_flag("MOVIEPY_GPU_ENCODE_STRICT", "0")


def _parse_gpu_id() -> int:
    try:
        return int(os.getenv("MOVIEPY_GPU_DEVICE", "0"))
    except Exception:
        return 0


def _map_codec(codec: str) -> Optional[str]:
    """Map MoviePy/FFmpeg-ish codec strings to PyNvEncoder codec values."""
    c = (codec or "").strip().lower()
    if not c:
        return None
    if "hevc" in c or "h265" in c or "libx265" in c or c == "h265_nvenc":
        return "hevc"
    if "h264" in c or "x264" in c or "libx264" in c or c == "h264_nvenc":
        return "h264"
    return None


def _map_nvenc_preset(preset: str) -> str:
    """Best-effort mapping from FFmpeg preset names to NVENC preset IDs."""
    p = (preset or "").strip().lower()
    return {
        "ultrafast": "P1",
        "superfast": "P2",
        "veryfast": "P3",
        "faster": "P4",
        "fast": "P4",
        "medium": "P5",
        "slow": "P6",
        "slower": "P7",
        "veryslow": "P7",
        "placebo": "P7",
    }.get(p, "P5")


def _import_nvc():
    """Import NV codec bindings.

    Prefer legacy `PyNvCodec` when available, otherwise fall back to NVIDIA's
    pip-distributed successor `PyNvVideoCodec`.
    """

    # Windows: PyNvVideoCodec relies on CUDA/driver DLLs being discoverable.
    # Notably, current wheels depend on the CUDA 12 runtime DLL `cudart64_12.dll`.
    # Make a best-effort attempt to add plausible CUDA runtime directories.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        # 1) If installed via pip, `nvidia-cuda-runtime-cu12` provides cudart64_12.dll
        # under `.../site-packages/nvidia/cuda_runtime/bin`.
        try:
            import site

            roots = []
            try:
                roots.append(site.getusersitepackages())
            except Exception:
                pass
            try:
                roots.extend(site.getsitepackages() or [])
            except Exception:
                pass

            for root in roots:
                if not root:
                    continue
                dll_dir = os.path.join(root, "nvidia", "cuda_runtime", "bin")
                try:
                    if os.path.isfile(os.path.join(dll_dir, "cudart64_12.dll")):
                        os.add_dll_directory(dll_dir)
                except Exception:
                    pass
        except Exception:
            pass

        # 2) CUDA Toolkit installs (any version) often provide runtime DLLs under
        # CUDA_PATH\bin\x64 (or sometimes CUDA_PATH\bin).
        cuda_roots = []
        cuda_roots.append(os.environ.get("CUDA_PATH"))
        for name, value in os.environ.items():
            if name.startswith("CUDA_PATH_V") and value:
                cuda_roots.append(value)

        for cuda_root in cuda_roots:
            if not cuda_root:
                continue
            for rel in ("bin", os.path.join("bin", "x64"), os.path.join("lib", "x64")):
                dll_dir = os.path.join(cuda_root, rel)
                try:
                    if os.path.isdir(dll_dir):
                        os.add_dll_directory(dll_dir)
                except Exception:
                    pass

    try:
        import PyNvCodec as nvc  # type: ignore

        return nvc
    except Exception:
        import PyNvVideoCodec as nvc  # type: ignore

        return nvc


def _ffmpeg_remux_h26x(
    *,
    input_stream: str,
    output_file: str,
    fps: float,
    is_hevc: bool,
    audiofile: Optional[str],
    audio_codec: Optional[str],
    logger,
) -> None:
    stream_fmt = "hevc" if is_hevc else "h264"
    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-loglevel",
        "error",
        "-r",
        f"{fps:.02f}",
        "-f",
        stream_fmt,
        "-i",
        ffmpeg_escape_filename(input_stream),
    ]

    if audiofile is not None:
        cmd.extend(["-i", ffmpeg_escape_filename(audiofile)])
        cmd.extend(["-c:v", "copy"])
        cmd.extend(["-c:a", audio_codec or "copy"])
    else:
        cmd.extend(["-c:v", "copy", "-an"])

    cmd.append(ffmpeg_escape_filename(output_file))
    subprocess_call(cmd, logger=logger)


@dataclass
class _EncodeContext:
    gpu_id: int
    width: int
    height: int
    fps: float
    codec: str
    preset: str
    bitrate: Optional[str]


class _SurfaceConverterChain:
    def __init__(self, nvc, width: int, height: int, gpu_id: int, chain):
        self._nvc = nvc
        self._gpu_id = gpu_id
        self._cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        self._chain = [nvc.PySurfaceConverter(width, height, src, dst, gpu_id) for (src, dst) in chain]

    def run(self, surf):
        out = surf
        for cvt in self._chain:
            out = cvt.Execute(out, self._cc)
            if out.Empty():
                raise RuntimeError("GPU colorspace conversion failed")
        return out.Clone(self._gpu_id)


class PyNvCodecNVENCWriter:
    """Write H.264/HEVC using NVENC from CuPy RGB frames."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        codec: str,
        preset: str,
        bitrate: Optional[str],
        gpu_id: int,
        out_stream_path: str,
    ):
        import cupy as cp  # lazy
        nvc = _import_nvc()

        self._cp = cp
        self._nvc = nvc
        self._ctx = _EncodeContext(
            gpu_id=gpu_id,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            preset=preset,
            bitrate=bitrate,
        )
        self._out = open(out_stream_path, "wb")
        self._out_stream_path = out_stream_path
        self._packet = np.ndarray(shape=(0,), dtype=np.uint8)

        res = f"{width}x{height}"
        enc_params = {
            "codec": codec,
            "preset": preset,
            "s": res,
            "tuning_info": "high_quality",
        }
        if bitrate:
            enc_params["bitrate"] = bitrate

        self._enc = nvc.PyNvEncoder(enc_params, gpu_id)

        # Convert packed RGB to NV12 for NVENC.
        self._to_nv12 = _SurfaceConverterChain(
            nvc,
            width,
            height,
            gpu_id,
            chain=[
                (nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB),
                (nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420),
                (nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12),
            ],
        )

    def _cupy_rgb_to_surface_rgb_planar(self, frame):
        cp = self._cp
        nvc = self._nvc
        if not hasattr(frame, "__cuda_array_interface__"):
            raise RuntimeError("Expected a GPU frame (CUDA array)")

        f = frame
        if f.ndim != 3 or f.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 frame, got shape={getattr(f, 'shape', None)!r}")
        f = f[:, :, :3]
        if f.dtype != cp.uint8:
            f = f.astype(cp.uint8)
        try:
            if not f.flags.c_contiguous:
                f = cp.ascontiguousarray(f)
        except Exception:
            f = cp.ascontiguousarray(f)

        # Packed HWC -> planar CHW, then a device-to-device copy into VPF surface.
        chw = cp.transpose(f, (2, 0, 1))
        chw = cp.ascontiguousarray(chw)

        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, self._ctx.width, self._ctx.height, self._ctx.gpu_id)
        dst_ptr = surface.PlanePtr().GpuMem()
        dst_pitch = surface.Pitch()
        width_bytes = surface.Width()
        height_rows = surface.Height() * 3
        cp.cuda.runtime.memcpy2DAsync(
            dst_ptr,
            dst_pitch,
            chw.data.ptr,
            width_bytes,
            width_bytes,
            height_rows,
            cp.cuda.runtime.memcpyDeviceToDevice,
            0,
        )
        return surface

    def write_frame(self, frame_rgb):
        surface_rgb = self._cupy_rgb_to_surface_rgb_planar(frame_rgb)
        surface_nv12 = self._to_nv12.run(surface_rgb)
        success = self._enc.EncodeSingleSurface(surface_nv12, self._packet)
        if success:
            self._out.write(bytearray(self._packet))

    def close(self):
        if getattr(self, "_enc", None) is not None:
            while self._enc.FlushSinglePacket(self._packet):
                self._out.write(bytearray(self._packet))
        if getattr(self, "_out", None) is not None:
            try:
                self._out.close()
            finally:
                self._out = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def try_write_video_pynvcodec(
    *,
    clip,
    filename: str,
    fps: float,
    codec: str,
    bitrate: Optional[str],
    preset: str,
    audiofile: Optional[str],
    audio_codec: Optional[str],
    logger,
    gpu_render,
    pixel_format: Optional[str],
) -> bool:
    """Best-effort: encode via NVENC from GPU frames, then mux via FFmpeg.

    Returns True if the file was written. Returns False to indicate the caller
    should fall back to the standard FFmpeg stdin writer.
    """
    if not is_enabled():
        return False

    backend = gpu_encode_backend()
    if backend not in {"auto", "pynvcodec", "pynv", "nvenc"}:
        # Unknown/unsupported backend: do not attempt.
        return False

    if clip.mask is not None:
        # NVENC path doesn't preserve alpha.
        return False

    if pixel_format is not None:
        # Respect user request; NVENC path uses NV12 internally.
        return False

    mapped = _map_codec(codec)
    if mapped is None:
        return False

    # NV12 requires even dimensions.
    if (clip.w % 2) or (clip.h % 2):
        return False

    gpu_id = _parse_gpu_id()
    nvenc_preset = _map_nvenc_preset(preset)

    # We encode to an elementary stream, then remux to the requested container.
    suffix = ".hevc" if mapped == "hevc" else ".h264"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    try:
        # These imports are intentionally inside the try so missing optional deps
        # just trigger a fallback.
        import cupy as _cp  # noqa: F401
        _import_nvc()

        with PyNvCodecNVENCWriter(
            width=clip.w,
            height=clip.h,
            fps=float(fps),
            codec=mapped,
            preset=nvenc_preset,
            bitrate=bitrate,
            gpu_id=gpu_id,
            out_stream_path=tmp_path,
        ) as writer:
            n_frames = int(clip.duration * fps)
            for frame_index in logger.iter_bar(frame_index=range(n_frames)):
                t = np.float64(frame_index) / fps
                frame = gpu_render.get_frame_for_export_uint8_gpu(clip, t)
                writer.write_frame(frame)

        _ffmpeg_remux_h26x(
            input_stream=tmp_path,
            output_file=filename,
            fps=float(fps),
            is_hevc=(mapped == "hevc"),
            audiofile=audiofile,
            audio_codec=audio_codec,
            logger=logger,
        )
        return True
    except Exception:
        if _strict() or backend in {"pynvcodec", "pynv", "nvenc"}:
            raise
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
