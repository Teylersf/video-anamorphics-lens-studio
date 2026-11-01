"""Small, shared PyTorch rendering-device policy. GPL-2.0-only.

STUDIO_DEVICE=auto (default) prefers CUDA, then Metal MPS, then CPU. An explicit
unavailable accelerator fails rather than silently changing the user's choice.
Availability is not a throughput/VRAM guarantee; rendering errors are propagated.
This module never changes global MPS fallback flags, starts services, or installs
packages. Uses PyTorch APIs available in the supported 2.11 release.
"""
from __future__ import annotations

import os

import torch

_MODES = {"auto", "cuda", "mps", "cpu"}


def _available(kind: str) -> bool:
    try:
        if kind == "cuda":
            return bool(torch.cuda.is_available())
        if kind == "mps":
            backend = getattr(torch.backends, "mps", None)
            return bool(backend and backend.is_available())
    except (RuntimeError, AssertionError, OSError):
        return False
    return kind == "cpu"


def _unavailable(kind: str) -> str:
    if kind == "cuda":
        return ("CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch "
                "and working NVIDIA driver, or set STUDIO_DEVICE=auto or cpu.")
    backend = getattr(torch.backends, "mps", None)
    try:
        built = bool(backend and backend.is_built())
    except (RuntimeError, AssertionError, OSError):
        built = False
    if not built:
        return ("MPS was requested but this PyTorch build has no Metal support. "
                "Use a compatible macOS PyTorch build, or set STUDIO_DEVICE=cpu.")
    return ("MPS was requested but Metal is unavailable on this hardware/macOS. "
            "Check PyTorch's macOS requirements, or set STUDIO_DEVICE=cpu.")


def select_device(requested: str | None = None) -> torch.device:
    """Select auto/CUDA/MPS/CPU; None reads STUDIO_DEVICE (default auto)."""
    value = os.environ.get("STUDIO_DEVICE", "auto") if requested is None else requested
    if not isinstance(value, str):
        raise TypeError("device choice must be auto, cuda, mps or cpu")
    value = value.strip().lower()
    if value not in _MODES:
        raise ValueError("STUDIO_DEVICE must be auto, cuda, mps or cpu")
    if value == "auto":
        for kind in ("cuda", "mps", "cpu"):
            if _available(kind):
                return torch.device(kind)
    if not _available(value):
        raise RuntimeError(_unavailable(value))
    return torch.device(value)


def _device(value, *, check_available=True) -> torch.device:
    if value is None or isinstance(value, str) and value.strip().lower() in _MODES:
        if not check_available and isinstance(value, str) and value.strip().lower() != "auto":
            return torch.device(value.strip().lower())
        return select_device(value)
    try:
        result = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must identify CPU, CUDA or MPS") from error
    if result.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("only CPU, CUDA and MPS rendering devices are supported")
    return result


def backend_info(device: torch.device | str | None = None) -> dict:
    """JSON-safe backend description; no tensor allocation or speed claim."""
    selected = _device(device)
    if not _available(selected.type):
        raise RuntimeError(_unavailable(selected.type))
    if selected.type == "cuda":
        gpu = str(torch.cuda.get_device_name(selected))
        reason = "CUDA rendering is available."
    elif selected.type == "mps":
        get_name = getattr(torch.backends.mps, "get_name", None)
        try:
            gpu = str(get_name()) if get_name else "Metal GPU (MPS)"
        except (RuntimeError, AssertionError, OSError):
            gpu = "Metal GPU (MPS)"
        reason = "Metal MPS rendering is available; supported float operations run on the GPU."
    else:
        gpu = "CPU"
        reason = "CPU rendering selected; processing may be slower than an available accelerator."
    return {"device": str(selected), "gpu": gpu, "accelerated": selected.type != "cpu", "reason": reason}


def empty_cache(device: torch.device | str) -> None:
    """Release idle allocator cache on the selected backend; CPU is a no-op.

    Cleanup of an unavailable accelerator is a no-op, so it cannot hide an
    earlier device-selection error. Live tensors are never freed by this call.
    """
    selected = _device(device, check_available=False)
    if selected.type == "cuda" and _available("cuda"):
        with torch.cuda.device(selected):
            torch.cuda.empty_cache()
    elif selected.type == "mps" and _available("mps"):
        torch.mps.empty_cache()


__all__ = ["select_device", "backend_info", "empty_cache"]
