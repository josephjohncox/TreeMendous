"""Truthful diagnostics for the experimental CUDA implementation."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

EXPERIMENTAL_REASON = (
    "CUDA is experimental and capability-empty until hardware parity and "
    "compute-sanitizer gates pass"
)

try:
    boundary_summary_gpu: ModuleType | None = import_module(
        f"{__name__}.boundary_summary_gpu"
    )
except ImportError:
    boundary_summary_gpu = None

# Building or importing the extension is not runtime contract validation.
GPU_AVAILABLE = False


def is_gpu_available() -> bool:
    """Return whether CUDA has passed runtime validation (currently false)."""
    return GPU_AVAILABLE


def get_gpu_info() -> dict[str, Any]:
    """Return diagnostics without claiming stable runtime support."""
    details: dict[str, Any] = {
        "available": False,
        "error": EXPERIMENTAL_REASON,
        "extension_built": boundary_summary_gpu is not None,
    }
    if boundary_summary_gpu is not None:
        try:
            details["device"] = boundary_summary_gpu.get_cuda_device_info()
        except Exception as exc:
            details["device_error"] = str(exc)
    return details


def create_gpu_manager() -> Any:
    """Reject construction until CUDA passes the documented hardware gate."""
    raise ImportError(EXPERIMENTAL_REASON)


__all__ = [
    "EXPERIMENTAL_REASON",
    "GPU_AVAILABLE",
    "create_gpu_manager",
    "get_gpu_info",
    "is_gpu_available",
]
