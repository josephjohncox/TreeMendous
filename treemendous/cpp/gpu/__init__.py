"""Experimental CUDA implementation helpers.

A successful extension build/import is not treated as runtime contract support.
"""

EXPERIMENTAL_REASON = (
    "CUDA is experimental and capability-empty until hardware parity and "
    "compute-sanitizer gates pass"
)
GPU_AVAILABLE = False
try:
    from . import boundary_summary_gpu

    GPU_AVAILABLE = boundary_summary_gpu.GPU_AVAILABLE
except ImportError as exc:
    _import_error = str(exc)
    boundary_summary_gpu = None


def is_gpu_available() -> bool:
    """Return whether CUDA has passed runtime validation (currently false)."""
    return GPU_AVAILABLE


def get_gpu_info() -> dict:
    """Return CUDA diagnostics without claiming stable runtime support."""
    if not GPU_AVAILABLE:
        return {"available": False, "error": EXPERIMENTAL_REASON}
    try:
        info = boundary_summary_gpu.get_cuda_device_info()
        info["available"] = True
        return info
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def create_gpu_manager():
    """Create the raw experimental manager only after runtime validation."""
    if not GPU_AVAILABLE:
        raise ImportError(EXPERIMENTAL_REASON)
    return boundary_summary_gpu.GPUBoundarySummaryManager()


__all__ = [
    "EXPERIMENTAL_REASON",
    "GPU_AVAILABLE",
    "create_gpu_manager",
    "get_gpu_info",
    "is_gpu_available",
]
if GPU_AVAILABLE:
    __all__.append("boundary_summary_gpu")
