"""Diagnostics and raw factories for the experimental macOS Metal backend."""

from __future__ import annotations

import platform
import warnings
from importlib import import_module
from types import ModuleType
from typing import Any

IS_MACOS = platform.system() == "Darwin"
boundary_summary_metal: ModuleType | None = None
MixedBoundarySummaryManager: type[Any] | None = None
METAL_AVAILABLE = False

if IS_MACOS:
    try:
        boundary_summary_metal = import_module(f"{__name__}.boundary_summary_metal")
        METAL_AVAILABLE = bool(boundary_summary_metal.METAL_AVAILABLE)
        mixed_module = import_module(f"{__name__}.mixed")
        MixedBoundarySummaryManager = mixed_module.MixedBoundarySummaryManager
    except ImportError:
        boundary_summary_metal = None
        MixedBoundarySummaryManager = None


def is_metal_available() -> bool:
    """Return whether the raw Metal extension reports an available device."""
    return METAL_AVAILABLE


def get_metal_info() -> dict[str, Any]:
    """Return Metal device diagnostics."""
    if not IS_MACOS:
        return {"available": False, "error": "Metal is only available on macOS"}
    if not METAL_AVAILABLE or boundary_summary_metal is None:
        return {
            "available": False,
            "error": "Metal module unavailable; build with `just build-metal`",
        }
    try:
        result: Any = boundary_summary_metal.get_metal_device_info()
        return dict(result)
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def create_metal_manager() -> Any:
    """Create the raw experimental Metal manager when its device is available."""
    if not METAL_AVAILABLE or boundary_summary_metal is None:
        raise ImportError(
            "Metal acceleration unavailable; build with `just build-metal`"
        )
    manager_type: Any = boundary_summary_metal.MetalBoundarySummaryManager
    return manager_type()


def create_mixed_metal_manager(
    summary_path: str | None = None,
    best_fit_path: str | None = None,
    best_fit_min_intervals: int | None = None,
    summary_min_intervals: int | None = None,
    sync_cpu: bool | None = None,
    track_allocations: bool | None = None,
) -> Any:
    """Create the raw mixed CPU/Metal manager when available."""
    if not METAL_AVAILABLE or MixedBoundarySummaryManager is None:
        raise ImportError(
            "Metal acceleration unavailable; build with `just build-metal`"
        )
    return MixedBoundarySummaryManager(
        summary_path=summary_path,
        best_fit_path=best_fit_path,
        best_fit_min_intervals=best_fit_min_intervals,
        summary_min_intervals=summary_min_intervals,
        sync_cpu=sync_cpu,
        track_allocations=track_allocations,
    )


def benchmark_metal_speedup(
    num_intervals: int = 10_000, num_operations: int = 5_000
) -> dict[str, Any]:
    """Run the legacy experimental native benchmark when Metal is available.

    This compatibility API is deprecated because project benchmarks use the
    oracle-validated harness under ``tests/performance``.
    """
    warnings.warn(
        "benchmark_metal_speedup is deprecated and experimental; use the "
        "validated performance harness instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if not METAL_AVAILABLE or boundary_summary_metal is None:
        return {"error": "Metal not available", "available": False}
    benchmark = getattr(boundary_summary_metal, "benchmark_metal_speedup", None)
    if benchmark is None:
        return {
            "error": "Metal extension does not provide the legacy benchmark",
            "available": True,
        }
    result: Any = benchmark(num_intervals, num_operations)
    return dict(result)


__all__ = [
    "IS_MACOS",
    "METAL_AVAILABLE",
    "benchmark_metal_speedup",
    "create_metal_manager",
    "create_mixed_metal_manager",
    "get_metal_info",
    "is_metal_available",
]
