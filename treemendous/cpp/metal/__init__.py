"""
Metal-Accelerated Implementations

This module provides Metal/MPS-accelerated interval tree implementations for macOS.
Works with Apple Silicon (M1/M2/M3/M4) and Intel Macs with AMD GPUs.
"""

from typing import Optional
import platform

# Check if we're on macOS
IS_MACOS = platform.system() == 'Darwin'

# Try to import Metal module
METAL_AVAILABLE = False
if IS_MACOS:
    try:
        from . import boundary_summary_metal
        METAL_AVAILABLE = boundary_summary_metal.METAL_AVAILABLE
        from .mixed import MixedBoundarySummaryManager
    except ImportError as e:
        _import_error = str(e)
        boundary_summary_metal = None
        MixedBoundarySummaryManager = None
else:
    boundary_summary_metal = None
    MixedBoundarySummaryManager = None


def is_metal_available() -> bool:
    """Check if Metal acceleration is available"""
    return METAL_AVAILABLE


def get_metal_info() -> dict:
    """Get Metal device information"""
    if not IS_MACOS:
        return {
            "available": False,
            "error": "Metal is only available on macOS",
        }
    
    if not METAL_AVAILABLE:
        return {
            "available": False,
            "error": "Metal module not available. Build with: python setup_metal.py build_ext --inplace",
        }
    
    try:
        info = boundary_summary_metal.get_metal_device_info()
        return info
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def create_metal_manager():
    """Create a Metal-accelerated boundary summary manager"""
    if not METAL_AVAILABLE:
        raise ImportError(
            "Metal acceleration not available. "
            "Build with: python setup_metal.py build_ext --inplace"
        )
    
    return boundary_summary_metal.MetalBoundarySummaryManager()


def create_mixed_metal_manager(
    summary_path: Optional[str] = None,
    best_fit_path: Optional[str] = None,
    best_fit_min_intervals: Optional[int] = None,
    summary_min_intervals: Optional[int] = None,
    sync_cpu: Optional[bool] = None,
    track_allocations: Optional[bool] = None,
):
    """Create a mixed CPU/Metal boundary summary manager."""
    if not METAL_AVAILABLE:
        raise ImportError(
            "Metal acceleration not available. "
            "Build with: python setup_metal.py build_ext --inplace"
        )
    return MixedBoundarySummaryManager(
        summary_path=summary_path,
        best_fit_path=best_fit_path,
        best_fit_min_intervals=best_fit_min_intervals,
        summary_min_intervals=summary_min_intervals,
        sync_cpu=sync_cpu,
        track_allocations=track_allocations,
    )


def benchmark_metal_speedup(num_intervals: int = 10000, num_operations: int = 5000) -> dict:
    """
    Benchmark Metal vs CPU speedup
    
    Args:
        num_intervals: Number of intervals to test with
        num_operations: Number of operations to perform
        
    Returns:
        Dictionary with benchmark results including speedup factor
    """
    if not METAL_AVAILABLE:
        return {
            "error": "Metal not available",
            "available": False,
        }
    
    return boundary_summary_metal.benchmark_metal_speedup(num_intervals, num_operations)


# Export symbols
__all__ = [
    'METAL_AVAILABLE',
    'IS_MACOS',
    'is_metal_available',
    'get_metal_info',
    'create_metal_manager',
    'create_mixed_metal_manager',
    'benchmark_metal_speedup',
]

# Conditional exports
if METAL_AVAILABLE:
    __all__.extend([
        'boundary_summary_metal',
        'MixedBoundarySummaryManager',
    ])
