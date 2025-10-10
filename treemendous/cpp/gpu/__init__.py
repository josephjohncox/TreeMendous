"""
GPU-Accelerated Implementations

This module provides CUDA-accelerated interval tree implementations.
Requires CUDA toolkit and compatible GPU hardware.
"""

from typing import Optional

# Try to import GPU module
GPU_AVAILABLE = False
try:
    from . import boundary_summary_gpu
    GPU_AVAILABLE = boundary_summary_gpu.GPU_AVAILABLE
except ImportError as e:
    _import_error = str(e)
    boundary_summary_gpu = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available"""
    return GPU_AVAILABLE


def get_gpu_info() -> dict:
    """Get GPU device information"""
    if not GPU_AVAILABLE:
        return {
            "available": False,
            "error": "GPU module not available. Build with WITH_CUDA=1 to enable.",
        }
    
    try:
        info = boundary_summary_gpu.get_cuda_device_info()
        info["available"] = True
        return info
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def create_gpu_manager():
    """Create a GPU-accelerated boundary summary manager"""
    if not GPU_AVAILABLE:
        raise ImportError(
            "GPU acceleration not available. "
            "Build with: WITH_CUDA=1 python setup_gpu.py build_ext --inplace"
        )
    
    return boundary_summary_gpu.GPUBoundarySummaryManager()


def benchmark_gpu_speedup(num_intervals: int = 10000, num_operations: int = 5000) -> dict:
    """
    Benchmark GPU vs CPU speedup
    
    Args:
        num_intervals: Number of intervals to test with
        num_operations: Number of operations to perform
        
    Returns:
        Dictionary with benchmark results including speedup factor
    """
    if not GPU_AVAILABLE:
        return {
            "error": "GPU not available",
            "available": False,
        }
    
    return boundary_summary_gpu.benchmark_gpu_speedup(num_intervals, num_operations)


# Export symbols
__all__ = [
    'GPU_AVAILABLE',
    'is_gpu_available',
    'get_gpu_info',
    'create_gpu_manager',
    'benchmark_gpu_speedup',
]

# Conditional exports
if GPU_AVAILABLE:
    __all__.extend([
        'boundary_summary_gpu',
    ])

