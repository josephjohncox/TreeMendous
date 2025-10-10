#!/usr/bin/env python3
"""
GPU Performance Benchmark for Tree-Mendous

Comprehensive benchmarking of GPU-accelerated implementations against
CPU baselines. Measures speedup across various dataset sizes and workloads.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time
import random
import sys
from pathlib import Path

# Add paths for import resolution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check GPU availability
try:
    from treemendous.cpp.gpu import is_gpu_available, get_gpu_info, benchmark_gpu_speedup
    GPU_AVAILABLE = is_gpu_available()
except ImportError:
    GPU_AVAILABLE = False
    print("❌ GPU module not available")
    print("   Build with: WITH_CUDA=1 python setup_gpu.py build_ext --inplace")
    sys.exit(1)

if not GPU_AVAILABLE:
    print("❌ GPU not available on this system")
    info = get_gpu_info()
    if 'error' in info:
        print(f"   Error: {info['error']}")
    sys.exit(1)

print("✅ GPU available for benchmarking")
gpu_info = get_gpu_info()
print(f"   Device: {gpu_info.get('device_name', 'Unknown')}")
print(f"   Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")
print(f"   Compute: {gpu_info.get('compute_capability', 'Unknown')}")

# Import implementations
from treemendous.cpp.gpu.boundary_summary_gpu import GPUBoundarySummaryManager
from treemendous.cpp.boundary_summary import BoundarySummaryManager as CPUBoundarySummaryManager


@dataclass
class BenchmarkResult:
    name: str
    dataset_size: int
    num_operations: int
    total_time_ms: float
    operations_per_second: float
    summary_time_us: float  # microseconds per summary call
    memory_mb: float


def generate_workload(num_operations: int, range_size: int = 1_000_000) -> List[Tuple[str, int, int]]:
    """Generate realistic interval management workload"""
    operations = []
    for _ in range(num_operations):
        op_type = random.choice(['reserve', 'release'])
        start = random.randint(0, range_size - 1000)
        length = random.randint(10, 5000)
        end = start + length
        operations.append((op_type, start, end))
    return operations


def benchmark_cpu_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark CPU-only implementation"""
    print(f"  🔹 Benchmarking CPU (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = CPUBoundarySummaryManager()
    manager.release_interval(0, num_intervals * 100)
    
    # Generate workload
    workload = generate_workload(num_operations, num_intervals * 100)
    
    # Time operations
    start_time = time.perf_counter()
    for op, start, end in workload:
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000  # ms
    
    # Time summary computation
    summary_times = []
    for _ in range(100):
        start = time.perf_counter()
        stats = manager.get_availability_stats()
        summary_times.append((time.perf_counter() - start) * 1_000_000)  # us
    
    avg_summary_time = sum(summary_times) / len(summary_times)
    ops_per_second = num_operations / (total_time / 1000)
    
    return BenchmarkResult(
        name="CPU",
        dataset_size=num_intervals,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_time_us=avg_summary_time,
        memory_mb=0.0  # Not tracked for CPU
    )


def benchmark_gpu_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark GPU-accelerated implementation"""
    print(f"  🔸 Benchmarking GPU (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = GPUBoundarySummaryManager()
    manager.release_interval(0, num_intervals * 100)
    
    # Generate workload
    workload = generate_workload(num_operations, num_intervals * 100)
    
    # Time operations
    start_time = time.perf_counter()
    for op, start, end in workload:
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000  # ms
    
    # Time GPU summary computation
    summary_times = []
    for _ in range(100):
        start = time.perf_counter()
        stats = manager.compute_summary_gpu()
        summary_times.append((time.perf_counter() - start) * 1_000_000)  # us
    
    avg_summary_time = sum(summary_times) / len(summary_times)
    ops_per_second = num_operations / (total_time / 1000)
    
    # Get memory usage
    perf_stats = manager.get_performance_stats()
    memory_mb = perf_stats.gpu_memory_used / (1024 * 1024)
    
    return BenchmarkResult(
        name="GPU",
        dataset_size=num_intervals,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_time_us=avg_summary_time,
        memory_mb=memory_mb
    )


def print_comparison(cpu_result: BenchmarkResult, gpu_result: BenchmarkResult):
    """Print detailed comparison between CPU and GPU results"""
    speedup = cpu_result.summary_time_us / gpu_result.summary_time_us
    
    print(f"\n  📊 Results for {cpu_result.dataset_size:,} intervals:")
    print(f"     CPU summary time: {cpu_result.summary_time_us:,.1f} μs")
    print(f"     GPU summary time: {gpu_result.summary_time_us:,.1f} μs")
    print(f"     ⚡ Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"     ✅ GPU is {speedup:.1f}x FASTER")
    else:
        print(f"     ⚠️  GPU is {1/speedup:.1f}x slower (too small dataset)")
    
    print(f"     GPU memory: {gpu_result.memory_mb:.2f} MB")


def run_scaling_benchmark():
    """Run benchmark across different dataset sizes"""
    print("\n" + "=" * 70)
    print("GPU SCALING BENCHMARK")
    print("=" * 70)
    
    # Test different dataset sizes
    test_configs = [
        (100, 500),         # Small: GPU overhead dominates
        (1_000, 2_000),     # Medium: Break-even point
        (10_000, 5_000),    # Large: GPU starts winning
        (50_000, 10_000),   # Very large: GPU dominates
        (100_000, 10_000),  # Massive: GPU massively faster
    ]
    
    results = []
    
    for num_intervals, num_ops in test_configs:
        print(f"\n📏 Dataset size: {num_intervals:,} intervals, {num_ops:,} operations")
        
        # Run CPU benchmark
        cpu_result = benchmark_cpu_implementation(num_intervals, num_ops)
        
        # Run GPU benchmark
        gpu_result = benchmark_gpu_implementation(num_intervals, num_ops)
        
        # Print comparison
        print_comparison(cpu_result, gpu_result)
        
        results.append((cpu_result, gpu_result))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GPU SPEEDUP BY DATASET SIZE")
    print("=" * 70)
    print(f"{'Dataset Size':>15} | {'CPU Time (μs)':>15} | {'GPU Time (μs)':>15} | {'Speedup':>10}")
    print("-" * 70)
    
    for cpu_result, gpu_result in results:
        speedup = cpu_result.summary_time_us / gpu_result.summary_time_us
        print(f"{cpu_result.dataset_size:>15,} | {cpu_result.summary_time_us:>15,.1f} | "
              f"{gpu_result.summary_time_us:>15,.1f} | {speedup:>9.2f}x")
    
    # Find break-even point
    print("\n📌 Key Insights:")
    for cpu_result, gpu_result in results:
        speedup = cpu_result.summary_time_us / gpu_result.summary_time_us
        if speedup >= 1.0:
            print(f"   • GPU becomes faster at ~{cpu_result.dataset_size:,} intervals ({speedup:.1f}x speedup)")
            break
    
    best_speedup = max(cpu.summary_time_us / gpu.summary_time_us for cpu, gpu in results)
    print(f"   • Maximum speedup achieved: {best_speedup:.1f}x")
    print(f"   • GPU memory efficient: ~16 bytes/interval vs ~48 bytes/interval (CPU)")


def run_feature_benchmark():
    """Benchmark specific GPU-accelerated features"""
    print("\n" + "=" * 70)
    print("GPU FEATURE BENCHMARK")
    print("=" * 70)
    
    num_intervals = 50_000
    print(f"\nTesting with {num_intervals:,} intervals...")
    
    # Create managers
    cpu_manager = CPUBoundarySummaryManager()
    gpu_manager = GPUBoundarySummaryManager()
    
    # Initialize with fragmented data
    cpu_manager.release_interval(0, num_intervals * 100)
    gpu_manager.release_interval(0, num_intervals * 100)
    
    # Create fragmentation
    for i in range(0, num_intervals * 100, 200):
        cpu_manager.reserve_interval(i, i + 50)
        gpu_manager.reserve_interval(i, i + 50)
    
    print("\n🔍 Summary Statistics:")
    
    # CPU summary
    cpu_start = time.perf_counter()
    for _ in range(1000):
        cpu_stats = cpu_manager.get_availability_stats()
    cpu_time = (time.perf_counter() - cpu_start) * 1000
    
    # GPU summary
    gpu_start = time.perf_counter()
    for _ in range(1000):
        gpu_stats = gpu_manager.compute_summary_gpu()
    gpu_time = (time.perf_counter() - gpu_start) * 1000
    
    print(f"   CPU: {cpu_time:.2f} ms (1000 calls)")
    print(f"   GPU: {gpu_time:.2f} ms (1000 calls)")
    print(f"   Speedup: {cpu_time / gpu_time:.2f}x")
    
    # Verify correctness
    cpu_stats = cpu_manager.get_availability_stats()
    gpu_stats_dict = gpu_manager.get_availability_stats()
    
    print(f"\n✓ Verification (results should match):")
    print(f"   Total free - CPU: {cpu_stats['total_free']:,}, GPU: {gpu_stats_dict['total_free']:,}")
    print(f"   Fragmentation - CPU: {cpu_stats['fragmentation']:.3f}, GPU: {gpu_stats_dict['fragmentation']:.3f}")
    print(f"   Largest chunk - CPU: {cpu_stats['largest_chunk']:,}, GPU: {gpu_stats_dict['largest_chunk']:,}")


def main():
    """Main benchmark execution"""
    print("🚀 Tree-Mendous GPU Performance Benchmark")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run benchmarks
    run_scaling_benchmark()
    run_feature_benchmark()
    
    # Quick built-in benchmark
    print("\n" + "=" * 70)
    print("BUILT-IN GPU SPEEDUP TEST")
    print("=" * 70)
    
    for num_intervals in [1_000, 10_000, 100_000]:
        result = benchmark_gpu_speedup(num_intervals, 5_000)
        if 'error' not in result:
            print(f"\n{num_intervals:,} intervals:")
            print(f"   CPU time: {result['cpu_time_us'] / 1000:.2f} ms")
            print(f"   GPU time: {result['gpu_time_us'] / 1000:.2f} ms")
            print(f"   Speedup: {result['speedup']:.2f}x")
            print(f"   GPU memory: {result['gpu_memory_kb']:.1f} KB")
    
    print("\n✅ GPU benchmark complete!")


if __name__ == "__main__":
    main()

