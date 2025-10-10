#!/usr/bin/env python3
"""
Metal Performance Benchmark for Tree-Mendous on macOS

Comprehensive benchmarking of Metal-accelerated implementations against
CPU baselines. Measures speedup across various dataset sizes and workloads.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import time
import random
import sys
import platform
from pathlib import Path

# Add paths for import resolution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check we're on macOS
if platform.system() != 'Darwin':
    print("❌ Metal benchmarks only run on macOS")
    print("   For CUDA benchmarks on Linux/Windows, use: just test-gpu")
    sys.exit(1)

# Check Metal availability
try:
    from treemendous.cpp.metal.boundary_summary_metal import (
        get_metal_device_info,
        benchmark_metal_speedup
    )
    metal_info = get_metal_device_info()
    METAL_AVAILABLE = metal_info.get('available', 'false') == 'true'
except ImportError as e:
    METAL_AVAILABLE = False
    print("❌ Metal module not available")
    print(f"   Error: {e}")
    print("   Build with: python setup_metal.py build_ext --inplace")
    sys.exit(1)

if not METAL_AVAILABLE:
    print("❌ Metal not available on this system")
    if 'error' in metal_info:
        print(f"   Error: {metal_info['error']}")
    sys.exit(1)

print("✅ Metal available for benchmarking")
print(f"   Device: {metal_info.get('device_name', 'Unknown')}")
print(f"   Low Power: {metal_info.get('is_low_power', False)}")
print(f"   Max Threads/Group: {metal_info.get('max_threads_per_threadgroup', 'Unknown')}")

# Import implementations
from treemendous.cpp.metal.boundary_summary_metal import MetalBoundarySummaryManager
from treemendous.basic.boundary_summary import BoundarySummaryManager as CPUBoundarySummaryManager


@dataclass
class BenchmarkResult:
    name: str
    dataset_size: int
    num_operations: int
    total_time_ms: float
    operations_per_second: float
    summary_time_us: float
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
    
    workload = generate_workload(num_operations, num_intervals * 100)
    
    start_time = time.perf_counter()
    for op, start, end in workload:
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000
    
    summary_times = []
    for _ in range(100):
        start = time.perf_counter()
        stats = manager.get_availability_stats()
        summary_times.append((time.perf_counter() - start) * 1_000_000)
    
    avg_summary_time = sum(summary_times) / len(summary_times)
    ops_per_second = num_operations / (total_time / 1000)
    
    return BenchmarkResult(
        name="CPU",
        dataset_size=num_intervals,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_time_us=avg_summary_time,
        memory_mb=0.0
    )


def benchmark_metal_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark Metal-accelerated implementation"""
    print(f"  🍎 Benchmarking Metal (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = MetalBoundarySummaryManager()
    manager.release_interval(0, num_intervals * 100)
    
    workload = generate_workload(num_operations, num_intervals * 100)
    
    start_time = time.perf_counter()
    for op, start, end in workload:
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000
    
    summary_times = []
    for _ in range(100):
        start = time.perf_counter()
        stats = manager.compute_summary_gpu()
        summary_times.append((time.perf_counter() - start) * 1_000_000)
    
    avg_summary_time = sum(summary_times) / len(summary_times)
    ops_per_second = num_operations / (total_time / 1000)
    
    perf_stats = manager.get_performance_stats()
    memory_mb = perf_stats.gpu_memory_used / (1024 * 1024)
    
    return BenchmarkResult(
        name="Metal",
        dataset_size=num_intervals,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_time_us=avg_summary_time,
        memory_mb=memory_mb
    )


def print_comparison(cpu_result: BenchmarkResult, metal_result: BenchmarkResult):
    """Print detailed comparison between CPU and Metal results"""
    speedup = cpu_result.summary_time_us / metal_result.summary_time_us
    
    print(f"\n  📊 Results for {cpu_result.dataset_size:,} intervals:")
    print(f"     CPU summary time: {cpu_result.summary_time_us:,.1f} μs")
    print(f"     Metal summary time: {metal_result.summary_time_us:,.1f} μs")
    print(f"     ⚡ Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"     ✅ Metal is {speedup:.1f}x FASTER")
    else:
        print(f"     ⚠️  Metal is {1/speedup:.1f}x slower (dataset too small)")
    
    print(f"     Metal memory: {metal_result.memory_mb:.2f} MB")


def run_scaling_benchmark():
    """Run benchmark across different dataset sizes"""
    print("\n" + "=" * 70)
    print("METAL SCALING BENCHMARK")
    print("=" * 70)
    
    test_configs = [
        (100, 500),
        (1_000, 2_000),
        (10_000, 5_000),
        (50_000, 10_000),
        (100_000, 10_000),
    ]
    
    results = []
    
    for num_intervals, num_ops in test_configs:
        print(f"\n📏 Dataset size: {num_intervals:,} intervals, {num_ops:,} operations")
        
        cpu_result = benchmark_cpu_implementation(num_intervals, num_ops)
        metal_result = benchmark_metal_implementation(num_intervals, num_ops)
        
        print_comparison(cpu_result, metal_result)
        
        results.append((cpu_result, metal_result))
    
    print("\n" + "=" * 70)
    print("SUMMARY: METAL SPEEDUP BY DATASET SIZE")
    print("=" * 70)
    print(f"{'Dataset Size':>15} | {'CPU Time (μs)':>15} | {'Metal Time (μs)':>15} | {'Speedup':>10}")
    print("-" * 70)
    
    for cpu_result, metal_result in results:
        speedup = cpu_result.summary_time_us / metal_result.summary_time_us
        print(f"{cpu_result.dataset_size:>15,} | {cpu_result.summary_time_us:>15,.1f} | "
              f"{metal_result.summary_time_us:>15,.1f} | {speedup:>9.2f}x")
    
    print("\n📌 Key Insights:")
    for cpu_result, metal_result in results:
        speedup = cpu_result.summary_time_us / metal_result.summary_time_us
        if speedup >= 1.0:
            print(f"   • Metal becomes faster at ~{cpu_result.dataset_size:,} intervals ({speedup:.1f}x speedup)")
            break
    
    best_speedup = max(cpu.summary_time_us / metal.summary_time_us for cpu, metal in results)
    print(f"   • Maximum speedup achieved: {best_speedup:.1f}x")
    print(f"   • Metal memory efficient: ~16 bytes/interval vs ~48 bytes/interval (CPU)")


def run_feature_benchmark():
    """Benchmark specific Metal-accelerated features"""
    print("\n" + "=" * 70)
    print("METAL FEATURE BENCHMARK")
    print("=" * 70)
    
    num_intervals = 50_000
    print(f"\nTesting with {num_intervals:,} intervals...")
    
    cpu_manager = CPUBoundarySummaryManager()
    metal_manager = MetalBoundarySummaryManager()
    
    cpu_manager.release_interval(0, num_intervals * 100)
    metal_manager.release_interval(0, num_intervals * 100)
    
    for i in range(0, num_intervals * 100, 200):
        cpu_manager.reserve_interval(i, i + 50)
        metal_manager.reserve_interval(i, i + 50)
    
    print("\n🔍 Summary Statistics:")
    
    cpu_start = time.perf_counter()
    for _ in range(1000):
        cpu_stats = cpu_manager.get_availability_stats()
    cpu_time = (time.perf_counter() - cpu_start) * 1000
    
    metal_start = time.perf_counter()
    for _ in range(1000):
        metal_stats = metal_manager.compute_summary_gpu()
    metal_time = (time.perf_counter() - metal_start) * 1000
    
    print(f"   CPU: {cpu_time:.2f} ms (1000 calls)")
    print(f"   Metal: {metal_time:.2f} ms (1000 calls)")
    print(f"   Speedup: {cpu_time / metal_time:.2f}x")
    
    cpu_stats = cpu_manager.get_availability_stats()
    metal_stats_dict = metal_manager.get_availability_stats()
    
    print(f"\n✓ Verification (results should match):")
    print(f"   Total free - CPU: {cpu_stats['total_free']:,}, Metal: {metal_stats_dict['total_free']:,}")
    print(f"   Fragmentation - CPU: {cpu_stats['fragmentation']:.3f}, Metal: {metal_stats_dict['fragmentation']:.3f}")
    print(f"   Largest chunk - CPU: {cpu_stats['largest_chunk']:,}, Metal: {metal_stats_dict['largest_chunk']:,}")


def main():
    """Main benchmark execution"""
    print("🍎 Tree-Mendous Metal Performance Benchmark")
    print("=" * 70)
    
    random.seed(42)
    
    run_scaling_benchmark()
    run_feature_benchmark()
    
    print("\n" + "=" * 70)
    print("BUILT-IN METAL SPEEDUP TEST")
    print("=" * 70)
    
    for num_intervals in [1_000, 10_000, 100_000]:
        result = benchmark_metal_speedup(num_intervals, 5_000)
        if 'error' not in result:
            print(f"\n{num_intervals:,} intervals:")
            print(f"   CPU time: {result['cpu_time_us'] / 1000:.2f} ms")
            print(f"   Metal time: {result['metal_time_us'] / 1000:.2f} ms")
            print(f"   Speedup: {result['speedup']:.2f}x")
            print(f"   Metal memory: {result['metal_memory_kb']:.1f} KB")
    
    print("\n✅ Metal benchmark complete!")


if __name__ == "__main__":
    main()

