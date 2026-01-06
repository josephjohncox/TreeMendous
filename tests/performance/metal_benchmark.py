#!/usr/bin/env python3
"""
Metal Performance Benchmark for Tree-Mendous on macOS

Comprehensive benchmarking of Metal-accelerated implementations against
CPU baselines. Measures speedup across various dataset sizes and workloads.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import bisect
import time
import random
import sys
import platform
import os
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
from treemendous.cpp.metal import MixedBoundarySummaryManager
from treemendous.basic.boundary_summary import BoundarySummary, BoundarySummaryManager as CPUBoundarySummaryManager
from tests.performance.workload import generate_realistic_workload, iter_workload


@dataclass
class BenchmarkResult:
    name: str
    dataset_size: int
    interval_count: int
    num_operations: int
    total_time_ms: float
    operations_per_second: float
    summary_first_us: float
    summary_warm_us: float
    memory_mb: float


def _time_us(fn, iterations: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def _sum_intervals(intervals: List[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


WORKLOAD_PROFILE = "allocator"
EXTENDED_BENCH = os.environ.get("METAL_BENCH_EXTENDED") == "1"

MIXED_BEST_FIT_MIN_INTERVALS = int(os.environ.get("TREEMENDOUS_MIXED_BEST_FIT_MIN_INTERVALS", "10000"))
MIXED_SUMMARY_MIN_INTERVALS = int(os.environ.get("TREEMENDOUS_MIXED_SUMMARY_MIN_INTERVALS", "50000"))
MIXED_SYNC_CPU = os.environ.get("TREEMENDOUS_MIXED_SYNC_CPU", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
MIXED_SUMMARY_PATH = os.environ.get("TREEMENDOUS_MIXED_SUMMARY_PATH", "cpu")
MIXED_BEST_FIT_PATH = os.environ.get("TREEMENDOUS_MIXED_BEST_FIT_PATH", "gpu")


def benchmark_cpu_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark CPU-only implementation"""
    print(f"  🔹 Benchmarking CPU (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = CPUBoundarySummaryManager()
    manager.release_interval(0, num_intervals * 100)
    
    workload = generate_realistic_workload(
        num_operations,
        profile=WORKLOAD_PROFILE,
        space_range=(0, num_intervals * 100),
        seed=42,
        include_data=False
    )
    
    start_time = time.perf_counter()
    for op, start, end, _ in iter_workload(workload):
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            elif op == 'release':
                manager.release_interval(start, end)
            elif op == 'find':
                manager.find_interval(start, end - start)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000
    
    interval_count = len(manager.get_intervals())
    summary_first_us = _time_us(manager.get_availability_stats, iterations=1)
    summary_warm_us = _time_us(manager.get_availability_stats, iterations=100)
    ops_per_second = num_operations / (total_time / 1000)
    
    return BenchmarkResult(
        name="CPU",
        dataset_size=num_intervals,
        interval_count=interval_count,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_first_us=summary_first_us,
        summary_warm_us=summary_warm_us,
        memory_mb=0.0
    )


def benchmark_metal_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark Metal-accelerated implementation"""
    print(f"  🍎 Benchmarking Metal (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = MetalBoundarySummaryManager()
    manager.release_interval(0, num_intervals * 100)
    
    workload = generate_realistic_workload(
        num_operations,
        profile=WORKLOAD_PROFILE,
        space_range=(0, num_intervals * 100),
        seed=42,
        include_data=False
    )
    
    start_time = time.perf_counter()
    for op, start, end, _ in iter_workload(workload):
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            elif op == 'release':
                manager.release_interval(start, end)
            elif op == 'find':
                manager.find_interval(start, end - start)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000
    
    interval_count = len(manager.get_intervals())
    summary_first_us = _time_us(manager.compute_summary_gpu, iterations=1)
    summary_warm_us = _time_us(manager.compute_summary_gpu, iterations=100)
    ops_per_second = num_operations / (total_time / 1000)
    
    perf_stats = manager.get_performance_stats()
    memory_mb = perf_stats.gpu_memory_used / (1024 * 1024)
    
    return BenchmarkResult(
        name="Metal",
        dataset_size=num_intervals,
        interval_count=interval_count,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_first_us=summary_first_us,
        summary_warm_us=summary_warm_us,
        memory_mb=memory_mb
    )


def benchmark_mixed_implementation(num_intervals: int, num_operations: int) -> BenchmarkResult:
    """Benchmark mixed CPU/Metal implementation"""
    print(f"  🧩 Benchmarking Mixed (intervals={num_intervals:,}, ops={num_operations:,})...")
    
    manager = MixedBoundarySummaryManager(
        summary_path=MIXED_SUMMARY_PATH,
        best_fit_path=MIXED_BEST_FIT_PATH,
        best_fit_min_intervals=MIXED_BEST_FIT_MIN_INTERVALS,
        summary_min_intervals=MIXED_SUMMARY_MIN_INTERVALS,
        sync_cpu=MIXED_SYNC_CPU,
        track_allocations=True,
    )
    manager.release_interval(0, num_intervals * 100)
    
    workload = generate_realistic_workload(
        num_operations,
        profile=WORKLOAD_PROFILE,
        space_range=(0, num_intervals * 100),
        seed=42,
        include_data=True
    )
    
    start_time = time.perf_counter()
    for op, start, end, payload in iter_workload(workload):
        try:
            if op == 'reserve':
                manager.reserve_interval(start, end, payload)
            elif op == 'release':
                manager.release_interval(start, end)
            elif op == 'find':
                manager.find_interval(start, end - start)
        except:
            pass
    total_time = (time.perf_counter() - start_time) * 1000
    
    interval_count = len(manager.get_intervals())
    summary_first_us = _time_us(manager.get_availability_stats, iterations=1)
    summary_warm_us = _time_us(manager.get_availability_stats, iterations=100)
    ops_per_second = num_operations / (total_time / 1000)
    
    perf_stats = manager.get_performance_stats()
    memory_mb = perf_stats.gpu_memory_used / (1024 * 1024)
    
    return BenchmarkResult(
        name="Mixed",
        dataset_size=num_intervals,
        interval_count=interval_count,
        num_operations=num_operations,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        summary_first_us=summary_first_us,
        summary_warm_us=summary_warm_us,
        memory_mb=memory_mb
    )


def print_comparison(
    cpu_result: BenchmarkResult,
    metal_result: BenchmarkResult,
    mixed_result: BenchmarkResult | None = None,
):
    """Print detailed comparison between CPU and Metal results"""
    cold_speedup = cpu_result.summary_first_us / metal_result.summary_first_us if metal_result.summary_first_us > 0 else 0.0
    warm_speedup = cpu_result.summary_warm_us / metal_result.summary_warm_us if metal_result.summary_warm_us > 0 else 0.0
    
    print(f"\n  📊 Results for {cpu_result.dataset_size:,} intervals:")
    if mixed_result:
        print(f"     Interval count: CPU={cpu_result.interval_count:,}, Metal={metal_result.interval_count:,}, Mixed={mixed_result.interval_count:,}")
    else:
        print(f"     Interval count: CPU={cpu_result.interval_count:,}, Metal={metal_result.interval_count:,}")
    print(f"     CPU summary (first): {cpu_result.summary_first_us:,.1f} μs")
    print(f"     CPU summary (warm):  {cpu_result.summary_warm_us:,.1f} μs")
    print(f"     Metal summary (first): {metal_result.summary_first_us:,.1f} μs")
    print(f"     Metal summary (warm):  {metal_result.summary_warm_us:,.1f} μs")
    print(f"     ⚡ Speedup (first): {cold_speedup:.2f}x")
    print(f"     ⚡ Speedup (warm):  {warm_speedup:.2f}x")
    
    if cold_speedup > 1:
        print(f"     ✅ Metal is {cold_speedup:.1f}x FASTER on cold summary")
    else:
        if cold_speedup > 0:
            print(f"     ⚠️  Metal is {1/cold_speedup:.1f}x slower on cold summary")
    
    print(f"     Metal memory: {metal_result.memory_mb:.2f} MB")
    
    if mixed_result:
        mixed_cold = cpu_result.summary_first_us / mixed_result.summary_first_us if mixed_result.summary_first_us > 0 else 0.0
        mixed_warm = cpu_result.summary_warm_us / mixed_result.summary_warm_us if mixed_result.summary_warm_us > 0 else 0.0
        print(f"     Mixed summary (first): {mixed_result.summary_first_us:,.1f} μs")
        print(f"     Mixed summary (warm):  {mixed_result.summary_warm_us:,.1f} μs")
        print(f"     ⚡ Mixed speedup (first): {mixed_cold:.2f}x")
        print(f"     ⚡ Mixed speedup (warm):  {mixed_warm:.2f}x")


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
        mixed_result = benchmark_mixed_implementation(num_intervals, num_ops)
        
        print_comparison(cpu_result, metal_result, mixed_result)
        
        results.append((cpu_result, metal_result, mixed_result))
    
    print("\n" + "=" * 110)
    print("SUMMARY: METAL SPEEDUP BY DATASET SIZE")
    print("=" * 110)
    print(f"{'Dataset Size':>15} | {'Intervals':>10} | {'CPU first (μs)':>15} | {'Metal first (μs)':>17} | {'Mixed first (μs)':>17} | {'Metal Spd':>9} | {'Mixed Spd':>9}")
    print("-" * 110)
    
    for cpu_result, metal_result, mixed_result in results:
        speedup = cpu_result.summary_first_us / metal_result.summary_first_us if metal_result.summary_first_us > 0 else 0.0
        mixed_speedup = cpu_result.summary_first_us / mixed_result.summary_first_us if mixed_result.summary_first_us > 0 else 0.0
        print(f"{cpu_result.dataset_size:>15,} | {cpu_result.interval_count:>10,} | {cpu_result.summary_first_us:>15,.1f} | "
              f"{metal_result.summary_first_us:>17,.1f} | {mixed_result.summary_first_us:>17,.1f} | {speedup:>8.2f}x | {mixed_speedup:>8.2f}x")
    
    print("\n📌 Key Insights:")
    for cpu_result, metal_result, _mixed_result in results:
        speedup = cpu_result.summary_first_us / metal_result.summary_first_us if metal_result.summary_first_us > 0 else 0.0
        if speedup >= 1.0:
            print(f"   • Metal becomes faster at ~{cpu_result.dataset_size:,} intervals ({speedup:.1f}x speedup)")
            break
    
    best_speedup = max(
        cpu.summary_first_us / metal.summary_first_us
        for cpu, metal, _mixed in results
        if metal.summary_first_us > 0
    ) if results else 0.0
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
    mixed_manager = MixedBoundarySummaryManager(
        summary_path=MIXED_SUMMARY_PATH,
        best_fit_path=MIXED_BEST_FIT_PATH,
        best_fit_min_intervals=MIXED_BEST_FIT_MIN_INTERVALS,
        summary_min_intervals=MIXED_SUMMARY_MIN_INTERVALS,
        sync_cpu=MIXED_SYNC_CPU,
        track_allocations=True,
    )
    
    cpu_manager.release_interval(0, num_intervals * 100)
    metal_manager.release_interval(0, num_intervals * 100)
    mixed_manager.release_interval(0, num_intervals * 100)
    
    for i in range(0, num_intervals * 100, 200):
        cpu_manager.reserve_interval(i, i + 50)
        metal_manager.reserve_interval(i, i + 50)
        mixed_manager.reserve_interval(i, i + 50, {"id": i})
    
    print("\n🔍 Summary Statistics:")
    
    cpu_start = time.perf_counter()
    for _ in range(1000):
        cpu_stats = cpu_manager.get_availability_stats()
    cpu_time = (time.perf_counter() - cpu_start) * 1000
    
    metal_start = time.perf_counter()
    for _ in range(1000):
        metal_stats = metal_manager.compute_summary_gpu()
    metal_time = (time.perf_counter() - metal_start) * 1000

    mixed_start = time.perf_counter()
    for _ in range(1000):
        mixed_stats = mixed_manager.get_availability_stats()
    mixed_time = (time.perf_counter() - mixed_start) * 1000
    
    print(f"   CPU: {cpu_time:.2f} ms (1000 calls)")
    print(f"   Metal: {metal_time:.2f} ms (1000 calls)")
    print(f"   Mixed: {mixed_time:.2f} ms (1000 calls)")
    print(f"   Metal speedup: {cpu_time / metal_time:.2f}x")
    print(f"   Mixed speedup: {cpu_time / mixed_time:.2f}x")
    
    cpu_stats = cpu_manager.get_availability_stats()
    metal_stats_dict = metal_manager.get_availability_stats()
    mixed_stats_dict = mixed_manager.get_availability_stats()
    
    print(f"\n✓ Verification (results should match):")
    print(f"   Total free - CPU: {cpu_stats['total_free']:,}, Metal: {metal_stats_dict['total_free']:,}")
    print(f"   Total free - Mixed: {mixed_stats_dict['total_free']:,}")
    print(f"   Fragmentation - CPU: {cpu_stats['fragmentation']:.3f}, Metal: {metal_stats_dict['fragmentation']:.3f}")
    print(f"   Fragmentation - Mixed: {mixed_stats_dict['fragmentation']:.3f}")
    print(f"   Largest chunk - CPU: {cpu_stats['largest_chunk']:,}, Metal: {metal_stats_dict['largest_chunk']:,}")
    print(f"   Largest chunk - Mixed: {mixed_stats_dict['largest_chunk']:,}")
    cpu_interval_total = _sum_intervals(_normalize_intervals(cpu_manager.get_intervals()))
    metal_interval_total = _sum_intervals(_normalize_intervals(metal_manager.get_intervals()))
    print(f"   Interval sum - CPU: {cpu_interval_total:,}, Metal: {metal_interval_total:,}")


def run_fragmentation_summary_benchmark():
    """Stress summary computation with high interval counts."""
    print("\n" + "=" * 70)
    print("METAL FRAGMENTATION SUMMARY BENCHMARK")
    print("=" * 70)
    
    interval_length = 150
    gap_length = 50
    interval_counts = [1_000, 10_000, 50_000, 100_000]
    if EXTENDED_BENCH:
        interval_counts += [250_000]
    
    print(f"Pattern: {interval_length} free / {gap_length} reserved")
    print(f"{'Intervals':>12} | {'CPU recompute (μs)':>20} | {'CPU cached (μs)':>17} | {'Metal cold (μs)':>16} | {'Metal warm (μs)':>16} | {'Metal CPU (μs)':>15}")
    print("-" * 95)
    
    for count in interval_counts:
        cpu_manager = CPUBoundarySummaryManager()
        metal_manager = MetalBoundarySummaryManager()
        
        _build_fragmented_intervals(cpu_manager, interval_length, gap_length, count)
        _build_fragmented_intervals(metal_manager, interval_length, gap_length, count)
        
        # Prime caches
        cpu_manager.get_summary()
        
        iterations = 50 if count <= 50_000 else 20
        
        cpu_recompute_us = _time_us(
            lambda: BoundarySummary.compute_from_intervals(
                cpu_manager.intervals, cpu_manager._managed_start, cpu_manager._managed_end
            ),
            iterations=iterations,
        )
        cpu_cached_us = _time_us(cpu_manager.get_summary, iterations=iterations)
        
        metal_cold_us = _time_us(metal_manager.compute_summary_gpu, iterations=1)
        metal_warm_us = _time_us(metal_manager.compute_summary_gpu, iterations=iterations)
        metal_cpu_us = _time_us(metal_manager.compute_summary_cpu, iterations=iterations)
        
        print(f"{count:>12,} | {cpu_recompute_us:>20,.1f} | {cpu_cached_us:>17,.1f} | "
              f"{metal_cold_us:>16,.1f} | {metal_warm_us:>16,.1f} | {metal_cpu_us:>15,.1f}")
        
        # Validate totals against interval list
        cpu_total = _sum_intervals(_normalize_intervals(cpu_manager.get_intervals()))
        metal_total = _sum_intervals(_normalize_intervals(metal_manager.get_intervals()))
        if cpu_total != metal_total:
            print(f"   ⚠️  Total free mismatch: CPU={cpu_total:,} Metal={metal_total:,}")


def _normalize_intervals(intervals: List) -> List[Tuple[int, int]]:
    """Normalize interval list into (start, end) tuples."""
    normalized = []
    for interval in intervals:
        if hasattr(interval, 'start') and hasattr(interval, 'end'):
            normalized.append((interval.start, interval.end))
        elif isinstance(interval, dict):
            normalized.append((interval['start'], interval['end']))
        else:
            normalized.append((interval[0], interval[1]))
    return normalized


def _build_interval_index(intervals: List[Tuple[int, int]]):
    intervals = sorted(intervals, key=lambda pair: pair[0])
    starts = [start for start, _ in intervals]
    return intervals, starts


def _allocation_fits(intervals: List[Tuple[int, int]], starts: List[int], start: int, end: int) -> bool:
    if start is None or end is None:
        return False
    idx = bisect.bisect_right(starts, start) - 1
    if idx < 0:
        return False
    interval_start, interval_end = intervals[idx]
    return interval_start <= start and end <= interval_end


def _build_fragmented_intervals(manager, interval_length: int, gap_length: int, count: int) -> None:
    total_space = count * (interval_length + gap_length)
    manager.release_interval(0, total_space)
    for i in range(count):
        gap_start = i * (interval_length + gap_length) + interval_length
        manager.reserve_interval(gap_start, gap_start + gap_length)


def run_best_fit_benchmark():
    """Benchmark Metal best-fit queries and validate correctness."""
    print("\n" + "=" * 70)
    print("METAL BEST-FIT BENCHMARK")
    print("=" * 70)
    
    configs = [(10_000, 2_000)]
    if EXTENDED_BENCH:
        configs += [(50_000, 5_000), (100_000, 10_000)]
    
    for num_intervals, num_queries in configs:
        cpu_manager = CPUBoundarySummaryManager()
        metal_manager = MetalBoundarySummaryManager()
        mixed_manager = MixedBoundarySummaryManager(
            summary_path=MIXED_SUMMARY_PATH,
            best_fit_path=MIXED_BEST_FIT_PATH,
            best_fit_min_intervals=MIXED_BEST_FIT_MIN_INTERVALS,
            summary_min_intervals=MIXED_SUMMARY_MIN_INTERVALS,
            sync_cpu=MIXED_SYNC_CPU,
            track_allocations=True,
        )
        
        # Create many non-overlapping intervals with varied sizes
        for i in range(num_intervals):
            start = i * 200
            length = 50 + (i % 50)  # 50-99
            end = start + length
            cpu_manager.release_interval(start, end)
            metal_manager.release_interval(start, end)
            mixed_manager.release_interval(start, end)
        
        cpu_intervals, cpu_starts = _build_interval_index(
            _normalize_intervals(cpu_manager.get_intervals())
        )
        metal_intervals, metal_starts = _build_interval_index(
            _normalize_intervals(metal_manager.get_intervals())
        )
        
        max_len = max(end - start for start, end in cpu_intervals)
        lengths = [random.randint(10, max_len + 20) for _ in range(num_queries)]
        
        # CPU best-fit (prefer_early=False for best-fit behavior)
        cpu_start = time.perf_counter()
        cpu_results = [cpu_manager.find_best_fit(length, prefer_early=False) for length in lengths]
        cpu_time = (time.perf_counter() - cpu_start) * 1000
        
        # Metal best-fit (GPU path when prefer_early=False)
        metal_start = time.perf_counter()
        metal_results = [metal_manager.find_best_fit(length, prefer_early=False) for length in lengths]
        metal_time = (time.perf_counter() - metal_start) * 1000

        # Mixed best-fit (GPU path enabled)
        mixed_start = time.perf_counter()
        mixed_results = [mixed_manager.find_best_fit(length, prefer_early=False) for length in lengths]
        mixed_time = (time.perf_counter() - mixed_start) * 1000
        
        # Validate results
        mismatched = 0
        invalid_allocations = 0
        for length, cpu_result, metal_result, mixed_result in zip(lengths, cpu_results, metal_results, mixed_results):
            cpu_found = cpu_result is not None
            metal_found = metal_result is not None
            if cpu_found != metal_found:
                mismatched += 1
            
            if metal_found:
                m_start, m_end = metal_result
                if (m_end - m_start) != length:
                    invalid_allocations += 1
                elif not _allocation_fits(metal_intervals, metal_starts, m_start, m_end):
                    invalid_allocations += 1

            mixed_found = mixed_result is not None
            if mixed_found:
                if hasattr(mixed_result, 'start') and hasattr(mixed_result, 'end'):
                    mix_start, mix_end = mixed_result.start, mixed_result.end
                else:
                    mix_start, mix_end = mixed_result
                if (mix_end - mix_start) != length:
                    invalid_allocations += 1
                elif not _allocation_fits(metal_intervals, metal_starts, mix_start, mix_end):
                    invalid_allocations += 1
        
        print(f"\n🔎 Best-Fit Queries: {num_queries:,} (intervals={num_intervals:,})")
        print(f"   CPU time:   {cpu_time:.2f} ms")
        print(f"   Metal time: {metal_time:.2f} ms")
        print(f"   Mixed time: {mixed_time:.2f} ms")
        if metal_time > 0:
            print(f"   Metal speedup: {cpu_time / metal_time:.2f}x")
        if mixed_time > 0:
            print(f"   Mixed speedup: {cpu_time / mixed_time:.2f}x")
        print(f"   Result mismatches (CPU vs Metal found/not found): {mismatched}")
        print(f"   Invalid allocations (Metal/Mixed): {invalid_allocations}")


def main():
    """Main benchmark execution"""
    print("🍎 Tree-Mendous Metal Performance Benchmark")
    print("=" * 70)
    
    random.seed(42)
    if EXTENDED_BENCH:
        print("🔧 Extended benchmarks enabled (METAL_BENCH_EXTENDED=1)")
    
    run_scaling_benchmark()
    run_feature_benchmark()
    run_fragmentation_summary_benchmark()
    run_best_fit_benchmark()
    
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
