#!/usr/bin/env python3
"""
GPU Stress Test for Tree-Mendous

Extreme stress testing of GPU implementations under sustained load.
Tests GPU limits, memory pressure, thermal throttling, and sustained throughput.

This is NOT a benchmark - it's a torture test to find GPU limits and issues.
"""

import sys
import time
import random
import platform
import statistics
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.workload import generate_workload

# GPU implementations (platform-specific)
GPU_AVAILABLE = False
GPU_TYPE = None
GPU_INFO = {}
GpuImpl = None

if platform.system() == 'Darwin':
    # macOS - Metal
    try:
        from treemendous.cpp.metal.boundary_summary_metal import (
            MetalBoundarySummaryManager as GpuImpl,
            get_metal_device_info
        )
        GPU_INFO = get_metal_device_info()
        GPU_AVAILABLE = GPU_INFO.get('available', 'false') == 'true'
        GPU_TYPE = 'Metal'
    except ImportError as e:
        print(f"❌ Metal not available: {e}")
        sys.exit(1)
else:
    # Linux/Windows - CUDA
    try:
        from treemendous.cpp.gpu.boundary_summary_gpu import (
            GPUBoundarySummaryManager as GpuImpl,
            is_gpu_available,
            get_gpu_info
        )
        GPU_AVAILABLE = is_gpu_available()
        GPU_TYPE = 'CUDA'
        if GPU_AVAILABLE:
            GPU_INFO = get_gpu_info()
    except ImportError as e:
        print(f"❌ CUDA not available: {e}")
        sys.exit(1)

if not GPU_AVAILABLE:
    print("❌ No GPU available for stress testing")
    sys.exit(1)


@dataclass
class StressTestResult:
    """Results from a stress test"""
    test_name: str
    duration_seconds: float
    total_operations: int
    operations_per_second: float
    total_summary_queries: int
    summaries_per_second: float
    avg_summary_time_us: float
    min_summary_time_us: float
    max_summary_time_us: float
    gpu_memory_mb: float
    peak_intervals: int
    throttled: bool
    errors: int


def stress_test_sustained_load(
    duration_seconds: int = 60,
    target_ops_per_sec: int = 10_000
) -> StressTestResult:
    """
    Sustained load test - run GPU at constant rate for extended period.
    Tests for thermal throttling, memory leaks, and stability.
    """
    print(f"\n🔥 SUSTAINED LOAD TEST")
    print(f"   Duration: {duration_seconds}s")
    print(f"   Target: {target_ops_per_sec:,} ops/sec")
    print("-" * 60)
    
    impl = GpuImpl()
    impl.release_interval(0, 100_000_000)
    
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    
    total_ops = 0
    summary_times = []
    errors = 0
    peak_intervals = 0
    last_report = start_time
    
    # Pre-generate operations for speed
    op_batch_size = 1000
    operations = generate_workload(
        op_batch_size,
        seed=42,
        operation_mix={'reserve': 0.50, 'release': 0.30, 'find': 0.20},
        space_range=(0, 99_900_000),
        interval_size_range=(1000, 50000)
    )
    op_index = 0
    
    while time.perf_counter() < end_time:
        # Execute operation
        op, start, end = operations[op_index]
        op_index = (op_index + 1) % op_batch_size
        
        try:
            if op == 'reserve':
                impl.reserve_interval(start, end)
            elif op == 'release':
                impl.release_interval(start, end)
            elif op == 'find':
                impl.find_interval(start, end - start)
            
            total_ops += 1
            
            # Periodic summary queries (every 100 ops)
            if total_ops % 100 == 0:
                summary_start = time.perf_counter()
                
                if hasattr(impl, 'compute_summary_gpu'):
                    stats = impl.compute_summary_gpu()
                elif hasattr(impl, 'get_availability_stats'):
                    stats = impl.get_availability_stats()
                
                summary_time = (time.perf_counter() - summary_start) * 1_000_000
                summary_times.append(summary_time)
                
                # Track peak intervals
                try:
                    intervals = impl.get_intervals()
                    peak_intervals = max(peak_intervals, len(intervals))
                except:
                    pass
            
            # Progress report every 5 seconds
            now = time.perf_counter()
            if now - last_report >= 5.0:
                elapsed = now - start_time
                current_rate = total_ops / elapsed
                remaining = end_time - now
                
                print(f"   {elapsed:>5.0f}s: {total_ops:>8,} ops ({current_rate:>8,.0f} ops/sec) "
                      f"| {len(summary_times):>5} summaries | {remaining:>4.0f}s remaining")
                last_report = now
            
        except Exception as e:
            errors += 1
            if errors < 10:  # Only print first few errors
                print(f"   ⚠️  Error: {e}")
    
    total_duration = time.perf_counter() - start_time
    
    # Get final GPU stats
    gpu_memory = 0.0
    if hasattr(impl, 'get_performance_stats'):
        try:
            perf_stats = impl.get_performance_stats()
            gpu_memory = perf_stats.gpu_memory_used / (1024 * 1024)
        except:
            pass
    
    # Check for throttling (significant slowdown over time)
    throttled = False
    if len(summary_times) > 100:
        early_avg = statistics.mean(summary_times[:50])
        late_avg = statistics.mean(summary_times[-50:])
        if late_avg > early_avg * 1.5:  # 50% slowdown
            throttled = True
    
    return StressTestResult(
        test_name="Sustained Load",
        duration_seconds=total_duration,
        total_operations=total_ops,
        operations_per_second=total_ops / total_duration,
        total_summary_queries=len(summary_times),
        summaries_per_second=len(summary_times) / total_duration,
        avg_summary_time_us=statistics.mean(summary_times) if summary_times else 0,
        min_summary_time_us=min(summary_times) if summary_times else 0,
        max_summary_time_us=max(summary_times) if summary_times else 0,
        gpu_memory_mb=gpu_memory,
        peak_intervals=peak_intervals,
        throttled=throttled,
        errors=errors
    )


def stress_test_memory_pressure(
    target_intervals: int = 1_000_000,
    num_operations: int = 500_000
) -> StressTestResult:
    """
    Memory pressure test - build massive dataset to test GPU memory limits.
    """
    print(f"\n💾 MEMORY PRESSURE TEST")
    print(f"   Target: {target_intervals:,} intervals")
    print(f"   Operations: {num_operations:,}")
    print("-" * 60)
    
    impl = GpuImpl()
    total_space = target_intervals * 1000
    impl.release_interval(0, total_space)
    
    # Generate operations that build up many intervals
    operations = generate_workload(
        num_operations,
        seed=42,
        operation_mix={'reserve': 0.70, 'release': 0.20, 'find': 0.10},  # Heavy reserve
        space_range=(0, total_space - 100000),
        interval_size_range=(100, 5000)
    )
    
    start_time = time.perf_counter()
    summary_times = []
    errors = 0
    peak_intervals = 0
    
    for i, (op, start, end) in enumerate(operations):
        try:
            if op == 'reserve':
                impl.reserve_interval(start, end)
            elif op == 'release':
                impl.release_interval(start, end)
            elif op == 'find':
                impl.find_interval(start, end - start)
            
            # Frequent summary queries to stress GPU memory transfers
            if (i + 1) % 50 == 0:
                summary_start = time.perf_counter()
                
                if hasattr(impl, 'compute_summary_gpu'):
                    stats = impl.compute_summary_gpu()
                elif hasattr(impl, 'get_availability_stats'):
                    stats = impl.get_availability_stats()
                
                summary_times.append((time.perf_counter() - summary_start) * 1_000_000)
                
                # Track peak intervals
                try:
                    intervals = impl.get_intervals()
                    current_count = len(intervals)
                    peak_intervals = max(peak_intervals, current_count)
                    
                    if (i + 1) % 10_000 == 0:
                        print(f"   {i+1:>8,} ops: {current_count:>8,} intervals "
                              f"({statistics.mean(summary_times[-100:]):.1f}µs avg summary)")
                except:
                    pass
            
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"   ⚠️  Error at op {i}: {e}")
    
    duration = time.perf_counter() - start_time
    
    # Get GPU memory usage
    gpu_memory = 0.0
    if hasattr(impl, 'get_performance_stats'):
        try:
            perf_stats = impl.get_performance_stats()
            gpu_memory = perf_stats.gpu_memory_used / (1024 * 1024)
        except:
            pass
    
    return StressTestResult(
        test_name="Memory Pressure",
        duration_seconds=duration,
        total_operations=num_operations,
        operations_per_second=num_operations / duration,
        total_summary_queries=len(summary_times),
        summaries_per_second=len(summary_times) / duration,
        avg_summary_time_us=statistics.mean(summary_times) if summary_times else 0,
        min_summary_time_us=min(summary_times) if summary_times else 0,
        max_summary_time_us=max(summary_times) if summary_times else 0,
        gpu_memory_mb=gpu_memory,
        peak_intervals=peak_intervals,
        throttled=False,
        errors=errors
    )


def stress_test_burst_load(
    num_bursts: int = 100,
    ops_per_burst: int = 10_000
) -> StressTestResult:
    """
    Burst load test - rapid bursts of operations to test GPU kernel launch overhead.
    """
    print(f"\n⚡ BURST LOAD TEST")
    print(f"   Bursts: {num_bursts}")
    print(f"   Operations per burst: {ops_per_burst:,}")
    print("-" * 60)
    
    impl = GpuImpl()
    impl.release_interval(0, 10_000_000)
    
    operations = generate_workload(
        ops_per_burst,
        seed=42,
        operation_mix={'reserve': 0.45, 'release': 0.35, 'find': 0.20},
        space_range=(0, 9_900_000),
        interval_size_range=(1000, 10000)
    )
    
    start_time = time.perf_counter()
    total_ops = 0
    summary_times = []
    errors = 0
    peak_intervals = 0
    
    for burst_num in range(num_bursts):
        # Execute burst
        for op, start, end in operations:
            try:
                if op == 'reserve':
                    impl.reserve_interval(start, end)
                elif op == 'release':
                    impl.release_interval(start, end)
                elif op == 'find':
                    impl.find_interval(start, end - start)
                total_ops += 1
            except Exception as e:
                errors += 1
        
        # Summary query after each burst
        summary_start = time.perf_counter()
        
        try:
            if hasattr(impl, 'compute_summary_gpu'):
                stats = impl.compute_summary_gpu()
            elif hasattr(impl, 'get_availability_stats'):
                stats = impl.get_availability_stats()
            
            summary_times.append((time.perf_counter() - summary_start) * 1_000_000)
            
            # Track intervals
            intervals = impl.get_intervals()
            peak_intervals = max(peak_intervals, len(intervals))
            
            if (burst_num + 1) % 10 == 0:
                avg_summary = statistics.mean(summary_times[-10:])
                print(f"   Burst {burst_num+1:>3}/{num_bursts}: {len(intervals):>6,} intervals "
                      f"({avg_summary:.1f}µs summary)")
        except Exception as e:
            errors += 1
    
    duration = time.perf_counter() - start_time
    
    # Get GPU memory usage
    gpu_memory = 0.0
    if hasattr(impl, 'get_performance_stats'):
        try:
            perf_stats = impl.get_performance_stats()
            gpu_memory = perf_stats.gpu_memory_used / (1024 * 1024)
        except:
            pass
    
    return StressTestResult(
        test_name="Burst Load",
        duration_seconds=duration,
        total_operations=total_ops,
        operations_per_second=total_ops / duration,
        total_summary_queries=len(summary_times),
        summaries_per_second=len(summary_times) / duration,
        avg_summary_time_us=statistics.mean(summary_times) if summary_times else 0,
        min_summary_time_us=min(summary_times) if summary_times else 0,
        max_summary_time_us=max(summary_times) if summary_times else 0,
        gpu_memory_mb=gpu_memory,
        peak_intervals=peak_intervals,
        throttled=False,
        errors=errors
    )


def print_stress_test_report(results: List[StressTestResult]) -> None:
    """Print comprehensive stress test report"""
    
    print("\n" + "=" * 80)
    print(f"🔥 GPU STRESS TEST REPORT - {GPU_TYPE}")
    print("=" * 80)
    
    print(f"\nGPU Info:")
    print(f"   Device: {GPU_INFO.get('device_name', 'Unknown')}")
    if 'total_memory_gb' in GPU_INFO:
        print(f"   Memory: {GPU_INFO['total_memory_gb']:.1f} GB")
    if 'compute_capability' in GPU_INFO:
        print(f"   Compute: {GPU_INFO['compute_capability']}")
    
    print(f"\n{'Test':<20} {'Duration':<10} {'Ops':<12} {'Ops/sec':<12} {'Peak Int':<10} {'GPU MB':<8} {'Status':<10}")
    print("-" * 100)
    
    for result in results:
        status = "⚠️ THROTTLED" if result.throttled else "✅ OK"
        if result.errors > 0:
            status = f"❌ {result.errors} ERR"
        
        print(f"{result.test_name:<20} "
              f"{result.duration_seconds:<9.1f}s "
              f"{result.total_operations:<11,} "
              f"{result.operations_per_second:<11,.0f} "
              f"{result.peak_intervals:<9,} "
              f"{result.gpu_memory_mb:<7.2f} "
              f"{status:<10}")
    
    print(f"\n{'Test':<20} {'Summaries':<12} {'Sum/sec':<12} {'Avg(µs)':<10} {'Min(µs)':<10} {'Max(µs)':<10}")
    print("-" * 100)
    
    for result in results:
        print(f"{result.test_name:<20} "
              f"{result.total_summary_queries:<11,} "
              f"{result.summaries_per_second:<11,.0f} "
              f"{result.avg_summary_time_us:<9.1f} "
              f"{result.min_summary_time_us:<9.1f} "
              f"{result.max_summary_time_us:<9.1f}")
    
    # Overall assessment
    print("\n📊 Assessment:")
    
    total_errors = sum(r.errors for r in results)
    if total_errors == 0:
        print("   ✅ No errors detected")
    else:
        print(f"   ⚠️  Total errors: {total_errors}")
    
    throttled_tests = [r for r in results if r.throttled]
    if throttled_tests:
        print(f"   ⚠️  Thermal throttling detected in: {', '.join(r.test_name for r in throttled_tests)}")
    else:
        print("   ✅ No thermal throttling detected")
    
    max_memory = max(r.gpu_memory_mb for r in results)
    print(f"   💾 Peak GPU memory: {max_memory:.2f} MB")
    
    max_intervals = max(r.peak_intervals for r in results)
    print(f"   📊 Peak interval count: {max_intervals:,}")
    
    avg_ops_per_sec = statistics.mean([r.operations_per_second for r in results])
    print(f"   ⚡ Average throughput: {avg_ops_per_sec:,.0f} ops/sec")
    
    avg_summary_time = statistics.mean([r.avg_summary_time_us for r in results])
    print(f"   🎯 Average summary time: {avg_summary_time:.1f} µs")


def stress_test_batch_operations(
    num_batches: int = 100,
    batch_size: int = 1000
) -> StressTestResult:
    """
    Batch operations test - GPU's strength (amortized Python overhead).
    Tests GPU with bulk operations where it truly excels.
    """
    print(f"\n📦 BATCH OPERATIONS TEST")
    print(f"   Batches: {num_batches}")
    print(f"   Operations per batch: {batch_size:,}")
    print(f"   Total operations: {num_batches * batch_size:,}")
    print("-" * 60)
    
    impl = GpuImpl()
    impl.release_interval(0, 100_000_000)
    
    start_time = time.perf_counter()
    total_ops = 0
    summary_times = []
    errors = 0
    peak_intervals = 0
    
    for batch_num in range(num_batches):
        # Generate batch
        batch_reserves = []
        batch_releases = []
        
        for _ in range(batch_size):
            op = random.choice(['reserve', 'release'])
            start = random.randint(0, 99_900_000)
            length = random.randint(1000, 50000)
            
            if op == 'reserve':
                batch_reserves.append((start, start + length))
            else:
                batch_releases.append((start, start + length))
        
        # Execute batch operations (GPU-optimized)
        try:
            if hasattr(impl, 'batch_reserve'):
                if batch_reserves:
                    impl.batch_reserve(batch_reserves)
                if batch_releases:
                    impl.batch_release(batch_releases)
            else:
                # Fallback to individual ops
                for start, end in batch_reserves:
                    impl.reserve_interval(start, end)
                for start, end in batch_releases:
                    impl.release_interval(start, end)
            
            total_ops += batch_size
            
            # Summary after each batch
            summary_start = time.perf_counter()
            
            if hasattr(impl, 'compute_summary_gpu'):
                stats = impl.compute_summary_gpu()
            elif hasattr(impl, 'get_availability_stats'):
                stats = impl.get_availability_stats()
            
            summary_times.append((time.perf_counter() - summary_start) * 1_000_000)
            
            # Track intervals
            try:
                intervals = impl.get_intervals()
                peak_intervals = max(peak_intervals, len(intervals))
            except:
                pass
            
            if (batch_num + 1) % 10 == 0:
                avg_summary = statistics.mean(summary_times[-10:])
                elapsed = time.perf_counter() - start_time
                current_rate = total_ops / elapsed
                print(f"   Batch {batch_num+1:>3}/{num_batches}: {peak_intervals:>6,} intervals | "
                      f"{current_rate:>8,.0f} ops/sec | {avg_summary:.1f}µs summary")
        
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"   ⚠️  Error in batch {batch_num}: {e}")
    
    duration = time.perf_counter() - start_time
    
    # Get GPU memory usage
    gpu_memory = 0.0
    if hasattr(impl, 'get_performance_stats'):
        try:
            perf_stats = impl.get_performance_stats()
            gpu_memory = perf_stats.gpu_memory_used / (1024 * 1024)
        except:
            pass
    
    return StressTestResult(
        test_name="Batch Operations",
        duration_seconds=duration,
        total_operations=total_ops,
        operations_per_second=total_ops / duration,
        total_summary_queries=len(summary_times),
        summaries_per_second=len(summary_times) / duration,
        avg_summary_time_us=statistics.mean(summary_times) if summary_times else 0,
        min_summary_time_us=min(summary_times) if summary_times else 0,
        max_summary_time_us=max(summary_times) if summary_times else 0,
        gpu_memory_mb=gpu_memory,
        peak_intervals=peak_intervals,
        throttled=False,
        errors=errors
    )


def main():
    """Run GPU stress tests"""
    print("🔥 Tree-Mendous GPU Stress Test Suite")
    print("=" * 80)
    print(f"\nGPU Type: {GPU_TYPE}")
    print(f"Device: {GPU_INFO.get('device_name', 'Unknown')}")
    print("\n⚠️  WARNING: This is a stress test - GPU will run at high load!")
    print()
    
    # Set random seed
    random.seed(42)
    
    # Run stress tests
    results = []
    
    # Test 1: Batch operations (GPU's strength)
    print("\n" + "=" * 80)
    print("TEST 1/4: Batch Operations (100 batches × 1K ops) - GPU OPTIMIZED")
    print("=" * 80)
    result = stress_test_batch_operations(num_batches=100, batch_size=1000)
    results.append(result)
    print(f"\n✅ Completed: {result.total_operations:,} ops in {result.duration_seconds:.1f}s")
    print(f"   Throughput: {result.operations_per_second:,.0f} ops/sec")
    
    # Test 2: Sustained load (30 seconds for quick test, 300 for full stress)
    print("\n" + "=" * 80)
    print("TEST 2/4: Sustained Load (30s at 10K ops/sec)")
    print("=" * 80)
    result = stress_test_sustained_load(duration_seconds=30, target_ops_per_sec=10_000)
    results.append(result)
    print(f"\n✅ Completed: {result.total_operations:,} ops in {result.duration_seconds:.1f}s")
    
    # Test 3: Memory pressure
    print("\n" + "=" * 80)
    print("TEST 3/4: Memory Pressure (build to 100K intervals)")
    print("=" * 80)
    result = stress_test_memory_pressure(target_intervals=100_000, num_operations=200_000)
    results.append(result)
    print(f"\n✅ Completed: Peak {result.peak_intervals:,} intervals, {result.gpu_memory_mb:.2f} MB GPU memory")
    
    # Test 4: Burst load
    print("\n" + "=" * 80)
    print("TEST 4/4: Burst Load (50 bursts × 5K ops)")
    print("=" * 80)
    result = stress_test_burst_load(num_bursts=50, ops_per_burst=5_000)
    results.append(result)
    print(f"\n✅ Completed: {result.total_summary_queries:,} summary bursts")
    
    # Print comprehensive report
    print_stress_test_report(results)
    
    print("\n" + "=" * 80)
    print("✅ GPU stress test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

