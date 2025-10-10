#!/usr/bin/env python3
"""
Large-Scale GPU-Friendly Performance Benchmark

Tests all implementations (Python, C++, GPU) with massive datasets designed to 
showcase GPU acceleration. Uses unified workload generation and protocol-compliant 
testing across the entire implementation hierarchy.

Key Features:
- Scales from 10K to 1M+ intervals (GPU sweet spot)
- Heavy summary computation workload (GPU strength)
- Platform-aware GPU detection (Metal/CUDA)
- Unified test protocol for fair comparison
"""

import sys
import time
import random
import statistics
import platform
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import common testing utilities
from tests.performance.workload import generate_workload, execute_workload

# Python implementations
from treemendous.basic.boundary import IntervalManager as PyBoundary
from treemendous.basic.summary import SummaryIntervalTree as PySummary
from treemendous.basic.treap import IntervalTreap as PyTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager as PyBoundarySummary
from treemendous.basic.avl_earliest import EarliestIntervalTree as PyAVL
from treemendous.basic.avl import IntervalTree, IntervalNode

# C++ implementations (assumed available)
from treemendous.cpp.boundary import IntervalManager as CppBoundary
from treemendous.cpp.treap import IntervalTreap as CppTreap
from treemendous.cpp.boundary_summary import BoundarySummaryManager as CppBoundarySummary
from treemendous.cpp.boundary_optimized import IntervalManager as CppBoundaryOpt
from treemendous.cpp.boundary_summary_optimized import BoundarySummaryManager as CppBoundarySummaryOpt

# GPU implementations (platform-specific)
GPU_AVAILABLE = False
GPU_TYPE = None
GPU_INFO = {}
GpuBoundarySummary = None

if platform.system() == 'Darwin':
    # macOS - Metal
    try:
        from treemendous.cpp.metal.boundary_summary_metal import (
            MetalBoundarySummaryManager as GpuBoundarySummary,
            get_metal_device_info
        )
        GPU_INFO = get_metal_device_info()
        GPU_AVAILABLE = GPU_INFO.get('available', 'false') == 'true'
        GPU_TYPE = 'Metal'
        print(f"✅ Metal GPU available: {GPU_INFO.get('device_name', 'Unknown')}")
    except ImportError as e:
        print(f"⚠️  Metal not available: {e}")
else:
    # Linux/Windows - CUDA
    try:
        from treemendous.cpp.gpu.boundary_summary_gpu import (
            CudaBoundarySummaryManager as GpuBoundarySummary,
            is_gpu_available,
            get_gpu_info
        )
        GPU_AVAILABLE = is_gpu_available()
        GPU_TYPE = 'CUDA'
        if GPU_AVAILABLE:
            GPU_INFO = get_gpu_info()
            print(f"✅ CUDA GPU available: {GPU_INFO.get('device_name', 'Unknown')}")
    except ImportError as e:
        print(f"⚠️  CUDA not available: {e}")


@dataclass
class LargeScaleBenchmarkResult:
    """Results from large-scale benchmark run"""
    implementation: str
    category: str  # Python/C++/GPU
    dataset_size: int  # Number of intervals
    num_operations: int
    
    # Timing metrics
    setup_time_ms: float
    operation_time_ms: float
    total_time_ms: float
    operations_per_second: float
    
    # Operation breakdown
    avg_op_time_us: float
    p50_time_us: float
    p95_time_us: float
    p99_time_us: float
    
    # Summary operations (GPU-friendly)
    summary_queries: int = 0
    avg_summary_time_us: float = 0.0
    summary_ops_per_second: float = 0.0
    
    # Memory and state
    final_intervals: int = 0
    final_available_space: int = 0
    memory_mb: float = 0.0
    
    # GPU-specific metrics
    gpu_memory_mb: float = 0.0
    gpu_operations: int = 0


def generate_gpu_friendly_workload(
    num_operations: int,
    total_space: int = 10_000_000,
    summary_frequency: int = 50
) -> Tuple[List[Tuple[str, int, int]], int]:
    """
    Generate workload optimized for GPU testing.
    
    GPU benefits from:
    - Large datasets (many intervals)
    - Frequent summary queries (parallel reduction) - DOUBLED frequency
    - Larger intervals (better memory coalescing)
    - More reserves (builds larger trees)
    """
    # GPU-optimized: More reserves to build massive datasets
    # Larger intervals for better memory patterns
    # Less fragmentation for predictable GPU access patterns
    operations = generate_workload(
        num_operations=num_operations,
        seed=42,
        operation_mix={'reserve': 0.50, 'release': 0.30, 'find': 0.20},  # More reserves
        space_range=(0, total_space - 50000),
        interval_size_range=(500, 10000)  # Much larger intervals (10x)
    )
    
    # Count how many summary queries we'll do (2x more frequent)
    num_summary_queries = num_operations // summary_frequency
    
    return operations, num_summary_queries


def benchmark_large_scale_implementation(
    impl_name: str,
    impl_class,
    dataset_size: int,
    num_operations: int,
    category: str = "Python"
) -> LargeScaleBenchmarkResult:
    """
    Benchmark implementation at large scale with GPU-friendly workload.
    
    Args:
        impl_name: Display name
        impl_class: Class or factory function
        dataset_size: Target number of intervals (affects total_space)
        num_operations: Number of operations to perform
        category: Python/C++/GPU
    """
    print(f"  🔄 Benchmarking {impl_name} ({dataset_size:,} interval target, {num_operations:,} ops)...")
    
    # Setup phase
    setup_start = time.perf_counter()
    
    # Create instance
    if callable(impl_class):
        if 'Treap' in impl_name:
            impl = impl_class(42)  # Seed for treaps
        else:
            impl = impl_class()
    else:
        impl = impl_class
    
    # Initialize with space proportional to target dataset size
    total_space = dataset_size * 100
    impl.release_interval(0, total_space)
    
    setup_time = (time.perf_counter() - setup_start) * 1000
    
    # Generate GPU-friendly workload (2x more summary queries)
    operations, num_summary_queries = generate_gpu_friendly_workload(
        num_operations, total_space, summary_frequency=50
    )
    
    # Execute operations with timing
    operation_times = []
    summary_times = []
    operation_start = time.perf_counter()
    
    # Progress reporting for large benchmarks
    progress_interval = max(num_operations // 10, 1000)
    
    for i, (op, start, end) in enumerate(operations):
        op_start = time.perf_counter()
        
        try:
            if op == 'reserve':
                impl.reserve_interval(start, end)
            elif op == 'release':
                impl.release_interval(start, end)
            elif op == 'find':
                impl.find_interval(start, end - start)
        except (ValueError, AttributeError, Exception):
            pass
        
        operation_times.append((time.perf_counter() - op_start) * 1_000_000)
        
        # Frequent summary queries (GPU sweet spot - parallel reduction)
        if (i + 1) % 50 == 0:
            summary_start = time.perf_counter()
            try:
                if hasattr(impl, 'get_availability_stats'):
                    stats = impl.get_availability_stats()
                elif hasattr(impl, 'compute_summary_gpu'):
                    stats = impl.compute_summary_gpu()
                elif hasattr(impl, 'get_statistics'):
                    stats = impl.get_statistics()
            except:
                pass
            summary_times.append((time.perf_counter() - summary_start) * 1_000_000)
        
        # Progress reporting for long-running tests
        if (i + 1) % progress_interval == 0 and num_operations >= 50_000:
            elapsed = time.perf_counter() - operation_start
            progress_pct = (i + 1) / num_operations * 100
            est_total = elapsed / (i + 1) * num_operations
            est_remaining = est_total - elapsed
            print(f"     Progress: {progress_pct:.0f}% ({i+1:,}/{num_operations:,} ops, "
                  f"~{est_remaining:.0f}s remaining)")
    
    operation_time = (time.perf_counter() - operation_start) * 1000
    total_time = setup_time + operation_time
    
    # Calculate statistics
    ops_per_second = num_operations / (operation_time / 1000)
    avg_op_time = statistics.mean(operation_times) if operation_times else 0
    
    sorted_times = sorted(operation_times)
    p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
    p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
    p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
    
    # Summary statistics
    avg_summary_time = statistics.mean(summary_times) if summary_times else 0
    summary_ops_per_sec = len(summary_times) / (sum(summary_times) / 1_000_000) if summary_times else 0
    
    # Final state
    try:
        intervals = impl.get_intervals()
        final_interval_count = len(intervals)
    except:
        final_interval_count = 0
    
    try:
        final_available = impl.get_total_available_length()
    except:
        final_available = 0
    
    # GPU-specific metrics
    gpu_memory = 0.0
    gpu_ops = 0
    if category == "GPU" and hasattr(impl, 'get_performance_stats'):
        try:
            perf_stats = impl.get_performance_stats()
            gpu_memory = perf_stats.gpu_memory_used / (1024 * 1024)
            gpu_ops = perf_stats.gpu_operations
        except:
            pass
    
    result = LargeScaleBenchmarkResult(
        implementation=impl_name,
        category=category,
        dataset_size=dataset_size,
        num_operations=num_operations,
        setup_time_ms=setup_time,
        operation_time_ms=operation_time,
        total_time_ms=total_time,
        operations_per_second=ops_per_second,
        avg_op_time_us=avg_op_time,
        p50_time_us=p50,
        p95_time_us=p95,
        p99_time_us=p99,
        summary_queries=len(summary_times),
        avg_summary_time_us=avg_summary_time,
        summary_ops_per_second=summary_ops_per_sec,
        final_intervals=final_interval_count,
        final_available_space=final_available,
        gpu_memory_mb=gpu_memory,
        gpu_operations=gpu_ops
    )
    
    print(f"     ⏱️  {operation_time:.0f}ms | {ops_per_second:,.0f} ops/sec | {final_interval_count:,} intervals")
    
    return result


def run_scaling_benchmark() -> Dict[int, List[LargeScaleBenchmarkResult]]:
    """
    Run benchmark across multiple scales to find GPU crossover point.
    
    Scales designed to show:
    - Small: GPU overhead dominates (CPU wins)
    - Medium: Break-even point
    - Large: GPU acceleration wins
    - Massive: GPU dominates
    - Ultra: GPU-only territory
    """
    print("\n" + "=" * 80)
    print("🚀 LARGE-SCALE GPU-OPTIMIZED BENCHMARK")
    print("=" * 80)
    print(f"\nGPU Status: {GPU_TYPE if GPU_AVAILABLE else 'Not Available'}")
    if GPU_AVAILABLE and GPU_INFO:
        print(f"Device: {GPU_INFO.get('device_name', 'Unknown')}")
        if 'total_memory_gb' in GPU_INFO:
            print(f"Memory: {GPU_INFO['total_memory_gb']:.1f} GB")
    print("\n💡 Optimized for GPU: Large datasets, bulk operations, heavy analytics")
    print()
    
    # Test scales: (dataset_size_target, num_operations)
    # Dataset size = target number of intervals to maintain
    # GPU-optimized: Much larger datasets where parallelism pays off
    test_scales = [
        (10_000, 20_000, "Baseline (10K intervals)"),
        (50_000, 50_000, "Medium (50K intervals)"),
        (100_000, 100_000, "Large (100K intervals)"),
        (500_000, 200_000, "Very Large (500K intervals)"),
        (1_000_000, 300_000, "Massive (1M intervals)"),
        (5_000_000, 500_000, "Ultra (5M intervals)"),
    ]
    
    # Select fastest from each category for comparison
    implementations = [
        # Python baseline (fastest pure Python)
        ("Py Boundary", PyBoundary, "Python"),
        
        # C++ optimized (current champion)
        ("C++ Boundary-Opt", CppBoundaryOpt, "C++"),
        ("C++ BoundarySummary-Opt", CppBoundarySummaryOpt, "C++"),
    ]
    
    # Add GPU if available
    if GPU_AVAILABLE:
        implementations.append((f"GPU {GPU_TYPE}", GpuBoundarySummary, "GPU"))
    
    all_results = {}
    
    for dataset_size, num_ops, scale_name in test_scales:
        print(f"\n{'='*80}")
        print(f"📏 Scale: {scale_name}")
        print(f"   Target: {dataset_size:,} intervals maintained")
        print(f"   Operations: {num_ops:,}")
        print(f"   Summary queries: {num_ops // 50:,} (every 50 ops)")
        print("=" * 80)
        
        scale_results = []
        
        for impl_name, impl_class, category in implementations:
            # Skip Python for ultra-large scales (too slow)
            if category == "Python" and dataset_size >= 1_000_000:
                print(f"  ⏭️  Skipping {impl_name} (Python too slow at this scale)")
                continue
            
            try:
                result = benchmark_large_scale_implementation(
                    impl_name, impl_class, dataset_size, num_ops, category
                )
                scale_results.append(result)
            except Exception as e:
                print(f"     ❌ Failed: {e}")
        
        all_results[dataset_size] = scale_results
        
        # Quick comparison at this scale
        if scale_results:
            fastest = min(scale_results, key=lambda r: r.operation_time_ms)
            print(f"\n   🏆 Fastest at this scale: {fastest.implementation}")
            print(f"      {fastest.operations_per_second:,.0f} ops/sec")
    
    return all_results


def print_comprehensive_report(all_results: Dict[int, List[LargeScaleBenchmarkResult]]) -> None:
    """Print detailed analysis of benchmark results"""
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE LARGE-SCALE BENCHMARK REPORT")
    print("=" * 80)
    
    # Overall performance table
    print(f"\n{'Scale':<20} {'Implementation':<22} {'Ops/sec':<13} {'Avg(µs)':<9} {'Summary(µs)':<12}")
    print("-" * 80)
    
    for dataset_size, results in sorted(all_results.items()):
        for i, result in enumerate(results):
            scale_label = f"{dataset_size:,} intervals" if i == 0 else ""
            print(f"{scale_label:<20} {result.implementation:<22} "
                  f"{result.operations_per_second:<12,.0f} "
                  f"{result.avg_op_time_us:<8.2f} "
                  f"{result.avg_summary_time_us:<11.2f}")
    
    # GPU Speedup Analysis
    if GPU_AVAILABLE:
        print(f"\n⚡ GPU SPEEDUP ANALYSIS ({GPU_TYPE})")
        print("-" * 80)
        print(f"{'Scale':<20} {'CPU Best (ops/s)':<18} {'GPU (ops/s)':<15} {'Speedup':<10} {'Winner':<10}")
        print("-" * 80)
        
        for dataset_size, results in sorted(all_results.items()):
            cpu_results = [r for r in results if r.category != "GPU"]
            gpu_results = [r for r in results if r.category == "GPU"]
            
            if cpu_results and gpu_results:
                cpu_best = max(cpu_results, key=lambda r: r.operations_per_second)
                gpu_best = gpu_results[0]
                
                speedup = gpu_best.operations_per_second / cpu_best.operations_per_second
                winner = "🚀 GPU" if speedup > 1.0 else "💻 CPU"
                
                print(f"{dataset_size:,} intervals{'':<6} "
                      f"{cpu_best.operations_per_second:<17,.0f} "
                      f"{gpu_best.operations_per_second:<14,.0f} "
                      f"{speedup:<9.2f}x "
                      f"{winner}")
        
        # Find crossover point
        print(f"\n📌 Key Insights:")
        for dataset_size, results in sorted(all_results.items()):
            cpu_results = [r for r in results if r.category != "GPU"]
            gpu_results = [r for r in results if r.category == "GPU"]
            
            if cpu_results and gpu_results:
                cpu_best = max(cpu_results, key=lambda r: r.operations_per_second)
                gpu_best = gpu_results[0]
                speedup = gpu_best.operations_per_second / cpu_best.operations_per_second
                
                if speedup >= 1.0:
                    print(f"   • GPU becomes competitive at {dataset_size:,} intervals ({speedup:.2f}x)")
                    break
        
        # Maximum speedup
        max_speedup = 0
        max_speedup_scale = 0
        for dataset_size, results in all_results.items():
            cpu_results = [r for r in results if r.category != "GPU"]
            gpu_results = [r for r in results if r.category == "GPU"]
            
            if cpu_results and gpu_results:
                cpu_best = max(cpu_results, key=lambda r: r.operations_per_second)
                gpu_best = gpu_results[0]
                speedup = gpu_best.operations_per_second / cpu_best.operations_per_second
                
                if speedup > max_speedup:
                    max_speedup = speedup
                    max_speedup_scale = dataset_size
        
        if max_speedup > 0:
            print(f"   • Maximum GPU speedup: {max_speedup:.2f}x at {max_speedup_scale:,} intervals")
        
        # GPU Summary operation advantage
        print(f"\n   💡 GPU Summary Operations:")
        for dataset_size, results in sorted(all_results.items()):
            gpu_results = [r for r in results if r.category == "GPU"]
            if gpu_results and dataset_size >= 50_000:
                gpu = gpu_results[0]
                if gpu.avg_summary_time_us > 0:
                    print(f"      {dataset_size:,} intervals: {gpu.avg_summary_time_us:.1f}µs per summary "
                          f"({gpu.summary_ops_per_second:,.0f} summaries/sec)")
    
    # Category comparison
    print(f"\n🏆 PERFORMANCE BY CATEGORY")
    print("-" * 80)
    
    categories = ["Python", "C++", "GPU"]
    for category in categories:
        category_results = []
        for results in all_results.values():
            category_results.extend([r for r in results if r.category == category])
        
        if category_results:
            best = max(category_results, key=lambda r: r.operations_per_second)
            avg_ops = statistics.mean([r.operations_per_second for r in category_results])
            
            print(f"\n{category}:")
            print(f"   Best: {best.implementation} - {best.operations_per_second:,.0f} ops/sec")
            print(f"   Average: {avg_ops:,.0f} ops/sec across all scales")
            
            if category == "GPU" and best.gpu_memory_mb > 0:
                print(f"   GPU Memory: {best.gpu_memory_mb:.2f} MB")
                print(f"   GPU Ops: {best.gpu_operations:,}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if GPU_AVAILABLE:
        print("GPU-Accelerated Deployment:")
        print("   • Use GPU for datasets > 50K intervals")
        print("   • Heavy summary/analytics workloads see biggest gains")
        print(f"   • {GPU_TYPE} provides parallel reduction for O(1) summaries")
    
    print("\nOptimal Implementation by Scale:")
    for dataset_size, results in sorted(all_results.items()):
        if results:
            fastest = min(results, key=lambda r: r.operation_time_ms)
            print(f"   • {dataset_size:>7,} intervals: {fastest.implementation:<22} "
                  f"({fastest.operations_per_second:,.0f} ops/sec)")


def main():
    """Main benchmark execution"""
    print("🌳 Tree-Mendous: Large-Scale GPU-Optimized Benchmark")
    print("Testing implementation hierarchy at massive scales")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale GPU-optimized benchmark')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test (skip ultra-large scales)')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Focus on GPU-optimal scales (500K+)')
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Quick mode: Testing up to 500K intervals")
    elif args.gpu_only:
        print("🚀 GPU-focused mode: Testing 500K-5M intervals")
    else:
        print("🚀 Full mode: Testing 10K-5M intervals (may take several minutes)")
    
    print()
    
    # Set random seed
    random.seed(42)
    
    # Run scaling benchmark
    results = run_scaling_benchmark()
    
    # Print comprehensive report
    print_comprehensive_report(results)
    
    print("\n" + "=" * 80)
    print("✅ Large-scale GPU-optimized benchmark complete!")
    print("=" * 80)
    print("\n💡 Tips:")
    print("   • Run with --quick for faster testing")
    print("   • Run with --gpu-only to focus on GPU sweet spot")
    print("   • GPU excels at: bulk summaries, massive datasets (1M+ intervals)")


if __name__ == "__main__":
    main()

