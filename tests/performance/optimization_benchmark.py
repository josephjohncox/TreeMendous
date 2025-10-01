#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized C++ Implementations

Compares performance of:
- Original C++ Boundary vs Optimized C++ Boundary
- Original C++ BoundarySummary vs Optimized C++ BoundarySummary

Shows impact of:
- boost::flat_map vs std::map
- Small vector optimization
- Vector pre-allocation
- SIMD operations
"""

import time
import sys
from typing import List, Tuple, Callable
from dataclasses import dataclass

# Import workload generation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.performance.workload import generate_standard_workload, execute_workload


@dataclass
class BenchmarkResult:
    name: str
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    ops_per_sec: float
    optimization_flags: dict


def benchmark_implementation(
    name: str,
    manager_class: type,
    operations: List[Tuple[str, int, int]],
    warmup: int = 1,
    iterations: int = 5
) -> BenchmarkResult:
    """Benchmark a single implementation with multiple iterations."""
    
    times: List[float] = []
    
    # Warmup
    for _ in range(warmup):
        manager = manager_class()
        execute_workload(manager, operations)
    
    # Actual benchmark
    for _ in range(iterations):
        manager = manager_class()
        start = time.perf_counter()
        execute_workload(manager, operations)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    ops_per_sec = len(operations) / avg_time
    
    # Try to get optimization flags
    opt_flags = {}
    try:
        import importlib
        module_name = manager_class.__module__
        module = importlib.import_module(module_name)
        if hasattr(module, 'get_optimization_info'):
            opt_flags = module.get_optimization_info()
    except:
        pass
    
    return BenchmarkResult(
        name=name,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        std_dev=std_dev,
        ops_per_sec=ops_per_sec,
        optimization_flags=opt_flags
    )


def print_comparison(baseline: BenchmarkResult, optimized: BenchmarkResult):
    """Print detailed comparison between baseline and optimized."""
    speedup = baseline.avg_time / optimized.avg_time
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: {baseline.name} vs {optimized.name}")
    print(f"{'='*70}")
    
    print(f"\n  Baseline ({baseline.name}):")
    print(f"    Avg time:     {baseline.avg_time*1000:8.3f} ms")
    print(f"    Min time:     {baseline.min_time*1000:8.3f} ms")
    print(f"    Max time:     {baseline.max_time*1000:8.3f} ms")
    print(f"    Std dev:      {baseline.std_dev*1000:8.3f} ms")
    print(f"    Throughput:   {baseline.ops_per_sec:12,.0f} ops/sec")
    
    print(f"\n  Optimized ({optimized.name}):")
    print(f"    Avg time:     {optimized.avg_time*1000:8.3f} ms")
    print(f"    Min time:     {optimized.min_time*1000:8.3f} ms")
    print(f"    Max time:     {optimized.max_time*1000:8.3f} ms")
    print(f"    Std dev:      {optimized.std_dev*1000:8.3f} ms")
    print(f"    Throughput:   {optimized.ops_per_sec:12,.0f} ops/sec")
    
    print(f"\n  [STATS] IMPROVEMENT:")
    print(f"    Speedup:      {speedup:.2f}x faster")
    print(f"    Time saved:   {(baseline.avg_time - optimized.avg_time)*1000:.3f} ms ({(1 - 1/speedup)*100:.1f}%)")
    print(f"    Throughput:   +{(optimized.ops_per_sec - baseline.ops_per_sec):,.0f} ops/sec")
    
    if optimized.optimization_flags:
        print(f"\n  [TARGET] Optimizations Enabled:")
        for flag, enabled in optimized.optimization_flags.items():
            status = "✓" if enabled else "✗"
            print(f"    {status} {flag}")


def main():
    print("=== Tree-Mendous: Original vs Optimized Benchmark")
    print("="*70)
    
    # Generate test workload
    print("\n📦 Generating workload...")
    operations = generate_standard_workload(num_operations=10000)
    print(f"   Generated {len(operations)} operations")
    
    # Check what's available
    implementations = {}
    
    try:
        from treemendous.cpp.boundary import IntervalManager as BoundaryOriginal
        implementations['boundary_original'] = BoundaryOriginal
        print("   ✓ Original C++ Boundary")
    except ImportError:
        print("   ✗ Original C++ Boundary not available")
    
    try:
        from treemendous.cpp.boundary_optimized import IntervalManager as BoundaryOptimized
        implementations['boundary_optimized'] = BoundaryOptimized
        print("   ✓ Optimized C++ Boundary")
    except ImportError:
        print("   ✗ Optimized C++ Boundary not available")
    
    try:
        from treemendous.cpp.boundary_summary import BoundarySummaryManager as SummaryOriginal
        implementations['summary_original'] = SummaryOriginal
        print("   ✓ Original C++ BoundarySummary")
    except ImportError:
        print("   ✗ Original C++ BoundarySummary not available")
    
    try:
        from treemendous.cpp.boundary_summary_optimized import BoundarySummaryManager as SummaryOptimized
        implementations['summary_optimized'] = SummaryOptimized
        print("   ✓ Optimized C++ BoundarySummary")
    except ImportError:
        print("   ✗ Optimized C++ BoundarySummary not available")
    
    if not implementations:
        print("\n[FAIL] No implementations available. Build C++ extensions first:")
        print("   uv run python setup.py build_ext --inplace")
        return 1
    
    # Benchmark Boundary implementations
    if 'boundary_original' in implementations and 'boundary_optimized' in implementations:
        print("\n" + "="*70)
        print("BENCHMARK: C++ BOUNDARY MANAGER")
        print("="*70)
        
        print("\n🔄 Benchmarking original implementation...")
        baseline = benchmark_implementation(
            "C++ Boundary (std::map)",
            implementations['boundary_original'],
            operations,
            warmup=2,
            iterations=10
        )
        
        print("🔄 Benchmarking optimized implementation...")
        optimized = benchmark_implementation(
            "C++ Boundary (boost::flat_map + optimizations)",
            implementations['boundary_optimized'],
            operations,
            warmup=2,
            iterations=10
        )
        
        print_comparison(baseline, optimized)
    
    # Benchmark BoundarySummary implementations
    if 'summary_original' in implementations and 'summary_optimized' in implementations:
        print("\n" + "="*70)
        print("BENCHMARK: C++ BOUNDARY SUMMARY MANAGER")
        print("="*70)
        
        print("\n🔄 Benchmarking original implementation...")
        baseline = benchmark_implementation(
            "C++ BoundarySummary (std::map)",
            implementations['summary_original'],
            operations,
            warmup=2,
            iterations=10
        )
        
        print("🔄 Benchmarking optimized implementation...")
        optimized = benchmark_implementation(
            "C++ BoundarySummary (boost::flat_map + optimizations)",
            implementations['summary_optimized'],
            operations,
            warmup=2,
            iterations=10
        )
        
        print_comparison(baseline, optimized)
    
    print("\n" + "="*70)
    print("[OK] Benchmark complete!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
