#!/usr/bin/env python3
"""Simple benchmark: Original vs Optimized C++ Implementations"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.performance.workload import generate_standard_workload

def benchmark(name, manager_class, operations, iterations=10):
    """Run benchmark and return average time."""
    times = []
    
    # Warmup
    m = manager_class()
    for op_type, start, end in operations:
        if op_type == 'release':
            m.release_interval(start, end)
        elif op_type == 'reserve':
            m.reserve_interval(start, end)
    
    # Benchmark
    for _ in range(iterations):
        manager = manager_class()
        start_time = time.perf_counter()
        
        for op_type, start, end in operations:
            if op_type == 'release':
                manager.release_interval(start, end)
            elif op_type == 'reserve':
                manager.reserve_interval(start, end)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'name': name,
        'avg_ms': avg_time * 1000,
        'min_ms': min_time * 1000,
        'max_ms': max_time * 1000,
        'ops_per_sec': len(operations) / avg_time
    }

def main():
    print("=== Tree-Mendous: Optimization Benchmark")
    print("=" * 70)
    
    # Generate workload
    print("\n📦 Generating workload...")
    operations = generate_standard_workload(num_operations=10000)
    print(f"   Generated {len(operations)} operations\n")
    
    # Test Boundary implementations
    try:
        from treemendous.cpp.boundary import IntervalManager as BoundaryOriginal
        from treemendous.cpp.boundary_optimized import IntervalManager as BoundaryOptimized
        
        print("=" * 70)
        print("BENCHMARK: C++ BOUNDARY MANAGER")
        print("=" * 70)
        
        print("\n🔄 Benchmarking original implementation...")
        original = benchmark("C++ Boundary (std::map)", BoundaryOriginal, operations)
        
        print("🔄 Benchmarking optimized implementation...")
        optimized = benchmark("C++ Boundary-Opt (std::map + micro-opts)", BoundaryOptimized, operations)
        
        print("\n[STATS] RESULTS:")
        print(f"\n  Original:")
        print(f"    Avg time:     {original['avg_ms']:8.3f} ms")
        print(f"    Min time:     {original['min_ms']:8.3f} ms")
        print(f"    Max time:     {original['max_ms']:8.3f} ms")
        print(f"    Throughput:   {original['ops_per_sec']:12,.0f} ops/sec")
        
        print(f"\n  Optimized:")
        print(f"    Avg time:     {optimized['avg_ms']:8.3f} ms")
        print(f"    Min time:     {optimized['min_ms']:8.3f} ms")
        print(f"    Max time:     {optimized['max_ms']:8.3f} ms")
        print(f"    Throughput:   {optimized['ops_per_sec']:12,.0f} ops/sec")
        
        speedup = original['avg_ms'] / optimized['avg_ms']
        improvement_pct = (1 - 1/speedup) * 100
        throughput_gain = optimized['ops_per_sec'] - original['ops_per_sec']
        
        print(f"\n  [TARGET] IMPROVEMENT:")
        print(f"    Speedup:      {speedup:.2f}x faster")
        print(f"    Time saved:   {original['avg_ms'] - optimized['avg_ms']:.3f} ms ({improvement_pct:.1f}%)")
        print(f"    Throughput:   +{throughput_gain:,.0f} ops/sec")
        
    except ImportError as e:
        print(f"\n[FAIL] Could not import Boundary implementations: {e}")
    
    # Test BoundarySummary implementations
    try:
        from treemendous.cpp.boundary_summary import BoundarySummaryManager as SummaryOriginal
        from treemendous.cpp.boundary_summary_optimized import BoundarySummaryManager as SummaryOptimized
        
        print("\n" + "=" * 70)
        print("BENCHMARK: C++ BOUNDARY SUMMARY MANAGER")
        print("=" * 70)
        
        print("\n🔄 Benchmarking original implementation...")
        original = benchmark("C++ BoundarySummary (std::map)", SummaryOriginal, operations)
        
        print("🔄 Benchmarking optimized implementation...")
        optimized = benchmark("C++ BoundarySummary-Opt (std::map + micro-opts)", SummaryOptimized, operations)
        
        print("\n[STATS] RESULTS:")
        print(f"\n  Original:")
        print(f"    Avg time:     {original['avg_ms']:8.3f} ms")
        print(f"    Min time:     {original['min_ms']:8.3f} ms")
        print(f"    Max time:     {original['max_ms']:8.3f} ms")
        print(f"    Throughput:   {original['ops_per_sec']:12,.0f} ops/sec")
        
        print(f"\n  Optimized:")
        print(f"    Avg time:     {optimized['avg_ms']:8.3f} ms")
        print(f"    Min time:     {optimized['min_ms']:8.3f} ms")
        print(f"    Max time:     {optimized['max_ms']:8.3f} ms")
        print(f"    Throughput:   {optimized['ops_per_sec']:12,.0f} ops/sec")
        
        speedup = original['avg_ms'] / optimized['avg_ms']
        improvement_pct = (1 - 1/speedup) * 100
        throughput_gain = optimized['ops_per_sec'] - original['ops_per_sec']
        
        print(f"\n  [TARGET] IMPROVEMENT:")
        print(f"    Speedup:      {speedup:.2f}x faster")
        print(f"    Time saved:   {original['avg_ms'] - optimized['avg_ms']:.3f} ms ({improvement_pct:.1f}%)")
        print(f"    Throughput:   +{throughput_gain:,.0f} ops/sec")
        
    except ImportError as e:
        print(f"\n[FAIL] Could not import BoundarySummary implementations: {e}")
    
    print("\n" + "=" * 70)
    print("[OK] Benchmark complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
