#!/usr/bin/env python3
"""
Simple Performance Benchmark for Summary Trees

Tests the core summary tree implementation without external dependencies.
"""

import sys
import time
import random
import statistics
from pathlib import Path
from tests.performance.workload import generate_realistic_workload

# Add paths for import resolution  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))

try:
    from summary import SummaryIntervalTree
    print("[OK] Summary tree loaded successfully")
except ImportError as e:
    print(f"[FAIL] Failed to load summary tree: {e}")
    sys.exit(1)


def generate_operations(num_ops: int) -> list:
    """Generate realistic interval operations"""
    return generate_realistic_workload(
        num_operations=num_ops,
        profile="scheduler",
        space_range=(0, 1_000_000),
        seed=42,
        include_data=False
    )


def benchmark_basic_operations(num_ops: int = 10_000) -> dict:
    """Benchmark basic tree operations"""
    print(f"\n🔄 Benchmarking {num_ops:,} basic operations...")
    
    tree = SummaryIntervalTree()
    tree.release_interval(0, 1_000_000)  # Initialize with large available space
    
    operations = generate_operations(num_ops)
    op_times = {'reserve': [], 'release': [], 'find': []}
    
    start_total = time.perf_counter()
    
    for op, start, end in operations:
        start_time = time.perf_counter()
        
        if op == 'reserve':
            tree.reserve_interval(start, end)
        elif op == 'release':
            tree.release_interval(start, end)
        elif op == 'find':
            try:
                tree.find_interval(start, end - start)
            except ValueError:
                pass  # No suitable interval found
        
        op_times[op].append(time.perf_counter() - start_time)
    
    total_time = time.perf_counter() - start_total
    
    # Calculate averages
    results = {
        'total_time': total_time,
        'operations_per_second': num_ops / total_time,
        'avg_times_ms': {}
    }
    
    for op, times in op_times.items():
        if times:
            results['avg_times_ms'][op] = statistics.mean(times) * 1000
    
    # Get final tree state
    stats = tree.get_availability_stats()
    results['final_stats'] = stats
    
    return results


def benchmark_summary_operations() -> dict:
    """Benchmark summary-specific operations"""
    print("\n🌟 Benchmarking summary operations...")
    
    tree = SummaryIntervalTree()
    tree.release_interval(0, 100_000)
    
    # Create some fragmentation
    for i in range(0, 50_000, 2000):
        tree.reserve_interval(i, i + 500)
    
    operations = {
        'get_availability_stats': lambda: tree.get_availability_stats(),
        'find_best_fit': lambda: tree.find_best_fit(100),
        'find_largest_available': lambda: tree.find_largest_available(),
        'get_tree_summary': lambda: tree.get_tree_summary()
    }
    
    results = {}
    
    for op_name, op_func in operations.items():
        times = []
        for _ in range(1000):  # Run each operation 1000 times
            start_time = time.perf_counter()
            op_func()
            times.append(time.perf_counter() - start_time)
        
        results[op_name] = {
            'avg_time_us': statistics.mean(times) * 1_000_000,  # microseconds
            'min_time_us': min(times) * 1_000_000,
            'max_time_us': max(times) * 1_000_000
        }
    
    return results


def benchmark_scaling() -> dict:
    """Test performance scaling with tree size"""
    print("\n[PERF] Testing performance scaling...")
    
    sizes = [1_000, 5_000, 10_000, 50_000]
    results = {}
    
    for size in sizes:
        tree = SummaryIntervalTree()
        tree.release_interval(0, size)
        
        # Create proportional fragmentation
        num_reserves = size // 100
        for i in range(num_reserves):
            start = random.randint(0, size - 100)
            tree.reserve_interval(start, start + random.randint(10, 50))
        
        # Time key operations
        start_time = time.perf_counter()
        stats = tree.get_availability_stats()
        stats_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        best_fit = tree.find_best_fit(20)
        find_time = time.perf_counter() - start_time
        
        results[size] = {
            'intervals': len(tree.get_intervals()),
            'get_stats_us': stats_time * 1_000_000,
            'find_best_fit_us': find_time * 1_000_000,
            'utilization': stats['utilization'],
            'fragmentation': stats['fragmentation']
        }
    
    return results


def main():
    """Run all benchmarks"""
    print("=== Tree-Mendous Summary Tree Performance Benchmark")
    print("=" * 60)
    
    random.seed(42)  # Reproducible results
    
    # Basic operations benchmark
    basic_results = benchmark_basic_operations(10_000)
    print(f"\n[STATS] Basic Operations Results:")
    print(f"  Total time: {basic_results['total_time']:.3f}s")
    print(f"  Operations/sec: {basic_results['operations_per_second']:,.0f}")
    print(f"  Average operation times (ms):")
    for op, time_ms in basic_results['avg_times_ms'].items():
        print(f"    {op:8}: {time_ms:.4f}")
    
    stats = basic_results['final_stats']
    print(f"  Final tree state:")
    print(f"    Utilization: {stats['utilization']:.1%}")
    print(f"    Fragmentation: {stats['fragmentation']:.1%}")
    print(f"    Free chunks: {stats['free_chunks']}")
    
    # Summary operations benchmark
    summary_results = benchmark_summary_operations()
    print(f"\n🌟 Summary Operations Results:")
    for op_name, timings in summary_results.items():
        print(f"  {op_name:20}: {timings['avg_time_us']:6.1f}µs avg ({timings['min_time_us']:4.1f}-{timings['max_time_us']:5.1f}µs)")
    
    # Scaling benchmark
    scaling_results = benchmark_scaling()
    print(f"\n[PERF] Scaling Performance:")
    print(f"  {'Size':>8} {'Intervals':>10} {'Stats(µs)':>10} {'Find(µs)':>10} {'Util%':>6} {'Frag%':>6}")
    print("  " + "-" * 58)
    for size, metrics in scaling_results.items():
        print(f"  {size:>8,} {metrics['intervals']:>10} {metrics['get_stats_us']:>10.1f} "
              f"{metrics['find_best_fit_us']:>10.1f} {metrics['utilization']*100:>5.1f} {metrics['fragmentation']*100:>5.1f}")
    
    print(f"\n[OK] Benchmark complete!")
    print(f"Summary trees provide O(1) aggregate statistics and efficient scheduling queries.")


if __name__ == "__main__":
    main()
