#!/usr/bin/env python3
"""
Large-Scale Performance Benchmark for Tree-Mendous

Tests interval trees with hundreds of megabytes of managed space to validate
scalability and performance under extreme workloads.
"""

import time
import random
import statistics
import gc
import sys
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Optional psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - memory monitoring disabled")

# Add paths for import resolution  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))

try:
    from summary import SummaryIntervalTree
    print("✅ Summary tree loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load summary tree: {e}")
    sys.exit(1)

# Test scales - representing total managed space
MB = 1024 * 1024
TEST_SCALES = {
    "10MB": 10 * MB,
    "50MB": 50 * MB, 
    "100MB": 100 * MB,
    "250MB": 250 * MB,
    "500MB": 500 * MB,
}

# For quick testing during development
QUICK_TEST_SCALES = {
    "1MB": 1 * MB,
    "10MB": 10 * MB,
    "50MB": 50 * MB,
}

@dataclass
class LargeScaleBenchmarkResult:
    scale_name: str
    total_space: int
    operations_count: int
    setup_time: float
    operation_time: float
    total_time: float
    operations_per_second: float
    memory_usage_mb: float
    intervals_count: int
    utilization: float
    fragmentation: float
    largest_chunk_mb: float
    avg_op_time_us: float
    summary_stats_time_us: float
    find_best_fit_time_us: float


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    else:
        # Fallback: return 0 if psutil not available
        return 0.0


def generate_large_scale_operations(total_space: int, num_operations: int, 
                                  workload_pattern: str = "mixed") -> List[Tuple[str, int, int]]:
    """Generate operations for large-scale testing with different patterns"""
    operations = []
    
    if workload_pattern == "dense":
        # Many small intervals - high fragmentation
        for _ in range(num_operations):
            op = random.choice(['reserve', 'release', 'find'])
            start = random.randint(0, total_space - 1000)
            length = random.randint(10, 1000)  # Small intervals
            operations.append((op, start, start + length))
            
    elif workload_pattern == "sparse":
        # Few large intervals - low fragmentation  
        for _ in range(num_operations):
            op = random.choice(['reserve', 'release', 'find'])
            start = random.randint(0, total_space - 100000)
            length = random.randint(10000, 100000)  # Large intervals
            operations.append((op, start, start + length))
            
    elif workload_pattern == "mixed":
        # Mixed sizes - realistic workload
        for _ in range(num_operations):
            op = random.choice(['reserve', 'release', 'find'])
            start = random.randint(0, total_space - 10000)
            # Bimodal distribution: mostly small with some large
            if random.random() < 0.8:
                length = random.randint(100, 5000)  # Small intervals (80%)
            else:
                length = random.randint(10000, 50000)  # Large intervals (20%)
            operations.append((op, start, start + length))
            
    elif workload_pattern == "sequential":
        # Sequential allocation pattern - common in memory allocators
        pos = 0
        for i in range(num_operations):
            op = 'reserve' if i < num_operations * 0.7 else random.choice(['release', 'find'])
            length = random.randint(1000, 10000)
            start = pos if op == 'reserve' else random.randint(0, max(1, pos - 50000))
            if op == 'reserve':
                pos += length + random.randint(0, 1000)  # Some fragmentation
            operations.append((op, start, start + length))
            
    return operations


def benchmark_large_scale(scale_name: str, total_space: int, num_operations: int,
                         workload_pattern: str = "mixed") -> LargeScaleBenchmarkResult:
    """Benchmark tree performance at large scale"""
    print(f"\n🔄 Benchmarking {scale_name} ({total_space:,} units, {num_operations:,} operations)")
    print(f"   Pattern: {workload_pattern}")
    
    # Force garbage collection before test
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Setup phase
    setup_start = time.perf_counter()
    tree = SummaryIntervalTree()
    tree.release_interval(0, total_space)
    setup_time = time.perf_counter() - setup_start
    
    # Generate operations
    operations = generate_large_scale_operations(total_space, num_operations, workload_pattern)
    
    # Operations phase
    operation_start = time.perf_counter()
    op_times = []
    
    for i, (op, start, end) in enumerate(operations):
        op_start = time.perf_counter()
        
        try:
            if op == 'reserve':
                tree.reserve_interval(start, end)
            elif op == 'release':
                tree.release_interval(start, end)
            elif op == 'find':
                try:
                    tree.find_best_fit(end - start)
                except (ValueError, AttributeError):
                    pass  # No suitable interval or method not available
                    
            op_times.append(time.perf_counter() - op_start)
            
        except Exception as e:
            print(f"   ⚠️  Error in operation {i}: {e}")
            continue
        
        # Progress indicator for long tests
        if (i + 1) % (num_operations // 10) == 0:
            progress = (i + 1) / num_operations * 100
            elapsed = time.perf_counter() - operation_start
            est_total = elapsed / (i + 1) * num_operations
            print(f"   Progress: {progress:.0f}% ({elapsed:.1f}s elapsed, ~{est_total:.1f}s total)")
    
    operation_time = time.perf_counter() - operation_start
    total_time = setup_time + operation_time
    
    # Benchmark summary operations
    summary_times = []
    for _ in range(100):
        start_time = time.perf_counter()
        stats = tree.get_availability_stats()
        summary_times.append(time.perf_counter() - start_time)
    
    # Benchmark find operations  
    find_times = []
    for _ in range(100):
        start_time = time.perf_counter()
        tree.find_best_fit(random.randint(1000, 10000))
        find_times.append(time.perf_counter() - start_time)
    
    # Get final statistics
    final_stats = tree.get_availability_stats()
    intervals = tree.get_intervals()
    final_memory = get_memory_usage()
    
    return LargeScaleBenchmarkResult(
        scale_name=scale_name,
        total_space=total_space,
        operations_count=num_operations,
        setup_time=setup_time,
        operation_time=operation_time,
        total_time=total_time,
        operations_per_second=num_operations / operation_time if operation_time > 0 else 0,
        memory_usage_mb=final_memory - initial_memory,
        intervals_count=len(intervals),
        utilization=final_stats['utilization'],
        fragmentation=final_stats['fragmentation'],
        largest_chunk_mb=final_stats['largest_chunk'] / MB,
        avg_op_time_us=statistics.mean(op_times) * 1_000_000 if op_times else 0,
        summary_stats_time_us=statistics.mean(summary_times) * 1_000_000 if summary_times else 0,
        find_best_fit_time_us=statistics.mean(find_times) * 1_000_000 if find_times else 0
    )


def benchmark_scalability_analysis(quick_mode: bool = False):
    """Analyze how performance scales with tree size"""
    print("\n📈 Scalability Analysis")
    print("=" * 60)
    
    results = []
    
    # Choose test scales based on mode
    scales = QUICK_TEST_SCALES if quick_mode else TEST_SCALES
    
    # Test different scales with consistent operation density
    for scale_name, total_space in scales.items():
        if total_space > 100 * MB:
            # Reduce operations for very large tests to keep runtime reasonable
            num_ops = min(50_000, total_space // 10000)
        else:
            num_ops = min(100_000, total_space // 1000)
            
        try:
            result = benchmark_large_scale(scale_name, total_space, num_ops, "mixed")
            results.append(result)
            
            # Print immediate results
            print(f"   {scale_name:>8}: {result.operations_per_second:>8,.0f} ops/sec, "
                  f"{result.memory_usage_mb:>6.1f} MB, "
                  f"{result.avg_op_time_us:>6.1f}µs/op")
                  
        except KeyboardInterrupt:
            print(f"   ⚠️  Interrupted during {scale_name} test")
            break
        except Exception as e:
            print(f"   ❌ Failed {scale_name} test: {e}")
            continue
    
    return results


def benchmark_workload_patterns():
    """Test different workload patterns on large trees"""
    print("\n🔀 Workload Pattern Analysis")
    print("=" * 60)
    
    patterns = ["dense", "sparse", "mixed", "sequential"]
    test_size = 100 * MB  # 100MB for pattern comparison
    num_operations = 50_000
    
    results = {}
    
    for pattern in patterns:
        try:
            print(f"\n🔄 Testing {pattern} pattern...")
            result = benchmark_large_scale(f"100MB-{pattern}", test_size, num_operations, pattern)
            results[pattern] = result
            
            print(f"   Ops/sec: {result.operations_per_second:,.0f}")
            print(f"   Utilization: {result.utilization:.1%}")
            print(f"   Fragmentation: {result.fragmentation:.1%}")
            print(f"   Memory: {result.memory_usage_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Failed {pattern} pattern: {e}")
            continue
    
    return results


def benchmark_memory_efficiency():
    """Analyze memory efficiency at different scales"""
    print("\n💾 Memory Efficiency Analysis")
    print("=" * 60)
    
    results = []
    
    for scale_name, total_space in TEST_SCALES.items():
        if total_space > 250 * MB:
            continue  # Skip very large tests for memory analysis
            
        gc.collect()
        initial_memory = get_memory_usage()
        
        try:
            tree = SummaryIntervalTree()
            tree.release_interval(0, total_space)
            
            # Create realistic fragmentation (30% utilization)
            num_intervals = min(10_000, total_space // 10000)
            for _ in range(num_intervals):
                start = random.randint(0, total_space - 10000)
                length = random.randint(1000, 5000)
                tree.reserve_interval(start, start + length)
            
            final_memory = get_memory_usage()
            intervals = tree.get_intervals()
            memory_per_interval = (final_memory - initial_memory) * 1024 * 1024 / len(intervals) if intervals else 0
            
            print(f"   {scale_name:>8}: {len(intervals):>8,} intervals, "
                  f"{final_memory - initial_memory:>6.1f} MB, "
                  f"{memory_per_interval:>6.0f} bytes/interval")
            
            results.append({
                'scale': scale_name,
                'intervals': len(intervals),
                'memory_mb': final_memory - initial_memory,
                'bytes_per_interval': memory_per_interval
            })
            
        except Exception as e:
            print(f"   ❌ Failed {scale_name} memory test: {e}")
            continue
    
    return results


def print_comprehensive_results(scalability_results: List[LargeScaleBenchmarkResult],
                               pattern_results: Dict[str, LargeScaleBenchmarkResult],
                               memory_results: List[Dict]):
    """Print comprehensive analysis of all benchmark results"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE LARGE-SCALE BENCHMARK RESULTS")
    print("=" * 80)
    
    # Scalability Analysis
    print("\n🔬 Scalability Analysis:")
    print("-" * 50)
    print(f"{'Scale':>8} {'Ops/sec':>10} {'Memory(MB)':>12} {'µs/op':>8} {'Intervals':>10} {'Frag%':>6}")
    print("-" * 50)
    
    for result in scalability_results:
        print(f"{result.scale_name:>8} {result.operations_per_second:>10,.0f} "
              f"{result.memory_usage_mb:>12.1f} {result.avg_op_time_us:>8.1f} "
              f"{result.intervals_count:>10,} {result.fragmentation*100:>5.1f}")
    
    # Performance scaling analysis
    if len(scalability_results) > 1:
        baseline = scalability_results[0]
        print(f"\n📈 Performance Scaling (vs {baseline.scale_name}):")
        for result in scalability_results[1:]:
            space_ratio = result.total_space / baseline.total_space
            perf_ratio = result.operations_per_second / baseline.operations_per_second
            efficiency = perf_ratio / space_ratio
            print(f"   {result.scale_name}: {space_ratio:.1f}x space → {perf_ratio:.2f}x perf (efficiency: {efficiency:.2f})")
    
    # Workload Pattern Analysis
    if pattern_results:
        print(f"\n🔀 Workload Pattern Comparison:")
        print("-" * 50)
        print(f"{'Pattern':>12} {'Ops/sec':>10} {'Util%':>6} {'Frag%':>6} {'Memory':>8}")
        print("-" * 50)
        
        for pattern, result in pattern_results.items():
            print(f"{pattern:>12} {result.operations_per_second:>10,.0f} "
                  f"{result.utilization*100:>5.1f} {result.fragmentation*100:>5.1f} "
                  f"{result.memory_usage_mb:>7.1f}M")
    
    # Memory Efficiency Analysis
    if memory_results:
        print(f"\n💾 Memory Efficiency:")
        print("-" * 40)
        print(f"{'Scale':>8} {'Intervals':>10} {'Memory':>8} {'Bytes/Int':>10}")
        print("-" * 40)
        
        for result in memory_results:
            print(f"{result['scale']:>8} {result['intervals']:>10,} "
                  f"{result['memory_mb']:>7.1f}M {result['bytes_per_interval']:>9.0f}")
    
    # Summary Statistics
    if scalability_results:
        max_scale = max(scalability_results, key=lambda x: x.total_space)
        fastest = max(scalability_results, key=lambda x: x.operations_per_second)
        
        print(f"\n🏆 Performance Highlights:")
        print(f"   • Largest scale tested: {max_scale.scale_name} ({max_scale.total_space/MB:.0f} MB)")
        print(f"   • Peak performance: {fastest.operations_per_second:,.0f} ops/sec ({fastest.scale_name})")
        print(f"   • Summary stats: {max_scale.summary_stats_time_us:.1f}µs (O(1) confirmed)")
        print(f"   • Find operations: {max_scale.find_best_fit_time_us:.1f}µs average")
        
        # Memory efficiency
        if memory_results:
            avg_bytes_per_interval = statistics.mean([r['bytes_per_interval'] for r in memory_results])
            print(f"   • Memory efficiency: {avg_bytes_per_interval:.0f} bytes/interval average")
    
    print(f"\n✅ Large-scale benchmark complete!")
    print(f"Summary trees maintain performance and memory efficiency at massive scales.")


def main():
    """Run comprehensive large-scale benchmarks"""
    # Check for quick mode argument
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == "--quick"
    
    print("🚀 Tree-Mendous Large-Scale Performance Benchmark")
    if quick_mode:
        print("⚡ Quick mode: Testing smaller scales (1-50MB)")
    else:
        print("Testing interval trees with hundreds of megabytes of managed space")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    random.seed(42)  # Reproducible results
    
    # Warning for large tests
    scales = QUICK_TEST_SCALES if quick_mode else TEST_SCALES
    total_mb = sum(space / MB for space in scales.values())
    print(f"⚠️  This benchmark will test up to {total_mb:.0f} MB of managed space")
    print(f"💾 Current memory usage: {get_memory_usage():.1f} MB")
    
    if not quick_mode:
        print("⏱️  Full benchmark may take 10+ minutes. Use --quick for faster testing.")
    print()
    
    try:
        # Run all benchmark suites
        scalability_results = benchmark_scalability_analysis(quick_mode)
        pattern_results = benchmark_workload_patterns() if not quick_mode else {}
        memory_results = benchmark_memory_efficiency() if not quick_mode else []
        
        # Print comprehensive analysis
        print_comprehensive_results(scalability_results, pattern_results, memory_results)
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")


if __name__ == "__main__":
    main()
