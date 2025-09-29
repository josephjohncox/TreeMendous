#!/usr/bin/env python3
"""
Protocol-Compliant Performance Benchmark

Benchmarks all implementations using standardized protocols and generates
comprehensive performance reports with flame graph support.
"""

import sys
import time
import random
import statistics
import cProfile
import pstats
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager
from treemendous.basic.protocols import IntervalResult
from tests.performance.workload import generate_standard_workload, execute_workload

# C++ implementations (assumed available)
from treemendous.cpp.boundary import IntervalManager as CppBoundary
from treemendous.cpp.treap import IntervalTreap as CppTreap
CPP_BOUNDARY_AVAILABLE = True
CPP_TREAP_AVAILABLE = True

try:
    from treemendous.cpp import BoundarySummaryManager as CppBoundarySummary
    CPP_BOUNDARY_SUMMARY_AVAILABLE = True
except ImportError:
    CPP_BOUNDARY_SUMMARY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Standardized benchmark result using protocols"""
    implementation: str
    total_time_ms: float
    operations_per_second: float
    avg_operation_time_us: float
    p50_time_us: float
    p95_time_us: float
    p99_time_us: float
    memory_intervals: int
    final_available_length: int


def benchmark_implementation(impl, impl_name: str, operations: List[Tuple[str, int, int]]) -> BenchmarkResult:
    """Benchmark a single implementation with detailed metrics"""
    
    # Initialize
    impl.release_interval(0, 1_000_000)
    
    # Timing for each operation
    operation_times = []
    
    start_total = time.perf_counter()
    
    for op, start, end in operations:
        start_time = time.perf_counter()
        
        try:
            if op == 'reserve':
                impl.reserve_interval(start, end)
            elif op == 'release':
                impl.release_interval(start, end)
            elif op == 'find':
                result = impl.find_interval(start, end - start)
        except (ValueError, Exception):
            pass  # Some operations may fail (e.g., no suitable interval)
        
        operation_times.append((time.perf_counter() - start_time) * 1_000_000)  # microseconds
    
    total_time = time.perf_counter() - start_total
    
    # Calculate statistics
    ops_per_second = len(operations) / total_time
    avg_time_us = statistics.mean(operation_times)
    
    # Percentiles
    sorted_times = sorted(operation_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    # Final state
    intervals = impl.get_intervals()
    total_available = impl.get_total_available_length()
    
    return BenchmarkResult(
        implementation=impl_name,
        total_time_ms=total_time * 1000,
        operations_per_second=ops_per_second,
        avg_operation_time_us=avg_time_us,
        p50_time_us=p50,
        p95_time_us=p95,
        p99_time_us=p99,
        memory_intervals=len(intervals),
        final_available_length=total_available
    )


def run_comprehensive_benchmark(num_operations: int = 10_000) -> List[BenchmarkResult]:
    """Run benchmark across all available implementations"""
    
    print(f"🔥 Running comprehensive benchmark with {num_operations:,} operations")
    print("=" * 80)
    
    # Generate unified workload - SAME for all implementations
    operations = generate_standard_workload(num_operations)
    
    # Define implementations to test
    implementations = [
        ("Python Boundary", IntervalManager),
        ("Python Summary", SummaryIntervalTree),
        ("Python Treap", lambda: IntervalTreap(random_seed=42)),
        ("Python BoundarySummary", BoundarySummaryManager),
    ]
    
    # Add C++ implementations if available
    if CPP_BOUNDARY_AVAILABLE:
        implementations.append(("C++ Boundary", CppBoundary))
    if CPP_TREAP_AVAILABLE:
        implementations.append(("C++ Treap", lambda: CppTreap(42)))
    if CPP_BOUNDARY_SUMMARY_AVAILABLE:
        implementations.append(("C++ BoundarySummary", CppBoundarySummary))
    
    results = []
    
    for impl_name, impl_class in implementations:
        print(f"\n📊 Benchmarking {impl_name}...")
        
        # Create instance
        impl = impl_class()
        
        # Run benchmark
        result = benchmark_implementation(impl, impl_name, operations)
        results.append(result)
        
        # Print quick summary
        print(f"   Total time: {result.total_time_ms:.2f}ms")
        print(f"   Ops/sec: {result.operations_per_second:,.0f}")
        print(f"   Avg time: {result.avg_operation_time_us:.2f}µs")
    
    return results


def print_benchmark_report(results: List[BenchmarkResult]) -> None:
    """Print comprehensive benchmark report"""
    
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 80)
    
    # Main performance table
    print(f"\n{'Implementation':<25} {'Total(ms)':<12} {'Ops/sec':<12} {'Avg(µs)':<10} {'P95(µs)':<10} {'P99(µs)':<10}")
    print("-" * 90)
    
    for result in results:
        print(f"{result.implementation:<25} "
              f"{result.total_time_ms:<11.2f} "
              f"{result.operations_per_second:<11,.0f} "
              f"{result.avg_operation_time_us:<9.2f} "
              f"{result.p95_time_us:<9.2f} "
              f"{result.p99_time_us:<9.2f}")
    
    # Relative performance analysis
    print(f"\n📊 Relative Performance (normalized to fastest):")
    print("-" * 60)
    
    fastest_time = min(r.total_time_ms for r in results)
    
    for result in results:
        ratio = result.total_time_ms / fastest_time
        bar = "█" * int(ratio * 20)
        print(f"  {result.implementation:<25} {ratio:5.2f}x {bar}")
    
    # Memory efficiency
    print(f"\n💾 Memory Efficiency:")
    print("-" * 60)
    print(f"{'Implementation':<25} {'Intervals':<12} {'Avg Bytes/Interval':<20}")
    print("-" * 60)
    
    for result in results:
        # Rough estimate: different implementations have different overhead
        bytes_per_interval = 32  # Base estimate
        if "Summary" in result.implementation:
            bytes_per_interval = 80
        elif "Treap" in result.implementation:
            bytes_per_interval = 48
        
        print(f"{result.implementation:<25} {result.memory_intervals:<12} ~{bytes_per_interval}")
    
    # Consistency check
    print(f"\n✅ Consistency Verification:")
    print("-" * 60)
    
    # Check that all implementations report same total available length
    available_lengths = set(r.final_available_length for r in results)
    
    if len(available_lengths) == 1:
        print(f"   ✅ All implementations report identical available length: {available_lengths.pop():,} units")
    else:
        print(f"   ⚠️  Inconsistent available lengths: {available_lengths}")
    
    # Performance recommendations
    print(f"\n🎯 Performance Recommendations:")
    print("-" * 60)
    
    fastest = min(results, key=lambda r: r.total_time_ms)
    most_efficient = min(results, key=lambda r: r.avg_operation_time_us)
    lowest_latency = min(results, key=lambda r: r.p99_time_us)
    
    print(f"   Fastest overall: {fastest.implementation}")
    print(f"   Most efficient: {most_efficient.implementation}")
    print(f"   Lowest P99 latency: {lowest_latency.implementation}")


def profile_with_flamegraph(impl_class, impl_name: str, num_operations: int = 10_000) -> str:
    """Profile implementation and prepare for flame graph generation"""
    
    print(f"\n🔥 Profiling {impl_name} for flame graph...")
    
    operations = generate_standard_workload(num_operations)
    
    # Create profiled function
    def run_benchmark():
        impl = impl_class()
        impl.release_interval(0, 1_000_000)
        
        for op, start, end in operations:
            try:
                if op == 'reserve':
                    impl.reserve_interval(start, end)
                elif op == 'release':
                    impl.release_interval(start, end)
                elif op == 'find':
                    impl.find_interval(start, end - start)
            except (ValueError, Exception):
                pass
    
    # Profile with cProfile
    output_dir = Path("performance_profiles")
    output_dir.mkdir(exist_ok=True)
    
    profile_file = str(output_dir / f"{impl_name.lower().replace(' ', '_')}.prof")
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_benchmark()
    profiler.disable()
    
    # Save profile
    profiler.dump_stats(profile_file)
    
    print(f"   ✅ Profile saved to: {profile_file}")
    
    # Print top functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print(f"   Top 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return profile_file


def try_generate_flamegraph(profile_file: str) -> bool:
    """Try to generate SVG flame graph using flameprof"""
    try:
        import subprocess
        
        svg_file = profile_file.replace('.prof', '_flame.svg')
        
        result = subprocess.run(
            ['flameprof', profile_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            with open(svg_file, 'w') as f:
                f.write(result.stdout)
            
            print(f"   🔥 Flame graph generated: {svg_file}")
            print(f"      Open with: open {svg_file}")
            return True
        else:
            return False
            
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"   ⚠️  Flame graph generation failed: {e}")
        return False


def main():
    """Run comprehensive performance benchmarking"""
    
    print("🚀 Tree-Mendous Protocol-Compliant Performance Benchmark")
    print("=" * 80)
    print()
    
    # Parse command line arguments
    num_operations = 10_000
    enable_profiling = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--profile":
            enable_profiling = True
        elif sys.argv[1].isdigit():
            num_operations = int(sys.argv[1])
    
    if len(sys.argv) > 2 and sys.argv[2] == "--profile":
        enable_profiling = True
    
    # Run main benchmark
    results = run_comprehensive_benchmark(num_operations)
    
    # Print report
    print_benchmark_report(results)
    
    # Profiling with flame graphs
    if enable_profiling:
        print("\n" + "=" * 80)
        print("🔥 DETAILED PROFILING WITH FLAME GRAPHS")
        print("=" * 80)
        
        implementations_to_profile = [
            ("Python_Boundary", IntervalManager),
            ("Python_Summary", SummaryIntervalTree),
            ("Python_Treap", lambda: IntervalTreap(random_seed=42)),
            ("Python_BoundarySummary", BoundarySummaryManager),
        ]
        
        if CPP_BOUNDARY_AVAILABLE:
            implementations_to_profile.append(("CPP_Boundary", CppBoundary))
        
        profile_files = []
        for impl_name, impl_class in implementations_to_profile:
            profile_file = profile_with_flamegraph(impl_class, impl_name, num_operations=5_000)
            profile_files.append(profile_file)
        
        # Try to generate flame graphs
        print("\n🎨 Generating flame graph visualizations...")
        
        flamegraphs_generated = False
        for profile_file in profile_files:
            if try_generate_flamegraph(profile_file):
                flamegraphs_generated = True
        
        if not flamegraphs_generated:
            print("\n📋 Flame Graph Generation:")
            print("   To generate interactive visualizations, install flameprof:")
            print("   pip install flameprof")
            print()
            print("   Then run:")
            for pf in profile_files:
                svg_file = pf.replace('.prof', '_flame.svg')
                print(f"   flameprof {pf} > {svg_file}")
            print()
            print("   Alternative: Use snakeviz for interactive exploration:")
            print("   pip install snakeviz")
            print(f"   snakeviz {profile_files[0]}")
    
    else:
        print("\n💡 Tip: Run with --profile flag for detailed profiling and flame graphs")
        print(f"   Example: python {Path(__file__).name} --profile")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
