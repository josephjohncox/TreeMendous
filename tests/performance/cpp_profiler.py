#!/usr/bin/env python3
"""
C++ Performance Profiler

Profiles C++ extensions showing native code execution.
Requires: pip install py-spy (or uv add --dev py-spy)
"""

import sys
import random
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all implementations
from treemendous.basic.boundary import IntervalManager as PyBoundary
from treemendous.basic.treap import IntervalTreap as PyTreap
from tests.performance.workload import generate_standard_workload, execute_workload

# C++ implementations
from treemendous.cpp.boundary import IntervalManager as CppBoundary
from treemendous.cpp.treap import IntervalTreap as CppTreap
CPP_AVAILABLE = True


def run_cpp_intensive_workload(iterations: int = 50_000):
    """Run intensive C++ workload for profiling using unified workload"""
    
    print(f"🔥 Running C++ intensive workload ({iterations:,} operations)...")
    
    # Generate unified workload
    operations = generate_standard_workload(iterations)
    
    # Test multiple C++ implementations
    implementations = [
        ("C++ Boundary", CppBoundary()),
        ("C++ Treap", CppTreap(42)),
    ]
    
    # Try to add BoundarySummary
    try:
        from treemendous.cpp import BoundarySummaryManager as CppBoundarySummary
        implementations.append(("C++ BoundarySummary", CppBoundarySummary()))
    except ImportError:
        pass
    
    for impl_name, manager in implementations:
        print(f"\n[STATS] {impl_name}:")
        
        start_time = time.perf_counter()
        execute_workload(manager, operations)
        duration = time.perf_counter() - start_time
        
        print(f"   Completed {iterations:,} operations in {duration:.3f}s")
        print(f"   Throughput: {iterations/duration:,.0f} ops/sec")
        print(f"   Final available: {manager.get_total_available_length():,} units")


def compare_cpp_vs_python():
    """Compare C++ vs Python performance using unified workload"""
    
    print("\n[BALANCE]  C++ vs Python Performance Comparison")
    print("=" * 60)
    
    # Generate unified workload - SAME operations for fair comparison
    operations = generate_standard_workload(10_000)
    
    # Benchmark Python
    print("\n[STATS] Python Implementation:")
    py_manager = PyBoundary()
    
    start_time = time.perf_counter()
    execute_workload(py_manager, operations)
    py_time = time.perf_counter() - start_time
    
    print(f"   Time: {py_time*1000:.2f}ms")
    print(f"   Ops/sec: {len(operations)/py_time:,.0f}")
    
    # Benchmark C++
    print("\n=== C++ Implementation:")
    cpp_manager = CppBoundary()
    
    start_time = time.perf_counter()
    execute_workload(cpp_manager, operations)
    cpp_time = time.perf_counter() - start_time
    
    print(f"   Time: {cpp_time*1000:.2f}ms")
    print(f"   Ops/sec: {len(operations)/cpp_time:,.0f}")
    
    # Comparison
    speedup = py_time / cpp_time
    print(f"\n[FAST] C++ Speedup: {speedup:.1f}x faster")
    print(f"   Python time: {py_time*1000:.2f}ms")
    print(f"   C++ time: {cpp_time*1000:.2f}ms")
    print(f"   Time saved: {(py_time - cpp_time)*1000:.2f}ms")


def print_profiling_instructions():
    """Print instructions for profiling C++ code"""
    
    print("\n📋 C++ Profiling Instructions")
    print("=" * 60)
    print()
    print("🔥 To profile C++ implementations with flame graphs:")
    print()
    print("1. Install py-spy (shows both Python and C++ frames):")
    print("   uv pip install py-spy")
    print("   # Or: pip install py-spy")
    print()
    print("2. Profile this script with native frame capture:")
    print("   py-spy record --native -o cpp_profile.svg -- python tests/performance/cpp_profiler.py")
    print()
    print("3. View the flame graph:")
    print("   open cpp_profile.svg")
    print("   # Or upload to: https://www.speedscope.app/")
    print()
    print("Alternative format (speedscope - interactive):")
    print("   py-spy record --native --format speedscope -o profile.json -- python tests/performance/cpp_profiler.py")
    print("   # View at: https://www.speedscope.app/")
    print()
    print("For live profiling:")
    print("   py-spy top --native --pid <PID>")
    print()
    print("[STATS] macOS Instruments (GUI profiling):")
    print("   1. Open Instruments.app")
    print("   2. Choose 'Time Profiler'")
    print("   3. Click record and run: python tests/performance/cpp_profiler.py")
    print("   4. Shows full C++ call stacks with timing")
    print()
    print("🐧 Linux perf:")
    print("   perf record -g python tests/performance/cpp_profiler.py")
    print("   perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg")
    print()


def main():
    """Main profiling entry point"""
    
    print("🔥 Tree-Mendous C++ Profiler")
    print("=" * 60)
    print("Run this script under py-spy to profile C++ code")
    print()
    
    # Show what we're about to do
    print("This script will:")
    print("• Run intensive C++ workloads")
    print("• Compare C++ vs Python performance")
    print("• Show instructions for flame graph generation")
    print()
    
    # Run comparison
    compare_cpp_vs_python()
    
    # Run intensive workload (this is what should be profiled)
    run_cpp_intensive_workload(iterations=50_000)
    
    # Print instructions
    print_profiling_instructions()
    
    print("\n" + "=" * 60)
    print("[OK] C++ Profiler Complete!")
    print("=" * 60)
    print()
    print("[TIP] Quick Start:")
    print("   py-spy record --native -o flame.svg -- python tests/performance/cpp_profiler.py")


if __name__ == "__main__":
    main()
