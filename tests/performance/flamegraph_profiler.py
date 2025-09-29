#!/usr/bin/env python3
"""
Flame Graph Profiler for Tree-Mendous

Uses various profiling tools to generate flame graphs for performance analysis:
- cProfile + flameprof for detailed call graphs
- py-spy for sampling profiler (if available)
- snakeviz for interactive visualization (if available)
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager
from tests.performance.workload import generate_standard_workload, execute_workload


def profile_with_cprofile(func: Callable, output_file: str) -> pstats.Stats:
    """Profile function with cProfile and save results"""
    profiler = cProfile.Profile()
    
    print(f"📊 Profiling {func.__name__}...")
    profiler.enable()
    func()
    profiler.disable()
    
    # Save raw profile data
    profiler.dump_stats(output_file)
    
    # Create stats object for analysis
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    return stats


def print_profile_summary(stats: pstats.Stats, n: int = 20) -> None:
    """Print top N functions by cumulative time"""
    print(f"\n🔥 Top {n} functions by cumulative time:")
    print("=" * 80)
    
    # Redirect to string buffer to capture output
    stream = io.StringIO()
    stats.stream = stream
    stats.print_stats(n)
    
    print(stream.getvalue())


def benchmark_interval_manager(operations: List[Tuple[str, int, int]]) -> None:
    """Benchmark IntervalManager using unified workload"""
    manager = IntervalManager()
    execute_workload(manager, operations)


def benchmark_summary_tree(operations: List[Tuple[str, int, int]]) -> None:
    """Benchmark SummaryIntervalTree using unified workload"""
    tree = SummaryIntervalTree()
    execute_workload(tree, operations)


def benchmark_treap(operations: List[Tuple[str, int, int]]) -> None:
    """Benchmark IntervalTreap using unified workload"""
    treap = IntervalTreap(random_seed=42)
    execute_workload(treap, operations)


def benchmark_boundary_summary(operations: List[Tuple[str, int, int]]) -> None:
    """Benchmark BoundarySummaryManager using unified workload"""
    manager = BoundarySummaryManager()
    execute_workload(manager, operations)


def profile_all_implementations(num_operations: int = 10_000) -> Dict[str, pstats.Stats]:
    """Profile all interval tree implementations (Python and C++)"""
    
    print(f"🔥 Profiling all implementations with {num_operations} operations")
    print("=" * 80)
    
    # Generate shared workload - SAME operations for all implementations
    operations = generate_standard_workload(num_operations)
    
    # Python implementations
    implementations = [
        ("py_boundary", lambda: benchmark_interval_manager(operations)),
        ("py_summary", lambda: benchmark_summary_tree(operations)),
        ("py_treap", lambda: benchmark_treap(operations)),
        ("py_boundary_summary", lambda: benchmark_boundary_summary(operations))
    ]
    
    # Add C++ implementations if available (using same unified workload)
    try:
        from treemendous.cpp.boundary import IntervalManager as CppBoundary
        from treemendous.cpp.treap import IntervalTreap as CppTreap
        
        def benchmark_cpp_boundary(ops):
            manager = CppBoundary()
            execute_workload(manager, ops)
        
        def benchmark_cpp_treap(ops):
            treap = CppTreap(42)
            execute_workload(treap, ops)
        
        implementations.extend([
            ("cpp_boundary", lambda: benchmark_cpp_boundary(operations)),
            ("cpp_treap", lambda: benchmark_cpp_treap(operations))
        ])
        
        # Try to add C++ BoundarySummary
        try:
            from treemendous.cpp import BoundarySummaryManager as CppBoundarySummary
            
            def benchmark_cpp_boundary_summary(ops):
                manager = CppBoundarySummary()
                execute_workload(manager, ops)
            
            implementations.append(("cpp_boundary_summary", lambda: benchmark_cpp_boundary_summary(operations)))
            print("   ✅ C++ implementations included (Boundary, Treap, BoundarySummary)")
        except ImportError:
            print("   ✅ C++ implementations included (Boundary, Treap)")
            
    except ImportError:
        print("   ⚠️  C++ implementations not available")
    
    results = {}
    output_dir = Path("performance_profiles")
    output_dir.mkdir(exist_ok=True)
    
    for name, func in implementations:
        output_file = str(output_dir / f"{name}.prof")
        stats = profile_with_cprofile(func, output_file)
        results[name] = stats
        
        print(f"\n✅ {name.upper()} profiling complete")
        print(f"   Profile saved to: {output_file}")
        
        # Print top 10 hotspots
        print(f"\n   Top 10 hotspots:")
        stream = io.StringIO()
        stats.stream = stream
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        # Print just the function names and times
        lines = stream.getvalue().split('\n')
        for line in lines[5:15]:  # Skip headers
            if line.strip():
                print(f"   {line}")
    
    return results


def generate_flamegraph_instructions(prof_file: str) -> None:
    """Print instructions for generating flame graphs"""
    print(f"\n🔥 To generate flame graph from {prof_file}:")
    print("=" * 80)
    print("\nOption 1: Using flameprof (recommended)")
    print("  pip install flameprof")
    print(f"  flameprof {prof_file} > flamegraph.svg")
    print(f"  open flamegraph.svg")
    
    print("\nOption 2: Using snakeviz (interactive)")
    print("  pip install snakeviz")
    print(f"  snakeviz {prof_file}")
    
    print("\nOption 3: Using py-spy (live profiling)")
    print("  pip install py-spy")
    print("  py-spy record -o profile.svg -- python your_script.py")
    
    print("\nOption 4: Using gprof2dot + graphviz")
    print("  pip install gprof2dot")
    print(f"  gprof2dot -f pstats {prof_file} | dot -Tsvg -o callgraph.svg")


def try_generate_flamegraph(prof_file: str, impl_name: str) -> bool:
    """Try to generate flame graph if flameprof is available"""
    try:
        import subprocess
        
        output_file = prof_file.replace('.prof', '_flame.svg')
        
        print(f"🔥 Generating flame graph for {impl_name}...")
        result = subprocess.run(
            ['flameprof', prof_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout:
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            print(f"   ✅ Flame graph saved to: {output_file}")
            return True
        else:
            # flameprof can fail on C++ code with minimal Python frames
            if 'cpp' in impl_name.lower():
                print(f"   ℹ️  C++ code too fast for Python profiler (use py-spy instead)")
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"   ⚠️  flameprof failed: {error_msg[:100]}")
            return False
            
    except FileNotFoundError:
        print(f"   ℹ️  flameprof not installed")
        return False
    except Exception as e:
        print(f"   ⚠️  Error generating flame graph: {str(e)[:100]}")
        return False


def compare_implementations_performance() -> None:
    """Compare performance across all implementations (Python and C++)"""
    
    print("\n⚡ Performance Comparison (Python vs C++)")
    print("=" * 80)
    
    # Use unified workload for fair comparison
    operations = generate_standard_workload(10_000)
    
    # Python implementations
    implementations = [
        ("Python Boundary", lambda: benchmark_interval_manager(operations)),
        ("Python Summary Tree", lambda: benchmark_summary_tree(operations)),
        ("Python Treap", lambda: benchmark_treap(operations)),
        ("Python Boundary Summary", lambda: benchmark_boundary_summary(operations))
    ]
    
    # Add C++ implementations if available (using same unified workload)
    try:
        from treemendous.cpp.boundary import IntervalManager as CppBoundary
        from treemendous.cpp.treap import IntervalTreap as CppTreap
        
        def benchmark_cpp_boundary(ops):
            manager = CppBoundary()
            execute_workload(manager, ops)
        
        def benchmark_cpp_treap(ops):
            treap = CppTreap(42)
            execute_workload(treap, ops)
        
        implementations.extend([
            ("C++ Boundary", lambda: benchmark_cpp_boundary(operations)),
            ("C++ Treap", lambda: benchmark_cpp_treap(operations))
        ])
        
        # Try to add C++ BoundarySummary
        try:
            from treemendous.cpp import BoundarySummaryManager as CppBoundarySummary
            
            def benchmark_cpp_boundary_summary(ops):
                manager = CppBoundarySummary()
                execute_workload(manager, ops)
            
            implementations.append(("C++ BoundarySummary", lambda: benchmark_cpp_boundary_summary(operations)))
            print("   ✅ Including C++ implementations for comparison (Boundary, Treap, BoundarySummary)")
        except ImportError:
            print("   ✅ Including C++ implementations for comparison (Boundary, Treap)")
            
    except ImportError:
        print("   ⚠️  C++ implementations not available")
    
    results = []
    
    for name, func in implementations:
        # Warm-up run
        func()
        
        # Timed runs
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            func()
            duration = time.perf_counter() - start_time
            times.append(duration)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        results.append({
            'name': name,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min(times),
            'max_time': max(times)
        })
    
    # Print results table
    print(f"\n{'Implementation':<20} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Std Dev':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['avg_time']*1000:<11.2f}ms "
              f"{result['min_time']*1000:<11.2f}ms "
              f"{result['max_time']*1000:<11.2f}ms "
              f"{result['std_time']*1000:<9.2f}ms")
    
    # Calculate relative performance
    print(f"\n📊 Relative Performance (vs Python Boundary):")
    baseline = results[0]['avg_time']
    
    for result in results:
        if result['avg_time'] < baseline:
            # Faster than baseline - show speedup
            speedup = baseline / result['avg_time']
            print(f"   {result['name']:<20} {speedup:5.2f}x faster")
        elif result['avg_time'] > baseline:
            # Slower than baseline - show slowdown
            slowdown = result['avg_time'] / baseline
            print(f"   {result['name']:<20} {slowdown:5.2f}x slower")
        else:
            print(f"   {result['name']:<20} 1.00x (baseline)")


def profile_hotspots(impl_name: str, operations: List[Tuple[str, int, int]]) -> None:
    """Deep dive into performance hotspots for specific implementation"""
    
    print(f"\n🔍 Deep Profiling: {impl_name}")
    print("=" * 80)
    
    # Map implementation names to functions
    implementations = {
        'boundary': benchmark_interval_manager,
        'summary': benchmark_summary_tree,
        'treap': benchmark_treap,
        'boundary_summary': benchmark_boundary_summary
    }
    
    if impl_name not in implementations:
        print(f"❌ Unknown implementation: {impl_name}")
        print(f"   Available: {', '.join(implementations.keys())}")
        return
    
    func = lambda: implementations[impl_name](operations)
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    
    # Show different sort orders
    print("\n🔥 By Cumulative Time:")
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    
    print("\n⚡ By Total Time:")
    stats.sort_stats('time')
    stats.print_stats(15)
    
    print("\n📞 By Call Count:")
    stats.sort_stats('calls')
    stats.print_stats(15)


def main():
    """Run comprehensive profiling suite"""
    
    print("🔥 Tree-Mendous Flame Graph Profiler")
    print("=" * 80)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            # Profile all implementations
            stats_dict = profile_all_implementations(num_operations=10_000)
            
            # Try to generate flame graphs
            output_dir = Path("performance_profiles")
            for impl_name in stats_dict.keys():
                prof_file = str(output_dir / f"{impl_name}.prof")
                
                if not try_generate_flamegraph(prof_file, impl_name):
                    generate_flamegraph_instructions(prof_file)
        
        elif command == "compare":
            # Quick performance comparison
            compare_implementations_performance()
        
        elif command in ["boundary", "summary", "treap", "boundary_summary"]:
            # Deep profile specific implementation
            operations = generate_standard_workload(10_000)
            profile_hotspots(command, operations)
        
        else:
            print(f"❌ Unknown command: {command}")
            print("   Available: all, compare, boundary, summary, treap, boundary_summary")
    
    else:
        # Default: profile all with comparison
        print("Running default profiling suite...")
        print()
        
        # 1. Quick comparison
        compare_implementations_performance()
        
        # 2. Detailed profiling
        stats_dict = profile_all_implementations(num_operations=5_000)
        
        # 3. Try to generate visualizations
        print("\n🎨 Attempting to generate visualizations...")
        output_dir = Path("performance_profiles")
        
        flame_generated = False
        for impl_name in stats_dict.keys():
            prof_file = str(output_dir / f"{impl_name}.prof")
            if try_generate_flamegraph(prof_file, impl_name):
                flame_generated = True
        
        if not flame_generated:
            print("\n📋 Flame graph generation requires additional packages.")
            print("   Install with: uv sync --extra profiling")
            print("\n   Available profile files in: performance_profiles/")
            print("   Use: snakeviz performance_profiles/py_boundary.prof")
        
        # Check if any C++ profiles were generated
        cpp_profiles = [impl for impl in stats_dict.keys() if 'cpp' in impl.lower()]
        if cpp_profiles:
            print("\n💡 C++ Profiling Note:")
            print("   Python profilers (cProfile, flameprof) can't see into C++ code.")
            print("   For C++ flame graphs with native frames, use py-spy:")
            print()
            print("   py-spy record --native -o cpp_flame.svg -- uv run python tests/performance/cpp_profiler.py")
            print("   open cpp_flame.svg")
        
        print("\n✅ Profiling complete!")
        print("\n💡 Usage Tips:")
        print("   • python flamegraph_profiler.py all         - Profile all implementations")
        print("   • python flamegraph_profiler.py compare     - Quick Python vs C++ comparison")
        print("   • python flamegraph_profiler.py boundary    - Deep profile specific impl")
        print()
        print("📚 For more details:")
        print("   • Python profiling: docs/PERFORMANCE_PROFILING.md")
        print("   • C++ profiling: docs/CPP_PROFILING_GUIDE.md")


if __name__ == "__main__":
    main()
