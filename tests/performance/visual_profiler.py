#!/usr/bin/env python3
"""
Visual Performance Profiler

Creates ASCII-based performance visualizations without external dependencies.
Can also generate flame graphs if flameprof is available.
"""

import sys
import time
import random
import cProfile
import pstats
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager


def create_ascii_flame_chart(stats: pstats.Stats, max_width: int = 60) -> None:
    """Create ASCII visualization of performance hotspots"""
    
    print("\n🔥 ASCII Flame Chart (Top Functions by Cumulative Time)")
    print("=" * 80)
    
    # Get stats as list
    stats.calc_callees()
    
    # Extract function data
    function_data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func
        
        # Skip built-in functions
        if '<built-in' in filename or '<' in filename:
            continue
        
        # Simplify filename
        if 'treemendous' in filename:
            short_filename = filename.split('treemendous/')[-1]
        else:
            short_filename = Path(filename).name
        
        function_data.append({
            'name': f"{short_filename}:{func_name}",
            'cumulative_time': ct,
            'total_time': tt,
            'calls': nc
        })
    
    # Sort by cumulative time
    function_data.sort(key=lambda x: x['cumulative_time'], reverse=True)
    
    # Take top N functions
    top_functions = function_data[:20]
    
    if not top_functions:
        print("No function data available")
        return
    
    max_time = max(f['cumulative_time'] for f in top_functions)
    
    # Print header
    print(f"{'Function':<45} {'Time':<8} {'Calls':<8} {'Bar'}")
    print("-" * 80)
    
    for func in top_functions:
        # Create horizontal bar
        bar_length = int((func['cumulative_time'] / max_time) * max_width)
        bar = "█" * bar_length
        
        # Truncate long function names
        name = func['name']
        if len(name) > 44:
            name = "..." + name[-41:]
        
        print(f"{name:<45} {func['cumulative_time']:7.3f}s {func['calls']:<8} {bar}")


def create_call_tree_visualization(stats: pstats.Stats, max_depth: int = 3) -> None:
    """Create ASCII call tree showing function relationships"""
    
    print("\n🌲 Call Tree Visualization")
    print("=" * 80)
    
    # This is a simplified visualization - full call trees are complex
    # We'll show the top-level callers and their immediate callees
    
    stats.calc_callees()
    
    # Find main entry points (functions called from outside)
    entry_points = []
    
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func
        
        # Look for our main functions
        if 'treemendous' in filename and (ct > 0.01):  # Significant time spent
            entry_points.append((func, ct, nc))
    
    # Sort by cumulative time
    entry_points.sort(key=lambda x: x[1], reverse=True)
    
    # Print top entry points
    print("Top function calls:")
    
    for func, cum_time, calls in entry_points[:10]:
        filename, line, func_name = func
        short_name = Path(filename).name
        
        time_per_call = cum_time / calls if calls > 0 else 0
        
        print(f"├─ {short_name}:{func_name}")
        print(f"│  ├─ Cumulative: {cum_time:.3f}s")
        print(f"│  ├─ Calls: {calls}")
        print(f"│  └─ Time/call: {time_per_call*1000:.3f}ms")


def profile_and_visualize(impl_name: str, impl_class, num_operations: int = 10_000) -> None:
    """Profile implementation and create visualizations"""
    
    print(f"\n🔬 Profiling {impl_name}")
    print("=" * 80)
    
    # Generate operations
    import random
    random.seed(42)
    operations = []
    for _ in range(num_operations):
        op_type = random.choice(['reserve', 'release', 'find'])
        start = random.randint(0, 999_900)
        end = start + random.randint(10, 100)
        operations.append((op_type, start, end))
    
    # Create benchmark function
    def run_operations():
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
    
    # Profile
    profiler = cProfile.Profile()
    
    start_time = time.perf_counter()
    profiler.enable()
    run_operations()
    profiler.disable()
    total_time = time.perf_counter() - start_time
    
    # Create stats
    stats = pstats.Stats(profiler)
    
    # Performance summary
    print(f"\n⚡ Performance Summary:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Operations: {num_operations:,}")
    print(f"   Ops/sec: {num_operations/total_time:,.0f}")
    print(f"   Avg time/op: {(total_time/num_operations)*1_000_000:.2f}µs")
    
    # Create visualizations
    create_ascii_flame_chart(stats)
    create_call_tree_visualization(stats)
    
    # Save profile for external tools
    output_dir = Path("performance_profiles")
    output_dir.mkdir(exist_ok=True)
    
    safe_name = impl_name.lower().replace(' ', '_')
    profile_file = output_dir / f"{safe_name}.prof"
    profiler.dump_stats(str(profile_file))
    
    print(f"\n💾 Profile saved to: {profile_file}")
    print(f"   View with: snakeviz {profile_file}")
    print(f"   Or: flameprof {profile_file} > {profile_file.stem}_flame.svg")


def compare_all_implementations() -> None:
    """Compare all implementations side-by-side"""
    
    print("🏁 Implementation Performance Comparison")
    print("=" * 80)
    
    implementations = [
        ("Boundary Manager", IntervalManager),
        ("Summary Tree", SummaryIntervalTree),
        ("Treap", lambda: IntervalTreap(random_seed=42)),
        ("Boundary Summary", BoundarySummaryManager)
    ]
    
    # Try C++ implementations
    try:
        from treemendous.cpp.boundary import IntervalManager as CppBoundary
        implementations.append(("C++ Boundary", CppBoundary))
    except ImportError:
        pass
    
    try:
        from treemendous.cpp.treap import IntervalTreap as CppTreap  
        implementations.append(("C++ Treap", lambda: CppTreap(42)))
    except ImportError:
        pass
    
    try:
        from treemendous.cpp import BoundarySummaryManager as CppBoundarySummary
        implementations.append(("C++ BoundarySummary", CppBoundarySummary))
    except ImportError:
        pass
    
    # Generate workload
    num_ops = 5_000
    random.seed(42)
    operations = []
    for _ in range(num_ops):
        op_type = random.choice(['reserve', 'release', 'find'])
        start = random.randint(0, 999_900)
        end = start + random.randint(10, 100)
        operations.append((op_type, start, end))
    
    # Benchmark each
    results = []
    
    for impl_name, impl_class in implementations:
        print(f"Benchmarking {impl_name}...", end=' ')
        
        impl = impl_class()
        impl.release_interval(0, 1_000_000)
        
        start_time = time.perf_counter()
        
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
        
        duration = time.perf_counter() - start_time
        ops_per_sec = num_ops / duration
        
        results.append({
            'name': impl_name,
            'time': duration,
            'ops_per_sec': ops_per_sec
        })
        
        print(f"{duration*1000:.2f}ms")
    
    # Print comparison table
    print(f"\n📊 Performance Comparison ({num_ops:,} operations)")
    print("-" * 80)
    print(f"{'Implementation':<25} {'Time(ms)':<12} {'Ops/sec':<15} {'Relative':<10} {'Bar'}")
    print("-" * 80)
    
    fastest = min(r['time'] for r in results)
    
    for result in results:
        relative = result['time'] / fastest
        bar = "█" * int(relative * 20)
        
        print(f"{result['name']:<25} "
              f"{result['time']*1000:<11.2f} "
              f"{result['ops_per_sec']:<14,.0f} "
              f"{relative:<9.2f}x "
              f"{bar}")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        impl_name = sys.argv[1]
        
        implementations = {
            'boundary': ("Boundary Manager", IntervalManager),
            'summary': ("Summary Tree", SummaryIntervalTree),
            'treap': ("Treap", lambda: IntervalTreap(random_seed=42)),
            'boundary_summary': ("Boundary Summary", BoundarySummaryManager)
        }
        
        if impl_name == 'compare':
            compare_all_implementations()
        elif impl_name in implementations:
            name, impl_class = implementations[impl_name]
            profile_and_visualize(name, impl_class)
        else:
            print(f"❌ Unknown implementation: {impl_name}")
            print(f"   Available: {', '.join(implementations.keys())}, compare")
    
    else:
        # Default: compare all
        compare_all_implementations()
        
        print("\n💡 Tip: For detailed profiling of specific implementation:")
        print("   python visual_profiler.py boundary")
        print("   python visual_profiler.py summary")
        print("   python visual_profiler.py treap")
        print("   python visual_profiler.py boundary_summary")


if __name__ == "__main__":
    main()
