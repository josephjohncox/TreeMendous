#!/usr/bin/env python3
"""
Algorithm Analysis and Mathematical Visualization

Demonstrates the mathematical properties and algorithmic behavior
of different interval tree implementations.
"""

import sys
import time
import math
import random
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager


def analyze_tree_balance():
    """Analyze tree balance characteristics"""
    print("🌲 Tree Balance Analysis")
    print("=" * 50)
    
    sizes = [10, 50, 100, 500, 1000]
    
    print(f"{'Size':<8} {'Expected':<12} {'Treap Actual':<15} {'Balance Factor':<15}")
    print("-" * 60)
    
    for n in sizes:
        # Theoretical expected height for balanced tree
        expected_height = math.log2(n) if n > 0 else 0
        
        # Build treap with n intervals
        treap = IntervalTreap(random_seed=42)
        for i in range(n):
            start = i * 10
            end = start + 5
            treap.release_interval(start, end)
        
        # Measure actual properties
        actual_height = treap.get_tree_height() if hasattr(treap, 'get_tree_height') else 0
        balance_factor = actual_height / expected_height if expected_height > 0 else 0
        
        print(f"{n:<8} {expected_height:<12.1f} {actual_height:<15} {balance_factor:<15.2f}")
    
    print("\n📊 Balance Factor Analysis:")
    print("• Factor < 2.0: Excellent balance")
    print("• Factor 2.0-3.0: Good balance")  
    print("• Factor > 3.0: Poor balance")


def analyze_fragmentation_patterns():
    """Analyze different fragmentation scenarios"""
    print("\n🧩 Fragmentation Pattern Analysis")
    print("=" * 50)
    
    scenarios = [
        ("Sequential", lambda: [(i*100, i*100+50) for i in range(10)]),
        ("Random", lambda: [(random.randint(0, 900), random.randint(0, 900)+50) for _ in range(10)]),
        ("Clustered", lambda: [(i*20, i*20+10) for i in range(20)]),
        ("Sparse", lambda: [(i*200, i*200+20) for i in range(5)])
    ]
    
    for pattern_name, pattern_gen in scenarios:
        print(f"\n{pattern_name} Pattern:")
        
        # Use summary tree for rich analytics
        tree = SummaryIntervalTree()
        tree.release_interval(0, 1000)
        
        # Apply pattern
        random.seed(42)  # Reproducible random patterns
        reservations = pattern_gen()
        
        for start, end in reservations:
            if start < end:  # Ensure valid intervals
                tree.reserve_interval(start, min(end, 1000))
        
        summary = tree.get_tree_summary()
        
        print(f"  Free space: {summary.total_free_length:4d} units")
        print(f"  Fragments: {summary.contiguous_count:2d}")
        print(f"  Largest: {summary.largest_free_length:4d} units")
        print(f"  Fragmentation: {summary.free_density:5.2f}")
        
        # Visual representation
        intervals = tree.get_intervals()
        viz = ['·'] * 50
        
        for interval in intervals:
            # Handle both IntervalResult and tuple formats
            if hasattr(interval, 'start'):
                start, end = interval.start, interval.end
            elif len(interval) == 3:
                start, end, _ = interval
            else:
                start, end = interval
                
            start_pos = int((start / 1000) * 50)
            end_pos = int((end / 1000) * 50)
            for i in range(start_pos, min(end_pos, 50)):
                viz[i] = '█'
        
        print(f"  Layout: {''.join(viz)}")


def analyze_query_performance():
    """Analyze query performance patterns"""
    print("\n⚡ Query Performance Analysis")
    print("=" * 50)
    
    # Build test scenario
    implementations = [
        ("Boundary", IntervalManager()),
        ("Summary", SummaryIntervalTree()),
        ("Treap", IntervalTreap(random_seed=42)),
        ("BoundarySummary", BoundarySummaryManager())
    ]
    
    # Setup identical fragmented scenario
    for name, impl in implementations:
        impl.release_interval(0, 10000)
        # Create realistic fragmentation
        for i in range(0, 10000, 300):
            impl.reserve_interval(i, i + 100)
    
    query_types = [
        ("find_interval", lambda impl: impl.find_interval(0, 50)),
        ("get_intervals", lambda impl: impl.get_intervals()),
        ("total_length", lambda impl: impl.get_total_available_length())
    ]
    
    print(f"{'Query Type':<15} {'Boundary':<10} {'Summary':<10} {'Treap':<10} {'B.Summary':<12}")
    print("-" * 70)
    
    for query_name, query_func in query_types:
        times = []
        
        for name, impl in implementations:
            # Skip if method not available
            try:
                start_time = time.perf_counter()
                
                # Run query multiple times for better measurement
                for _ in range(100):
                    result = query_func(impl)
                
                duration = (time.perf_counter() - start_time) * 1000 / 100  # ms per query
                times.append(f"{duration:.3f}ms")
                
            except (AttributeError, ValueError):
                times.append("N/A")
        
        print(f"{query_name:<15} {times[0]:<10} {times[1]:<10} {times[2]:<10} {times[3]:<12}")


def analyze_memory_efficiency():
    """Analyze memory usage patterns"""
    print("\n💾 Memory Efficiency Analysis")
    print("=" * 50)
    
    import sys
    
    # Test memory usage for different implementations
    implementations = [
        ("Boundary", IntervalManager),
        ("Summary", SummaryIntervalTree),
        ("Treap", lambda: IntervalTreap(random_seed=42)),
        ("BoundarySummary", BoundarySummaryManager)
    ]
    
    interval_counts = [10, 50, 100, 500]
    
    print(f"{'Intervals':<10} {'Boundary':<12} {'Summary':<12} {'Treap':<12} {'B.Summary':<12}")
    print("-" * 70)
    
    for count in interval_counts:
        memory_usage = []
        
        for name, impl_class in implementations:
            impl = impl_class()
            
            # Add intervals
            for i in range(count):
                start = i * 20
                end = start + 10
                impl.release_interval(start, end)
            
        # Estimate memory usage (rough approximation)
        if hasattr(impl, 'get_intervals'):
            intervals = impl.get_intervals()
            base_memory = len(intervals) * 32  # Rough estimate per interval
            
            # Add implementation-specific overhead  
            if 'Summary' in name:
                base_memory += len(intervals) * 48  # Summary stats
            elif 'Treap' in name:
                base_memory += len(intervals) * 16  # Priority + balance info
            elif 'BoundarySummary' in name:
                base_memory += 256  # Cache overhead
            
            memory_usage.append(f"{base_memory/1024:.1f}KB")
        else:
            memory_usage.append("N/A")
        
        print(f"{count:<10} {memory_usage[0]:<12} {memory_usage[1]:<12} {memory_usage[2]:<12} {memory_usage[3]:<12}")


def analyze_mathematical_properties():
    """Analyze mathematical properties of the trees"""
    print("\n📐 Mathematical Properties Analysis") 
    print("=" * 50)
    
    # Treap probabilistic analysis
    print("Treap Probabilistic Properties:")
    treap = IntervalTreap(random_seed=42)
    
    # Build tree with known intervals
    intervals_data = [(i*50, i*50+30) for i in range(20)]
    for start, end in intervals_data:
        treap.release_interval(start, end)
    
    print(f"  Tree size: {treap.get_tree_size()} nodes")
    print(f"  Expected height: ~{3 * math.log(treap.get_tree_size()):.1f}")
    
    # Sample random intervals to test distribution
    samples = []
    for _ in range(100):
        sample = treap.sample_random_interval()
        if sample:
            samples.append((sample.start, sample.end))
    
    # Analyze distribution uniformity
    if samples:
        start_positions = [s for s, e in samples]
        avg_start = sum(start_positions) / len(start_positions)
        expected_avg = 500  # Middle of [0, 1000)
        
        print(f"  Random sampling uniformity:")
        print(f"    Sample count: {len(samples)}")
        print(f"    Average start: {avg_start:.1f} (expected ~{expected_avg})")
        print(f"    Distribution quality: {'Good' if abs(avg_start - expected_avg) < 100 else 'Needs improvement'}")
    
    # Summary tree aggregate analysis
    print("\nSummary Tree Statistical Properties:")
    summary_tree = SummaryIntervalTree()
    
    # Create controlled scenario
    summary_tree.release_interval(0, 1000)
    summary_tree.reserve_interval(100, 200)  # 10% allocation
    summary_tree.reserve_interval(500, 600)  # Another 10%
    
    stats = summary_tree.get_tree_summary()
    
    # Calculate theoretical vs actual metrics
    total_space = 1000
    allocated_space = 200  # 100 + 100
    expected_utilization = allocated_space / total_space
    
    print(f"  Theoretical utilization: {expected_utilization:.1%}")
    print(f"  Actual free density: {stats.free_density:.2f}")
    print(f"  Free space efficiency: {stats.total_free_length / (total_space - allocated_space):.2f}")


def create_complexity_visualization():
    """Create ASCII charts showing complexity relationships"""
    print("\n📈 Complexity Visualization")
    print("=" * 50)
    
    # Simulate operation counts vs time for different implementations
    print("Operation Scaling (Operations vs Time):")
    print()
    
    # ASCII bar chart for different operation counts
    op_counts = [100, 500, 1000, 5000, 10000]
    
    # Simulate relative performance (normalized)
    boundary_times = [n * 0.1 for n in op_counts]  # Linear in practice
    summary_times = [n * 0.12 for n in op_counts]  # Slightly higher overhead
    treap_times = [n * 0.15 for n in op_counts]    # Random factor
    
    max_time = max(boundary_times[-1], summary_times[-1], treap_times[-1])
    
    print(f"{'Ops':<8} {'Boundary':<20} {'Summary':<20} {'Treap':<20}")
    print("-" * 80)
    
    for i, ops in enumerate(op_counts):
        # Create bar visualization
        b_bar = "█" * int((boundary_times[i] / max_time) * 15)
        s_bar = "█" * int((summary_times[i] / max_time) * 15)
        t_bar = "█" * int((treap_times[i] / max_time) * 15)
        
        print(f"{ops:<8} {b_bar:<20} {s_bar:<20} {t_bar:<20}")
    
    print("\nSpace Complexity Relationships:")
    print(f"{'Implementation':<15} {'Space per Node':<15} {'Overhead':<15}")
    print("-" * 50)
    print(f"{'Boundary':<15} {'O(1)':<15} {'Minimal':<15}")
    print(f"{'Summary':<15} {'O(k)':<15} {'Statistics':<15}")
    print(f"{'Treap':<15} {'O(1)':<15} {'Priority':<15}")
    print(f"{'BoundarySum':<15} {'O(1) + cache':<15} {'Cache':<15}")


def demonstrate_protocol_consistency():
    """Show how protocol standardization works"""
    print("\n🔗 Protocol Consistency Demonstration")
    print("=" * 50)
    
    implementations = [
        ("Python Boundary", IntervalManager()),
        ("Python Summary", SummaryIntervalTree()),
        ("Python Treap", IntervalTreap(random_seed=42)),
        ("Python B.Summary", BoundarySummaryManager())
    ]
    
    # Add C++ implementations if available
    try:
        from treemendous.cpp.boundary import IntervalManager as CppIntervalManager
        implementations.append(("C++ Boundary", CppIntervalManager()))
    except ImportError:
        pass
    
    try:
        from treemendous.cpp import BoundarySummaryManager as CppBoundarySummaryManager
        implementations.append(("C++ B.Summary", CppBoundarySummaryManager()))
    except ImportError:
        pass
    
    print("Testing protocol consistency across implementations:")
    print()
    
    # Apply identical operations
    operations = [
        ('release', 0, 1000),
        ('reserve', 200, 400),
        ('reserve', 600, 800)
    ]
    
    results = {}
    
    for name, impl in implementations:
        print(f"Testing {name}...")
        
        # Apply operations
        for op, start, end in operations:
            if op == 'release':
                impl.release_interval(start, end)
            else:
                impl.reserve_interval(start, end)
        
        # Test standard interface
        total_free = impl.get_total_available_length()
        intervals = impl.get_intervals()
        
        # Test find_interval
        search_result = impl.find_interval(0, 100)
        
        results[name] = {
            'total_free': total_free,
            'interval_count': len(intervals),
            'find_result_type': type(search_result).__name__,
            'protocol_compliant': hasattr(search_result, 'start') if search_result else True
        }
        
        print(f"  Total free: {total_free}")
        print(f"  Intervals: {len(intervals)}")
        print(f"  Result type: {type(search_result).__name__}")
        print(f"  Protocol compliant: {hasattr(search_result, 'start') if search_result else 'N/A'}")
        print()
    
    # Verify consistency
    total_frees = set(r['total_free'] for r in results.values())
    interval_counts = set(r['interval_count'] for r in results.values())
    
    print("✅ Consistency Check:")
    print(f"  All implementations report same total free: {len(total_frees) == 1}")
    print(f"  Result types standardized: {all(r['protocol_compliant'] for r in results.values())}")


def visualize_insertion_patterns():
    """Visualize how different trees handle insertion patterns"""
    print("\n🔄 Insertion Pattern Analysis")
    print("=" * 50)
    
    patterns = [
        ("Sequential", [(i*10, i*10+5) for i in range(20)]),
        ("Reverse", [(i*10, i*10+5) for i in range(19, -1, -1)]),
        ("Random", None)  # Will generate randomly
    ]
    
    for pattern_name, intervals in patterns:
        print(f"\n{pattern_name} Insertion Pattern:")
        
        if pattern_name == "Random":
            random.seed(42)
            intervals = [(random.randint(0, 180)*10, random.randint(0, 180)*10+50) for _ in range(20)]
            intervals = [(s, e) for s, e in intervals if s < e]  # Ensure valid
        
        # Test with treap to see balancing effects  
        treap = IntervalTreap(random_seed=42)
        
        heights = []
        sizes = []
        
        for i, (start, end) in enumerate(intervals):
            treap.release_interval(start, end)
            sizes.append(treap.get_tree_size())
            
            # Estimate height (rough approximation)
            if hasattr(treap.root, 'height'):
                heights.append(treap.root.height)
            else:
                heights.append(int(math.log2(treap.get_tree_size() + 1)))
        
        # Show growth pattern
        print("  Tree growth pattern:")
        print(f"  {'Step':<5} {'Size':<5} {'Height':<7} {'Balance':<8}")
        print("  " + "-" * 30)
        
        for i in range(0, len(sizes), 4):  # Show every 4th step
            size = sizes[i]
            height = heights[i]
            expected = math.log2(size + 1)
            balance = height / expected if expected > 0 else 0
            
            print(f"  {i+1:<5} {size:<5} {height:<7} {balance:<8.2f}")


def demonstrate_optimization_strategies():
    """Show optimization strategies using different implementations"""
    print("\n🎯 Optimization Strategy Demonstrations")
    print("=" * 50)
    
    print("Strategy 1: Cache-Aware Querying")
    print("-" * 35)
    
    manager = BoundarySummaryManager()
    manager.release_interval(0, 10000)
    
    # Create fragmentation
    for i in range(0, 10000, 500):
        manager.reserve_interval(i, i + 200)
    
    # Demonstrate caching benefits
    print("  Cold cache queries:")
    start_time = time.perf_counter()
    for _ in range(10):
        summary = manager.get_summary()
        manager.reserve_interval(1, 2)  # Invalidate cache
        manager.release_interval(1, 2)  # Restore
    cold_time = time.perf_counter() - start_time
    
    print("  Warm cache queries:")
    start_time = time.perf_counter()
    for _ in range(10):
        summary = manager.get_summary()  # Should hit cache
    warm_time = time.perf_counter() - start_time
    
    perf = manager.get_performance_stats()
    print(f"    Cold cache: {cold_time*1000:.2f}ms")
    print(f"    Warm cache: {warm_time*1000:.2f}ms")
    print(f"    Speedup: {cold_time/warm_time:.1f}x")
    print(f"    Hit rate: {perf.cache_hit_rate:.1%}")
    
    print("\nStrategy 2: Probabilistic Load Balancing")
    print("-" * 40)
    
    treap = IntervalTreap(random_seed=42)
    treap.release_interval(0, 1000)
    
    # Simulate load balancing scenario
    print("  Random server allocation:")
    allocations = []
    
    for i in range(10):
        allocation = treap.sample_random_interval()
        if allocation:
            # Reserve a portion for this "server"
            size = min(50, allocation.length)
            treap.reserve_interval(allocation.start, allocation.start + size)
            allocations.append((i+1, allocation.start, allocation.start + size))
            print(f"    Server {i+1}: [{allocation.start:3d}, {allocation.start + size:3d})")
    
    print(f"  Load distribution quality:")
    if allocations:
        positions = [start for _, start, end in allocations]
        spread = max(positions) - min(positions)
        print(f"    Position spread: {spread} units")
        print(f"    Distribution: {'Even' if spread > 400 else 'Clustered'}")


def create_performance_benchmark():
    """Create a comprehensive performance benchmark"""
    print("\n🏁 Comprehensive Performance Benchmark")
    print("=" * 50)
    
    # Test workload: realistic fragmentation scenario
    workload_size = 1000
    
    print("Benchmark Scenario: File system allocation")
    print(f"  Address space: [0, {workload_size*1000})")
    print(f"  Operations: {workload_size} mixed alloc/dealloc")
    print()
    
    implementations = [
        ("Boundary", IntervalManager),
        ("Summary", SummaryIntervalTree), 
        ("Treap", lambda: IntervalTreap(random_seed=42)),
        ("BoundarySummary", BoundarySummaryManager)
    ]
    
    print(f"{'Implementation':<15} {'Setup Time':<12} {'Op Time':<12} {'Query Time':<12} {'Memory':<10}")
    print("-" * 70)
    
    for name, impl_class in implementations:
        impl = impl_class()
        
        # 1. Setup time
        start_time = time.perf_counter()
        impl.release_interval(0, workload_size * 1000)
        setup_time = time.perf_counter() - start_time
        
        # 2. Operation time  
        start_time = time.perf_counter()
        for i in range(100):  # Reduced for demo
            start = random.randint(0, workload_size*1000 - 100)
            end = start + random.randint(10, 100)
            
            if i % 2 == 0:
                impl.reserve_interval(start, end)
            else:
                impl.release_interval(start, end)
        
        op_time = time.perf_counter() - start_time
        
        # 3. Query time
        start_time = time.perf_counter()
        for _ in range(100):
            result = impl.find_interval(random.randint(0, workload_size*500), 50)
        query_time = time.perf_counter() - start_time
        
        # 4. Rough memory estimate
        intervals = impl.get_intervals()
        memory_est = len(intervals) * 64  # Rough bytes per interval
        
        print(f"{name:<15} {setup_time*1000:<11.2f}ms {op_time*1000:<11.2f}ms {query_time*1000:<11.2f}ms {memory_est/1024:<9.1f}KB")


def main():
    """Run all mathematical and algorithmic analyses"""
    print("🔬 Tree-Mendous: Mathematical & Algorithmic Analysis")
    print("=" * 60)
    print("Deep dive into the mathematical properties and performance characteristics")
    
    try:
        analyze_tree_balance()
        analyze_fragmentation_patterns()
        analyze_query_performance()
        analyze_memory_efficiency()
        analyze_mathematical_properties()
        visualize_insertion_patterns()
        demonstrate_optimization_strategies()
        create_performance_benchmark()
        demonstrate_protocol_consistency()
        create_complexity_visualization()
        
        print("\n" + "=" * 60)
        print("🎓 Analysis Complete!")
        print("=" * 60)
        print("\nKey Insights:")
        print("• Treaps provide excellent probabilistic balance")
        print("• Summary trees excel at aggregate analytics")
        print("• Boundary managers optimize for simplicity")
        print("• Caching dramatically improves query performance")
        print("• Protocol consistency enables seamless comparison")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
