#!/usr/bin/env python3
"""
Tree Structure Visualization Demo

Demonstrates how different interval tree implementations organize data
and provides visual representations of their internal structures.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager


def print_section(title: str) -> None:
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def visualize_interval_layout(intervals, total_span=1000, width=50):
    """Create ASCII visualization of interval layout"""
    print("\nInterval Layout:")
    print("█" * width)
    
    # Create visualization array
    viz = [' '] * width
    
    for interval in intervals:
        # Handle both IntervalResult and tuple formats
        if hasattr(interval, 'start'):
            start, end = interval.start, interval.end
        elif len(interval) == 3:
            start, end, _ = interval
        else:
            start, end = interval
            
        start_pos = int((start / total_span) * width)
        end_pos = int((end / total_span) * width)
        
        for i in range(start_pos, min(end_pos, width)):
            viz[i] = '█'
    
    print(''.join(viz))
    
    # Add scale
    scale_marks = []
    for i in range(0, width + 1, 10):
        pos = int((i / width) * total_span)
        scale_marks.append(f"{pos:>4}")
    
    print(' '.join(scale_marks))
    print()


def demo_basic_boundary_manager():
    """Demonstrate basic boundary manager"""
    print_section("Basic Boundary Manager")
    
    manager = IntervalManager()
    
    print("1. Initialize with 1000 units:")
    manager.release_interval(0, 1000)
    print(f"   Available: {manager.get_total_available_length()} units")
    
    print("\n2. Reserve some intervals:")
    reservations = [(100, 200), (400, 600), (800, 900)]
    
    for start, end in reservations:
        manager.reserve_interval(start, end)
        print(f"   Reserved [{start}, {end})")
    
    print(f"\n3. Remaining available: {manager.get_total_available_length()} units")
    
    intervals = manager.get_intervals()
    print(f"   Available intervals: {len(intervals)}")
    for interval in intervals:
        if hasattr(interval, 'start'):
            print(f"     [{interval.start}, {interval.end}) - {interval.length} units")
        else:
            start, end = interval
            print(f"     [{start}, {end}) - {end - start} units")
    
    visualize_interval_layout(intervals)
    
    print("\n4. Find suitable intervals:")
    test_sizes = [50, 150, 300]
    for size in test_sizes:
        result = manager.find_interval(0, size)
        if result:
            print(f"   {size} units: Found at [{result.start}, {result.end})")
        else:
            print(f"   {size} units: No suitable interval found")


def demo_summary_tree():
    """Demonstrate summary-enhanced tree"""
    print_section("Summary-Enhanced Tree")
    
    tree = SummaryIntervalTree()
    
    print("1. Build tree with multiple releases:")
    releases = [(0, 200), (300, 500), (600, 750), (800, 1000)]
    
    for start, end in releases:
        tree.release_interval(start, end)
        print(f"   Released [{start}, {end})")
    
    print("\n2. Tree summary statistics:")
    summary = tree.get_tree_summary()
    print(f"   Total free: {summary.total_free_length} units")
    print(f"   Interval count: {summary.contiguous_count}")
    print(f"   Largest chunk: {summary.largest_free_length} units")
    print(f"   Average chunk: {summary.avg_free_length:.1f} units")
    print(f"   Fragmentation: {summary.free_density:.2f}")
    
    intervals = tree.get_intervals()
    visualize_interval_layout(intervals)
    
    print("\n3. Smart allocation using summary:")
    # Try to find best fit for different sizes
    test_sizes = [100, 200, 300]
    for size in test_sizes:
        if summary.largest_free_length >= size:
            result = tree.find_best_fit(size)
            if result:
                start, end = result
                print(f"   Best fit for {size}: [{start}, {end})")
            else:
                print(f"   No best fit found for {size}")
        else:
            print(f"   {size} units too large (max: {summary.largest_free_length})")


def demo_treap_randomization():
    """Demonstrate treap probabilistic properties"""
    print_section("Randomized Treap")
    
    treap = IntervalTreap(random_seed=42)
    
    print("1. Build treap with random operations:")
    operations = [(0, 100), (200, 300), (400, 500), (600, 700), (800, 900)]
    
    for start, end in operations:
        treap.release_interval(start, end)
        print(f"   Released [{start}, {end}) - Tree size: {treap.get_tree_size()}")
    
    print(f"\n2. Tree properties:")
    print(f"   Tree size: {treap.get_tree_size()} nodes")
    print(f"   Total available: {treap.get_total_available_length()} units")
    print(f"   Properties valid: {treap.verify_treap_properties()}")
    
    intervals = treap.get_intervals()
    visualize_interval_layout(intervals)
    
    print("\n3. Random sampling demonstration:")
    samples = []
    for i in range(10):
        sample = treap.sample_random_interval()
        if sample:
            samples.append((sample.start, sample.end))
    
    print("   Random samples:")
    for i, (start, end) in enumerate(samples[:5]):
        print(f"     Sample {i+1}: [{start}, {end})")
    
    print("\n4. Split operation:")
    print("   Splitting at position 450...")
    left, right = treap.split(450)
    
    print(f"   Left treap: {left.get_tree_size()} nodes, {left.get_total_available_length()} units")
    print(f"   Right treap: {right.get_tree_size()} nodes, {right.get_total_available_length()} units")


def demo_boundary_summary_performance():
    """Demonstrate boundary summary with caching"""
    print_section("Boundary Summary with Performance Caching")
    
    manager = BoundarySummaryManager()
    
    print("1. Initialize and create fragmented layout:")
    manager.release_interval(0, 10000)
    
    # Create realistic fragmentation pattern
    allocations = [(i*500, i*500 + 200) for i in range(0, 20, 2)]
    for start, end in allocations:
        manager.reserve_interval(start, end)
        print(f"   Reserved [{start}, {end})")
    
    print("\n2. Performance-optimized queries:")
    
    # Warm up cache
    start_time = time.time()
    summary1 = manager.get_summary()
    cache_time = time.time() - start_time
    
    # Cached access
    start_time = time.time()
    for _ in range(100):
        summary2 = manager.get_summary()
    cached_time = time.time() - start_time
    
    perf = manager.get_performance_stats()
    print(f"   Cache performance:")
    print(f"     First access: {cache_time*1000:.2f}ms")
    print(f"     100 cached accesses: {cached_time*1000:.2f}ms")
    print(f"     Cache hit rate: {perf.cache_hit_rate:.1%}")
    
    print(f"\n3. Summary statistics:")
    print(f"   Total free: {summary1.total_free_length:,} units")
    print(f"   Fragments: {summary1.interval_count}")
    print(f"   Fragmentation: {summary1.fragmentation_index:.2f}")
    print(f"   Utilization: {summary1.utilization:.1%}")
    
    intervals = manager.get_intervals()
    visualize_interval_layout(intervals, total_span=10000, width=50)
    
    print("\n4. Advanced allocation strategies:")
    
    # Best fit allocation
    best_fit = manager.find_best_fit(150)
    if best_fit:
        print(f"   Best fit (150 units): [{best_fit.start}, {best_fit.end})")
    
    # Largest available
    largest = manager.find_largest_available()
    if largest:
        print(f"   Largest available: [{largest.start}, {largest.end}) - {largest.length} units")


def demo_cross_implementation_comparison():
    """Compare behavior across implementations"""
    print_section("Cross-Implementation Comparison")
    
    implementations = [
        ("Boundary Manager", IntervalManager()),
        ("Summary Tree", SummaryIntervalTree()),
        ("Treap", IntervalTreap(random_seed=42)),
        ("Boundary Summary", BoundarySummaryManager())
    ]
    
    # Apply same operations to all
    operations = [
        ('release', 0, 1000),
        ('reserve', 200, 400),
        ('reserve', 600, 800),
        ('release', 100, 300),  # Overlapping release
    ]
    
    print("Applying identical operations to all implementations:")
    for op, start, end in operations:
        print(f"  {op.title()} [{start}, {end})")
    
    print(f"\n{'Implementation':<20} {'Total Free':<12} {'Intervals':<10} {'Notes'}")
    print("-" * 60)
    
    for name, impl in implementations:
        # Apply operations
        for op, start, end in operations:
            if op == 'release':
                impl.release_interval(start, end)
            else:
                impl.reserve_interval(start, end)
        
        total_free = impl.get_total_available_length()
        intervals = impl.get_intervals()
        interval_count = len(intervals)
        
        # Implementation-specific notes
        notes = ""
        if hasattr(impl, 'get_tree_summary'):
            summary = impl.get_tree_summary()
            notes = f"frag: {summary.free_density:.2f}"
        elif hasattr(impl, 'verify_treap_properties'):
            valid = impl.verify_treap_properties()
            notes = f"valid: {valid}"
        elif hasattr(impl, 'get_performance_stats'):
            perf = impl.get_performance_stats()
            notes = f"ops: {perf.operation_count}"
        
        print(f"{name:<20} {total_free:<12} {interval_count:<10} {notes}")


def demo_real_world_scenarios():
    """Demonstrate real-world usage patterns"""
    print_section("Real-World Scenario Simulations")
    
    print("🏭 Manufacturing Line Scheduling")
    print("-" * 40)
    
    # 8-hour production line (480 minutes)
    line = BoundarySummaryManager()
    line.release_interval(0, 480)
    
    # Schedule production runs
    jobs = [
        ("Setup", 0, 30),
        ("Product A", 60, 120),
        ("Changeover", 150, 180),
        ("Product B", 200, 280),
        ("Maintenance", 420, 480)
    ]
    
    print("Scheduled jobs:")
    for job_name, start, end in jobs:
        line.reserve_interval(start, end)
        print(f"  {job_name:<12}: {start:3d}-{end:3d} min ({end-start:2d} min)")
    
    stats = line.get_availability_stats()
    print(f"\nLine efficiency:")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Available slots: {stats['free_chunks']}")
    print(f"  Largest gap: {stats['largest_chunk']} minutes")
    
    # Find slot for urgent job
    urgent_slot = line.find_best_fit(45)  # 45-minute job
    if urgent_slot:
        print(f"  Urgent job slot: [{urgent_slot.start}, {urgent_slot.end}) min")
    
    print("\n💾 Memory Pool Management")
    print("-" * 40)
    
    # 1MB memory pool
    memory = IntervalTreap(random_seed=time.time_ns() % 2**32)
    memory.release_interval(0, 1024*1024)
    
    # Simulate allocations
    allocations = []
    allocation_sizes = [4096, 8192, 2048, 16384, 1024, 32768]
    
    print("Memory allocations:")
    for i, size in enumerate(allocation_sizes):
        result = memory.find_interval(0, size)
        if result:
            memory.reserve_interval(result.start, result.end)
            allocations.append((f"Alloc-{i+1}", result.start, result.end, size))
            print(f"  Allocation {i+1}: {size:5d} bytes at offset {result.start:6d}")
        else:
            print(f"  Allocation {i+1}: {size:5d} bytes - FAILED")
    
    print(f"\nMemory status:")
    print(f"  Total allocated: {sum(size for _, _, _, size in allocations):,} bytes")
    print(f"  Available: {memory.get_total_available_length():,} bytes")
    print(f"  Fragments: {memory.get_tree_size()} pieces")
    
    # Show fragmentation pattern
    intervals = memory.get_intervals()
    if intervals:
        print(f"\nFragmentation pattern:")
        for i, interval in enumerate(intervals[:5]):  # Show first 5
            if hasattr(interval, 'start'):
                print(f"  Free block {i+1}: [{interval.start:6d}, {interval.end:6d}) - {interval.length:5d} bytes")
            else:
                start, end, _ = interval if len(interval) == 3 else (*interval, None)
                print(f"  Free block {i+1}: [{start:6d}, {end:6d}) - {end-start:5d} bytes")
        if len(intervals) > 5:
            print(f"  ... and {len(intervals) - 5} more fragments")


def demo_algorithmic_complexity():
    """Demonstrate performance characteristics"""
    print_section("Algorithmic Performance Analysis")
    
    # Test different tree sizes
    sizes = [100, 500, 1000, 2000]
    implementations = [
        ("Boundary", IntervalManager),
        ("Summary", SummaryIntervalTree),
        ("Treap", lambda: IntervalTreap(random_seed=42)),
        ("BoundarySummary", BoundarySummaryManager)
    ]
    
    print(f"{'Size':<8} {'Boundary':<12} {'Summary':<12} {'Treap':<12} {'B.Summary':<12}")
    print("-" * 60)
    
    for size in sizes:
        times = []
        
        for impl_name, impl_class in implementations:
            impl = impl_class()
            
            # Benchmark insertion performance
            start_time = time.perf_counter()
            
            # Create fragmented pattern
            for i in range(0, size, 10):
                impl.release_interval(i*2, i*2 + 10)
                if i % 40 == 0:  # Periodic reservations
                    impl.reserve_interval(i*2 + 5, i*2 + 8)
            
            duration = time.perf_counter() - start_time
            times.append(f"{duration*1000:.1f}ms")
        
        print(f"{size:<8} {times[0]:<12} {times[1]:<12} {times[2]:<12} {times[3]:<12}")


def demo_decision_tree():
    """Help users choose the right implementation"""
    print_section("Implementation Selection Guide")
    
    scenarios = [
        {
            "name": "Simple File System",
            "characteristics": ["Basic allocation", "Low memory overhead"],
            "recommendation": "IntervalManager",
            "rationale": "Minimal overhead, fast operations"
        },
        {
            "name": "Database Buffer Pool",
            "characteristics": ["Need utilization metrics", "Defragmentation analysis"],
            "recommendation": "SummaryIntervalTree", 
            "rationale": "Rich analytics for buffer management"
        },
        {
            "name": "Load Balancer",
            "characteristics": ["Dynamic traffic", "Fair distribution"],
            "recommendation": "IntervalTreap",
            "rationale": "Randomization ensures fairness"
        },
        {
            "name": "High-Frequency Trading",
            "characteristics": ["Microsecond queries", "Millions of operations"],
            "recommendation": "BoundarySummaryManager",
            "rationale": "Cached statistics for speed"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}")
        print(f"   Characteristics: {', '.join(scenario['characteristics'])}")
        print(f"   Recommended: {scenario['recommendation']}")
        print(f"   Why: {scenario['rationale']}")


def main():
    """Run all visualization demos"""
    print("🌳 Tree-Mendous: Interval Tree Architecture Visualization")
    print("=" * 60)
    print("Exploring different interval tree implementations and their trade-offs")
    
    try:
        demo_basic_boundary_manager()
        demo_summary_tree()
        demo_treap_randomization()
        demo_boundary_summary_performance()
        demo_cross_implementation_comparison()
        demo_algorithmic_complexity()
        demo_decision_tree()
        
        print_section("Summary")
        print("✅ All visualizations completed successfully!")
        print("\nKey Takeaways:")
        print("• Each implementation has distinct strengths")
        print("• Protocol consistency enables easy comparison")
        print("• Choose based on your specific requirements")
        print("• All implementations provide equivalent correctness")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
