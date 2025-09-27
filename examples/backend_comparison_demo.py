#!/usr/bin/env python3
"""
Backend Comparison Demo

Demonstrates how to switch between Python and C++ implementations
while maintaining identical functionality through unified interfaces.
"""

import sys
from pathlib import Path

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common.backend_config import (
    parse_backend_args, handle_backend_args, create_example_tree, 
    get_tree_analytics, get_backend_manager, detect_tree_features
)


def demo_same_operations_different_backends():
    """Run identical operations on different backends"""
    print("🔄 Same Operations, Different Backends")
    print("=" * 50)
    
    manager = get_backend_manager()
    available_backends = manager.get_available_backends()
    
    if len(available_backends) < 2:
        print("⚠️  Need at least 2 backends for meaningful comparison")
        available_list = list(available_backends.keys())
        if available_list:
            print(f"Only available: {available_list[0]}")
        return
    
    # Define test scenario
    test_operations = [
        ('release', 0, 1000, "Initialize full capacity"),
        ('reserve', 100, 200, "Schedule first task"),
        ('reserve', 300, 450, "Schedule second task"),
        ('release', 125, 175, "Release partial overlap"),
        ('reserve', 600, 750, "Schedule third task"),
        ('release', 800, 900, "Add more capacity"),
    ]
    
    print("Test scenario:")
    for i, (op, start, end, description) in enumerate(test_operations):
        print(f"  {i+1}. {op.title()} [{start}, {end}): {description}")
    
    # Run on each available backend
    backend_results = {}
    
    for backend_id, backend_info in available_backends.items():
        print(f"\n🔧 Testing {backend_info.name}:")
        
        try:
            # Create tree with consistent seed
            tree = create_example_tree(backend_id, random_seed=42)
            
            # Apply operations
            for op, start, end, description in test_operations:
                if op == 'reserve':
                    tree.reserve_interval(start, end)
                else:
                    tree.release_interval(start, end)
            
            # Get final analytics
            analytics = get_tree_analytics(tree)
            backend_results[backend_id] = analytics
            
            print(f"  Final state:")
            print(f"    Total available: {analytics.get('total_available', 'N/A')}")
            print(f"    Intervals count: {analytics.get('tree_size', 'N/A')}")
            if 'utilization' in analytics:
                print(f"    Utilization: {analytics['utilization']:.1%}")
            if 'fragmentation' in analytics:
                print(f"    Fragmentation: {analytics['fragmentation']:.1%}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            backend_results[backend_id] = {'error': str(e)}
    
    # Compare results
    print(f"\n📊 Backend Results Comparison:")
    
    # Check consistency
    total_available_values = []
    for backend_id, result in backend_results.items():
        if 'total_available' in result:
            total_available_values.append(result['total_available'])
    
    if len(set(total_available_values)) <= 1:
        print("✅ All backends produced consistent results")
    else:
        print("⚠️  Results vary between backends:")
        for backend_id, result in backend_results.items():
            if 'total_available' in result:
                backend_name = available_backends[backend_id].name
                print(f"  {backend_name}: {result['total_available']} total available")


def demo_performance_comparison():
    """Compare performance across available backends"""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    manager = get_backend_manager()
    
    # Run built-in benchmark
    print("Running performance benchmark across all backends...")
    results = manager.benchmark_available_backends(num_operations=5000)
    
    if not results:
        print("❌ No backends available for benchmarking")
        return
    
    # Analyze results
    print(f"\n📈 Performance Results:")
    print(f"{'Backend':25} {'Ops/sec':>12} {'Language':>10} {'Tier':>15}")
    print("-" * 65)
    
    # Sort by performance
    sorted_results = sorted(
        [(bid, res) for bid, res in results.items() if 'ops_per_second' in res],
        key=lambda x: x[1]['ops_per_second'],
        reverse=True
    )
    
    baseline_performance = None
    
    for backend_id, result in sorted_results:
        backend_info = manager.available_backends[backend_id]
        ops_per_sec = result['ops_per_second']
        
        if baseline_performance is None:
            baseline_performance = ops_per_sec
            speedup_str = "baseline"
        else:
            speedup = ops_per_sec / baseline_performance if baseline_performance > 0 else 1.0
            speedup_str = f"{speedup:.1f}x"
        
        print(f"{backend_info.name:25} {ops_per_sec:>12,.0f} {backend_info.language:>10} "
              f"{backend_info.performance_tier:>15}")
        print(f"{'':25} {'(' + speedup_str + ')':>12}")
    
    # Show failed backends
    failed_backends = [(bid, res) for bid, res in results.items() if 'error' in res]
    if failed_backends:
        print(f"\n❌ Failed backends:")
        for backend_id, result in failed_backends:
            backend_info = manager.available_backends[backend_id]
            print(f"  {backend_info.name}: {result['error']}")


def demo_feature_comparison():
    """Compare features across different backends"""
    print("\n🎯 Feature Comparison Demo")
    print("=" * 50)
    
    manager = get_backend_manager()
    available = manager.get_available_backends()
    
    # Comprehensive feature test matrix
    features_to_test = [
        ('get_intervals', 'Basic interval retrieval'),
        ('find_interval', 'Find suitable interval'),
        ('find_best_fit', 'Best-fit allocation'),
        ('get_availability_stats', 'O(1) summary statistics'),
        ('sample_random_interval', 'Random sampling'),
        ('get_statistics', 'Tree structure statistics'),
        ('find_overlapping_intervals', 'Overlap queries'),
        ('split', 'Tree split operations'),
        ('merge_treap', 'Tree merge operations'),
        ('verify_treap_properties', 'Property verification'),
        ('get_rank', 'Rank operations'),
        ('find_largest_available', 'Largest block finder'),
        ('get_tree_summary', 'Comprehensive summaries'),
    ]
    
    # Create better column headers with model names
    backend_info = []
    for backend_id, info in available.items():
        # Extract meaningful model name
        if 'summary' in backend_id.lower():
            model_name = "Summary"
        elif 'treap' in backend_id.lower():
            model_name = "Treap" 
        elif 'boundary' in backend_id.lower():
            model_name = "Boundary"
        elif 'avl' in backend_id.lower():
            model_name = "AVL"
        elif 'ic' in backend_id.lower():
            model_name = "IC"
        else:
            model_name = info.name.split()[1] if len(info.name.split()) > 1 else info.name.split()[0]
        
        # Add language suffix for clarity
        lang_suffix = "Py" if info.language == "Python" else "C++"
        display_name = f"{model_name}-{lang_suffix}"
        
        backend_info.append((backend_id, display_name, info))
    
    # Print header
    print(f"{'Feature':30} ", end="")
    for _, display_name, _ in backend_info:
        print(f"{display_name:>12}", end="")
    print()
    
    print("-" * (30 + 12 * len(backend_info)))
    
    # Test each feature using improved detection
    for method_name, description in features_to_test:
        print(f"{description:30} ", end="")
        
        for backend_id, display_name, info in backend_info:
            try:
                tree = create_example_tree(backend_id, random_seed=42)
                tree_features = detect_tree_features(tree)
                
                # Map method names to feature names from detect_tree_features
                feature_mapping = {
                    'get_intervals': 'basic_intervals',
                    'find_interval': 'find_operations', 
                    'find_best_fit': 'best_fit_allocation',
                    'get_availability_stats': 'summary_statistics',
                    'sample_random_interval': 'random_sampling',
                    'get_statistics': 'tree_statistics',
                    'find_overlapping_intervals': 'overlap_queries',
                    'split': 'split_operations',
                    'merge_treap': 'merge_operations',
                    'verify_treap_properties': 'property_verification',
                    'get_rank': 'rank_operations',
                    'find_largest_available': 'largest_block_finder',
                    'get_tree_summary': 'comprehensive_summaries',
                }
                
                feature_name = feature_mapping.get(method_name)
                has_feature = feature_name in tree_features if feature_name else hasattr(tree, method_name)
                
                # Special fallback handling
                if not has_feature and method_name == 'find_best_fit':
                    # Some implementations might have find_interval as alternative
                    has_feature = 'find_operations' in tree_features
                
                status = "✅" if has_feature else "❌"
                print(f"{status:>12}", end="")
                
            except Exception as e:
                print(f"{'❌':>12}", end="")
        print()
    
    # Additional comparison: Performance tiers
    print(f"\n{'Performance Tier':30} ", end="")
    for _, display_name, info in backend_info:
        tier_short = info.performance_tier.split('_')[0].capitalize()[:8]
        print(f"{tier_short:>12}", end="")
    print()
    
    print(f"{'Est. Speedup':30} ", end="")
    for _, display_name, info in backend_info:
        speedup = f"{info.estimated_speedup:.1f}x"
        print(f"{speedup:>12}", end="")
    print()
    
    print(f"\nLegend:")
    print(f"  ✅ = Feature available")
    print(f"  ❌ = Feature not available") 
    print(f"  Performance tiers: Baseline < Optimized < High")
    print(f"  Speedup relative to Python baseline")


def demo_algorithm_with_backend_switching():
    """Demonstrate algorithm running with different backends"""
    print("\n🧮 Algorithm with Backend Switching")
    print("=" * 50)
    
    def simple_scheduling_algorithm(backend_id: str, tasks: list) -> dict:
        """Simple scheduling algorithm that works with any backend"""
        tree = create_example_tree(backend_id, random_seed=42)
        tree.release_interval(0, 1000)  # Available time
        
        scheduled = []
        rejected = []
        
        for task_id, duration, priority in tasks:
            # Try to find suitable slot
            try:
                if hasattr(tree, 'find_best_fit'):
                    # Use advanced best-fit if available
                    result = tree.find_best_fit(duration)
                    if result:
                        start, end = result[0], result[0] + duration
                        tree.reserve_interval(start, end)
                        scheduled.append((task_id, start, end))
                    else:
                        rejected.append(task_id)
                elif hasattr(tree, 'find_interval'):
                    # Use basic find_interval
                    result = tree.find_interval(0, duration)
                    start, end = result
                    tree.reserve_interval(start, end)
                    scheduled.append((task_id, start, end))
                else:
                    # Very basic: just allocate sequentially
                    analytics = get_tree_analytics(tree)
                    if analytics.get('total_available', 0) >= duration:
                        # Simplified allocation
                        start = 0  # Would need more logic for real implementation
                        end = start + duration
                        tree.reserve_interval(start, end)
                        scheduled.append((task_id, start, end))
                    else:
                        rejected.append(task_id)
                        
            except ValueError:
                rejected.append(task_id)
        
        # Get final analytics
        final_analytics = get_tree_analytics(tree)
        
        return {
            'scheduled': scheduled,
            'rejected': rejected,
            'final_analytics': final_analytics
        }
    
    # Test tasks
    test_tasks = [
        (1, 50, 3),   # (task_id, duration, priority)
        (2, 30, 1),
        (3, 80, 2),
        (4, 40, 3),
        (5, 60, 1),
    ]
    
    print("Test tasks:")
    for task_id, duration, priority in test_tasks:
        print(f"  Task {task_id}: duration={duration}, priority={priority}")
    
    # Run algorithm on each available backend
    manager = get_backend_manager()
    available = manager.get_available_backends()
    
    for backend_id, backend_info in available.items():
        print(f"\n{backend_info.name} results:")
        
        try:
            result = simple_scheduling_algorithm(backend_id, test_tasks)
            
            print(f"  Scheduled: {len(result['scheduled'])} tasks")
            print(f"  Rejected: {len(result['rejected'])} tasks")
            
            analytics = result['final_analytics']
            if 'utilization' in analytics:
                print(f"  Final utilization: {analytics['utilization']:.1%}")
            if 'total_available' in analytics:
                print(f"  Remaining capacity: {analytics['total_available']}")
                
        except Exception as e:
            print(f"  ❌ Algorithm failed: {e}")


def main():
    """Run backend comparison demonstrations"""
    # Parse command line arguments
    args = parse_backend_args("Tree-Mendous Backend Comparison Demo")
    
    # Handle special commands
    selected_backend = handle_backend_args(args)
    if selected_backend is None:
        return
    
    print("🔧 Tree-Mendous Backend Comparison Demo")
    print("Demonstrating seamless switching between Python and C++ implementations")
    print("=" * 75)
    
    # Show selected backend
    manager = get_backend_manager()
    backend_info = manager.available_backends[selected_backend]
    print(f"Selected backend: {backend_info.name} ({backend_info.language})")
    print(f"Performance tier: {backend_info.performance_tier}")
    print(f"Estimated speedup: {backend_info.estimated_speedup:.1f}x")
    
    # Run all demos
    demo_same_operations_different_backends()
    demo_performance_comparison()
    demo_feature_comparison()
    demo_algorithm_with_backend_switching()
    
    print("\n" + "=" * 75)
    print("✅ Backend comparison demo complete!")
    print("\n🎯 Key capabilities demonstrated:")
    print("  • Seamless switching between Python and C++ implementations")
    print("  • Unified interface regardless of backend choice")
    print("  • Performance comparison with automatic benchmarking")
    print("  • Feature compatibility testing across implementations")
    print("  • Algorithm adaptation to available backend capabilities")
    print("\n💡 Usage tips:")
    print("  • Use --backend=auto for best available implementation")
    print("  • Use --backend=cpp_summary for maximum performance")
    print("  • Use --backend=py_treap for probabilistic balancing")
    print("  • Use --list-backends to see all available options")
    print("  • Use --benchmark-backends for performance comparison")


if __name__ == "__main__":
    main()
