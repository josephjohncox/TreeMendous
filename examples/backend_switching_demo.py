#!/usr/bin/env python3
"""
Backend Switching Demo

Simple demonstration of how to switch between Python and C++ implementations
while maintaining identical functionality through unified interfaces.

Usage:
  python backend_switching_demo.py                    # Auto-select best backend
  python backend_switching_demo.py --backend=py_treap # Use Python treap
  python backend_switching_demo.py --backend=cpp_summary # Use C++ summary tree
  python backend_switching_demo.py --list-backends    # Show available backends
  python backend_switching_demo.py --benchmark-backends # Compare all backends
"""

import sys
import time
import random
from pathlib import Path

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common.backend_config import (
    parse_backend_args, handle_backend_args, create_example_tree, 
    get_tree_analytics, get_backend_manager
)


def demo_simple_scheduling(backend_id: str, verbose: bool = False):
    """Simple scheduling demo using specified backend"""
    print(f"\n📅 Simple Scheduling Demo")
    print(f"Backend: {get_backend_manager().available_backends[backend_id].name}")
    print("=" * 50)
    
    # Create tree using specified backend
    tree = create_example_tree(backend_id, random_seed=42)
    
    if verbose:
        print(f"✅ Created {get_backend_manager().available_backends[backend_id].name}")
    
    # Simple scheduling scenario: meeting room for the day
    tree.release_interval(0, 1440)  # 24 hours = 1440 minutes
    
    meetings = [
        ("Team Standup", 30),      # 30 minutes
        ("Client Call", 90),       # 1.5 hours  
        ("Planning Meeting", 120), # 2 hours
        ("1-on-1 Review", 60),     # 1 hour
        ("Project Demo", 45),      # 45 minutes
    ]
    
    print("Scheduling meetings for the day:")
    scheduled_meetings = []
    
    for meeting_name, duration in meetings:
        start_time = time.perf_counter()
        
        try:
            # Try different allocation methods based on backend capabilities
            if hasattr(tree, 'find_best_fit'):
                # Use advanced best-fit if available
                result = tree.find_best_fit(duration)
                if result:
                    start, end = result[0], result[0] + duration
                    tree.reserve_interval(start, end)
                    scheduled_meetings.append((meeting_name, start, end))
                    
                    start_hour = start // 60
                    start_min = start % 60
                    end_hour = end // 60
                    end_min = end % 60
                    
                    print(f"  ✅ {meeting_name}: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}")
                else:
                    print(f"  ❌ {meeting_name}: No suitable time slot")
                    
            elif hasattr(tree, 'find_interval'):
                # Use basic find_interval
                result = tree.find_interval(0, duration)
                start, end = result[0], result[0] + duration
                tree.reserve_interval(start, end)
                scheduled_meetings.append((meeting_name, start, end))
                
                start_hour = start // 60
                start_min = start % 60
                end_hour = end // 60
                end_min = end % 60
                
                print(f"  ✅ {meeting_name}: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}")
            else:
                print(f"  ⚠️  {meeting_name}: Backend doesn't support interval finding")
                
        except ValueError:
            print(f"  ❌ {meeting_name}: No available time slot")
        
        operation_time = time.perf_counter() - start_time
        if verbose:
            print(f"      (Operation took {operation_time*1000:.2f}ms)")
    
    # Get final analytics
    analytics = get_tree_analytics(tree)
    
    print(f"\n📊 Final Schedule Analysis:")
    print(f"  Meetings scheduled: {len(scheduled_meetings)}/{len(meetings)}")
    
    if 'utilization' in analytics:
        print(f"  Room utilization: {analytics['utilization']:.1%}")
    if 'fragmentation' in analytics:
        print(f"  Schedule fragmentation: {analytics['fragmentation']:.1%}")
    if 'free_chunks' in analytics:
        print(f"  Free time blocks: {analytics['free_chunks']}")
    if 'largest_chunk' in analytics:
        print(f"  Largest free block: {analytics['largest_chunk']} minutes")
    
    return len(scheduled_meetings), analytics


def demo_performance_comparison(backends_to_test: list = None):
    """Compare performance across multiple backends"""
    print(f"\n⚡ Performance Comparison Across Backends")
    print("=" * 50)
    
    manager = get_backend_manager()
    available = manager.get_available_backends()
    
    if backends_to_test is None:
        backends_to_test = list(available.keys())
    
    # Filter to only available backends
    backends_to_test = [b for b in backends_to_test if b in available]
    
    if len(backends_to_test) < 2:
        print("⚠️  Need at least 2 backends for meaningful comparison")
        return
    
    # Performance test: many operations
    num_operations = 5000
    print(f"Testing {num_operations:,} operations per backend...")
    
    # Generate consistent workload
    random.seed(42)
    operations = []
    for _ in range(num_operations):
        op = random.choice(['reserve', 'release'])
        start = random.randint(0, 10000)
        duration = random.randint(10, 100)
        operations.append((op, start, start + duration))
    
    results = {}
    
    print(f"\n{'Backend':20} {'Ops/sec':>10} {'Final Size':>12} {'Features':>10}")
    print("-" * 54)
    
    for backend_id in backends_to_test:
        backend_info = available[backend_id]
        
        try:
            # Create and initialize tree
            tree = create_example_tree(backend_id, random_seed=42)
            tree.release_interval(0, 20000)  # Large initial space
            
            # Benchmark operations
            start_time = time.perf_counter()
            
            for op, start, end in operations:
                try:
                    if op == 'reserve':
                        tree.reserve_interval(start, end)
                    else:
                        tree.release_interval(start, end)
                except (ValueError, AttributeError):
                    continue
            
            total_time = time.perf_counter() - start_time
            ops_per_second = num_operations / total_time if total_time > 0 else 0
            
            # Get analytics
            analytics = get_tree_analytics(tree)
            final_size = analytics.get('tree_size', 0)
            
            # Count features
            feature_count = len([f for f in ['find_best_fit', 'get_availability_stats', 
                               'sample_random_interval', 'get_statistics'] 
                               if hasattr(tree, f)])
            
            results[backend_id] = {
                'ops_per_second': ops_per_second,
                'final_size': final_size,
                'feature_count': feature_count,
                'analytics': analytics
            }
            
            # Extract model name for display
            if 'summary' in backend_id:
                model_display = "Summary"
            elif 'treap' in backend_id:
                model_display = "Treap"
            elif 'boundary' in backend_id:
                model_display = "Boundary"
            elif 'avl' in backend_id:
                model_display = "AVL"
            elif 'ic' in backend_id:
                model_display = "IC"
            else:
                model_display = backend_info.name.split()[0]
            
            lang = "C++" if backend_info.language == "C++" else "Py"
            display_name = f"{model_display}-{lang}"
            
            print(f"{display_name:20} {ops_per_second:>10,.0f} {final_size:>12,} {feature_count:>10}")
            
        except Exception as e:
            print(f"{backend_info.name:20} {'ERROR':>10} {'N/A':>12} {'N/A':>10}")
            if hasattr(e, '__str__'):
                print(f"  Error: {e}")
    
    # Performance analysis
    if len(results) > 1:
        print(f"\n📈 Performance Analysis:")
        fastest = max(results.items(), key=lambda x: x[1]['ops_per_second'])
        slowest = min(results.items(), key=lambda x: x[1]['ops_per_second'])
        
        fastest_info = available[fastest[0]]
        slowest_info = available[slowest[0]]
        
        speedup = fastest[1]['ops_per_second'] / slowest[1]['ops_per_second']
        
        print(f"  Fastest: {fastest_info.name} ({fastest[1]['ops_per_second']:,.0f} ops/sec)")
        print(f"  Slowest: {slowest_info.name} ({slowest[1]['ops_per_second']:,.0f} ops/sec)")
        print(f"  Speedup ratio: {speedup:.1f}x")


def demo_feature_showcase_by_backend():
    """Showcase unique features of each backend"""
    print(f"\n🌟 Feature Showcase by Backend")
    print("=" * 50)
    
    manager = get_backend_manager()
    available = manager.get_available_backends()
    
    for backend_id, info in available.items():
        print(f"\n{info.name} ({info.language}):")
        
        try:
            tree = create_example_tree(backend_id, random_seed=42)
            tree.release_interval(0, 1000)
            tree.reserve_interval(100, 200)  # Create some allocation
            tree.reserve_interval(300, 400)
            
            # Showcase unique features
            if hasattr(tree, 'get_availability_stats'):
                stats = tree.get_availability_stats()
                print(f"  📊 Summary Statistics (O(1)):")
                print(f"    Utilization: {stats['utilization']:.1%}")
                print(f"    Fragmentation: {stats['fragmentation']:.1%}")
                print(f"    Free chunks: {stats['free_chunks']}")
            
            if hasattr(tree, 'sample_random_interval'):
                samples = []
                for _ in range(3):
                    sample = tree.sample_random_interval()
                    if sample:
                        samples.append(sample)
                print(f"  🎲 Random Sampling: {samples}")
            
            if hasattr(tree, 'find_overlapping_intervals'):
                overlaps = tree.find_overlapping_intervals(150, 350)
                print(f"  🔍 Overlap Detection [150,350): {overlaps}")
            
            if hasattr(tree, 'get_statistics'):
                stats = tree.get_statistics()
                if isinstance(stats, dict):
                    if 'balance_factor' in stats:
                        print(f"  ⚖️  Tree Balance: {stats['balance_factor']:.2f}")
                    if 'height' in stats:
                        print(f"  📏 Tree Height: {stats['height']}")
                else:
                    # C++ statistics object
                    print(f"  📏 Tree Height: {getattr(stats, 'height', 'N/A')}")
                    print(f"  ⚖️  Balance Factor: {getattr(stats, 'balance_factor', 'N/A')}")
            
            if hasattr(tree, 'find_best_fit'):
                try:
                    best_fit = tree.find_best_fit(50)
                    print(f"  🎯 Best Fit (50 units): {best_fit}")
                except ValueError:
                    print(f"  🎯 Best Fit (50 units): No suitable slot")
            
            if hasattr(tree, 'split'):
                print(f"  ✂️  Supports tree split/merge operations")
            
        except Exception as e:
            print(f"  ❌ Error showcasing features: {e}")


def main():
    """Main demonstration function"""
    print("🔄 Tree-Mendous Backend Switching System")
    print("Seamless switching between Python and C++ implementations")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_backend_args("Backend Switching Demo")
    
    # Handle special arguments
    selected_backend = handle_backend_args(args)
    if selected_backend is None:
        return
    
    print(f"Selected backend: {get_backend_manager().available_backends[selected_backend].name}")
    
    # Run demonstrations
    demo_simple_scheduling(selected_backend, args.verbose)
    demo_performance_comparison()
    demo_feature_showcase_by_backend()
    
    print(f"\n✅ Backend switching demonstrations complete!")
    print(f"Key benefits:")
    print(f"  • Seamless switching between implementations")
    print(f"  • Identical API across all backends")
    print(f"  • Performance optimization without code changes")
    print(f"  • Feature detection and graceful fallbacks")


if __name__ == "__main__":
    main()
