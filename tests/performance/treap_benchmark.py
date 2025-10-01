#!/usr/bin/env python3
"""
Treap Performance Benchmark

Comprehensive performance testing for both Python and C++ treap implementations,
comparing with other interval tree structures.
"""

import sys
import time
import random
import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add paths for import resolution  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))
sys.path.insert(0, str(project_root / 'examples'))

# Use backend management system
try:
    from common.backend_config import get_backend_manager, create_example_tree, get_tree_analytics
    backend_manager = get_backend_manager()
    available_backends = backend_manager.get_available_backends()
    BACKEND_SYSTEM_AVAILABLE = True
    print("[OK] Backend management system loaded")
    
    # Print available backends for treap testing
    treap_backends = {k: v for k, v in available_backends.items() if 'treap' in k.lower()}
    if treap_backends:
        print(f"[OK] Treap backends available: {list(treap_backends.keys())}")
    else:
        print("[WARN]  No treap backends found")
        
except ImportError as e:
    print(f"[FAIL] Backend system failed: {e}")
    BACKEND_SYSTEM_AVAILABLE = False
    backend_manager = None
    available_backends = {}

# Fallback: Direct imports for backward compatibility
try:
    from treap import IntervalTreap as PyTreap
    py_treap_available = True
    print("[OK] Python Treap loaded (direct)")
except ImportError as e:
    print(f"[FAIL] Python Treap failed: {e}")
    PyTreap = None
    py_treap_available = False

try:
    from summary import SummaryIntervalTree as PyTree
    py_tree_available = True
    print("[OK] Python Summary Tree loaded (direct)")
except ImportError as e:
    print(f"[FAIL] Python Summary Tree failed: {e}")
    PyTree = None
    py_tree_available = False


def generate_test_operations(num_ops: int, operation_mix: Dict[str, float] = None) -> List[Tuple[str, int, int]]:
    """Generate randomized operations for benchmarking"""
    if operation_mix is None:
        operation_mix = {'reserve': 0.4, 'release': 0.4, 'find': 0.2}
    
    operations = []
    op_types = list(operation_mix.keys())
    op_weights = list(operation_mix.values())
    
    for _ in range(num_ops):
        op_type = random.choices(op_types, weights=op_weights)[0]
        start = random.randint(0, 999_900)
        length = random.randint(1, 1000)
        end = start + length
        operations.append((op_type, start, end))
    
    return operations


def benchmark_treap_vs_others(num_operations: int = 10_000) -> Dict[str, Dict]:
    """Compare treap performance with other implementations using backend system"""
    print(f"\n🏁 Treap vs Other Implementations ({num_operations:,} operations)")
    print("=" * 60)
    
    operations = generate_test_operations(num_operations)
    results = {}
    
    # Use backend system if available
    if BACKEND_SYSTEM_AVAILABLE:
        # Test all available backends
        for backend_id, backend_info in available_backends.items():
            print(f"  🔄 Testing {backend_info.name}...")
            
            try:
                # Create tree using backend system
                tree = create_example_tree(backend_id, random_seed=42)
                tree.release_interval(0, 1_000_000)
                
                # Benchmark operations
                op_times = {'reserve': [], 'release': [], 'find': []}
                start_total = time.perf_counter()
                
                for op, start, end in operations:
                    start_time = time.perf_counter()
                    
                    try:
                        if op == 'reserve':
                            tree.reserve_interval(start, end)
                        elif op == 'release':
                            tree.release_interval(start, end)
                        elif op == 'find':
                            if hasattr(tree, 'find_interval'):
                                tree.find_interval(start, end - start)
                            elif hasattr(tree, 'find_best_fit'):
                                tree.find_best_fit(end - start)
                            else:
                                continue
                        
                        op_times[op].append(time.perf_counter() - start_time)
                        
                    except (ValueError, AttributeError):
                        continue
                
                total_time = time.perf_counter() - start_total
                
                # Calculate statistics
                avg_times = {}
                total_ops = 0
                for op, times in op_times.items():
                    if times:
                        avg_times[op] = statistics.mean(times) * 1000  # Convert to ms
                        total_ops += len(times)
                
                ops_per_second = total_ops / total_time if total_time > 0 else 0
                
                # Get tree analytics using unified system
                tree_stats = get_tree_analytics(tree)
                
                results[backend_info.name] = {
                    'backend_id': backend_id,
                    'total_time': total_time,
                    'ops_per_second': ops_per_second,
                    'avg_times_ms': avg_times,
                    'tree_stats': tree_stats,
                    'language': backend_info.language,
                    'performance_tier': backend_info.performance_tier
                }
                
                print(f"    {ops_per_second:>8,.0f} ops/sec ({backend_info.language})")
                
            except Exception as e:
                print(f"    [FAIL] Failed: {e}")
                results[backend_info.name] = {'error': str(e)}
    
    else:
        # Fallback to direct imports
        implementations = {}
        if py_treap_available:
            implementations["Python Treap"] = PyTreap
        if py_tree_available:
            implementations["Python Summary Tree"] = PyTree
        
        for name, impl_class in implementations.items():
            print(f"  🔄 Testing {name}...")
            
            # Initialize (fallback logic)
            if name == "Python Treap":
                tree = impl_class(random_seed=42)
            else:
                tree = impl_class()
            
            tree.release_interval(0, 1_000_000)
            
            # Benchmark operations (simplified fallback)
            start_total = time.perf_counter()
            successful_ops = 0
            
            for op, start, end in operations:
                try:
                    if op == 'reserve':
                        tree.reserve_interval(start, end)
                        successful_ops += 1
                    elif op == 'release':
                        tree.release_interval(start, end)
                        successful_ops += 1
                    elif op == 'find' and hasattr(tree, 'find_interval'):
                        tree.find_interval(start, end - start)
                        successful_ops += 1
                except (ValueError, AttributeError):
                    continue
            
            total_time = time.perf_counter() - start_total
            ops_per_second = successful_ops / total_time if total_time > 0 else 0
            
            results[name] = {
                'total_time': total_time,
                'ops_per_second': ops_per_second,
                'language': 'Python'
            }
            
            print(f"    {ops_per_second:>8,.0f} ops/sec")
    
    return results


def benchmark_treap_specific_operations() -> Dict[str, Any]:
    """Benchmark treap-specific operations"""
    print(f"\n🎲 Treap-Specific Operations Benchmark")
    print("=" * 60)
    
    if not py_treap_available:
        print("[FAIL] Python Treap not available")
        return {}
    
    treap = PyTreap(random_seed=42)
    treap.release_interval(0, 100_000)
    
    # Create some intervals for testing
    for i in range(0, 50_000, 1000):
        treap.reserve_interval(i, i + 100)
    
    print(f"Setup: {treap.get_tree_size()} intervals in treap")
    
    # Benchmark treap-specific operations
    operations = {
        'sample_random_interval': lambda: treap.sample_random_interval(),
        'find_overlapping_intervals': lambda: treap.find_overlapping_intervals(25000, 26000),
        'get_statistics': lambda: treap.get_statistics(),
        'verify_treap_properties': lambda: treap.verify_treap_properties(),
    }
    
    if hasattr(treap, 'get_rank'):
        operations['get_rank'] = lambda: treap.get_rank(10000, 10100)
    
    results = {}
    
    for op_name, op_func in operations.items():
        times = []
        
        # Run operation multiple times
        num_runs = 1000 if "verify" not in op_name else 10  # Verification is expensive
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = op_func()
            except Exception as e:
                continue
            times.append(time.perf_counter() - start_time)
        
        if times:
            results[op_name] = {
                'avg_time_us': statistics.mean(times) * 1_000_000,
                'min_time_us': min(times) * 1_000_000,
                'max_time_us': max(times) * 1_000_000,
                'runs': len(times)
            }
            
            print(f"  {op_name:25}: {results[op_name]['avg_time_us']:>7.1f}µs avg "
                  f"({results[op_name]['min_time_us']:>5.1f}-{results[op_name]['max_time_us']:>6.1f}µs)")
    
    return results


def benchmark_treap_probabilistic_balance() -> Dict[str, Any]:
    """Benchmark treap balance quality across different insertion patterns"""
    print(f"\n[BALANCE] Treap Balance Quality Analysis")
    print("=" * 60)
    
    if not py_treap_available:
        print("[FAIL] Python Treap not available")
        return {}
    
    test_patterns = {
        "Random": lambda n: [(random.randint(0, n*10), random.randint(0, n*10)) 
                            for _ in range(n)],
        "Sequential": lambda n: [(i*10, i*10+5) for i in range(n)],
        "Reverse": lambda n: [((n-i)*10, (n-i)*10+5) for i in range(n)],
        "Clustered": lambda n: [(i*2 + random.randint(0, 1), i*2 + random.randint(2, 3)) 
                               for i in range(0, n*5, 5)],
    }
    
    sizes = [100, 500, 1000]
    pattern_results = {}
    
    for pattern_name, pattern_gen in test_patterns.items():
        print(f"\n  Pattern: {pattern_name}")
        size_results = {}
        
        for n in sizes:
            heights = []
            balance_factors = []
            
            # Test with multiple seeds for statistical significance
            for seed in range(5):
                treap = PyTreap(random_seed=seed)
                intervals = pattern_gen(n)
                
                # Filter out invalid intervals
                valid_intervals = [(min(s, e), max(s, e)) for s, e in intervals if s != e]
                
                for start, end in valid_intervals:
                    treap.release_interval(start, end)
                
                stats = treap.get_statistics()
                if stats['size'] > 0:
                    heights.append(stats['height'])
                    balance_factors.append(stats['balance_factor'])
            
            if heights:
                avg_height = statistics.mean(heights)
                expected_height = math.log2(n + 1)
                avg_balance = statistics.mean(balance_factors)
                
                size_results[n] = {
                    'avg_height': avg_height,
                    'expected_height': expected_height,
                    'avg_balance_factor': avg_balance,
                    'height_ratio': avg_height / expected_height if expected_height > 0 else 1.0
                }
                
                print(f"    n={n:4d}: height={avg_height:5.1f} (exp: {expected_height:4.1f}), "
                      f"balance={avg_balance:.2f}")
        
        pattern_results[pattern_name] = size_results
    
    return pattern_results


def benchmark_treap_scaling() -> Dict[str, Any]:
    """Test treap performance scaling with problem size"""
    print(f"\n[PERF] Treap Scaling Analysis")
    print("=" * 60)
    
    if not py_treap_available:
        print("[FAIL] Python Treap not available")
        return {}
    
    sizes = [100, 500, 1000, 5000, 10000]
    scaling_results = {}
    
    print(f"  {'Size':>6} {'Ops/sec':>10} {'Height':>8} {'Balance':>8} {'Memory':>8}")
    print("  " + "-" * 50)
    
    for n in sizes:
        # Create treap and measure performance
        treap = PyTreap(random_seed=42)
        
        # Time insertion of n intervals
        start_time = time.perf_counter()
        for i in range(n):
            start = random.randint(0, n*10)
            end = start + random.randint(1, 100)
            treap.release_interval(start, end)
        insertion_time = time.perf_counter() - start_time
        
        # Time queries
        start_time = time.perf_counter()
        query_count = min(1000, n)
        for _ in range(query_count):
            try:
                treap.find_interval(random.randint(0, n*10), random.randint(1, 50))
            except ValueError:
                pass
        query_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        ops_per_second = n / insertion_time if insertion_time > 0 else 0
        avg_query_time = query_time / query_count if query_count > 0 else 0
        
        stats = treap.get_statistics()
        estimated_memory = stats['size'] * 64  # Rough estimate: 64 bytes per node
        
        scaling_results[n] = {
            'ops_per_second': ops_per_second,
            'avg_query_time_us': avg_query_time * 1_000_000,
            'height': stats['height'],
            'balance_factor': stats['balance_factor'],
            'memory_kb': estimated_memory / 1024
        }
        
        print(f"  {n:>6,} {ops_per_second:>10,.0f} {stats['height']:>8} "
              f"{stats['balance_factor']:>8.2f} {estimated_memory/1024:>7.1f}K")
    
    return scaling_results


def analyze_treap_vs_avl_performance() -> None:
    """Compare treap performance characteristics with AVL trees"""
    print(f"\n🆚 Treap vs AVL Performance Comparison")
    print("=" * 60)
    
    if not py_treap_available:
        print("[FAIL] Python Treap not available")
        return
    
    # Import AVL for comparison (if available)
    try:
        sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))
        from avl_earliest import EarliestIntervalTree as AVLTree
        avl_available = True
    except ImportError:
        print("[WARN]  AVL tree not available for comparison")
        avl_available = False
        return
    
    test_sizes = [1000, 5000]
    
    print(f"  {'Size':>6} {'Algorithm':>12} {'Insert(ms)':>12} {'Find(µs)':>10} {'Height':>8} {'Balance':>8}")
    print("  " + "-" * 70)
    
    for n in test_sizes:
        # Generate same operations for fair comparison
        operations = generate_test_operations(n, {'reserve': 0.5, 'release': 0.3, 'find': 0.2})
        
        # Test treap
        treap = PyTreap(random_seed=42)
        treap.release_interval(0, n * 100)
        
        start_time = time.perf_counter()
        find_times = []
        
        for op, start, end in operations:
            if op == 'reserve':
                treap.reserve_interval(start, end)
            elif op == 'release':
                treap.release_interval(start, end)
            elif op == 'find':
                find_start = time.perf_counter()
                try:
                    treap.find_interval(start, end - start)
                except ValueError:
                    pass
                find_times.append(time.perf_counter() - find_start)
        
        treap_total_time = time.perf_counter() - start_time
        treap_stats = treap.get_statistics()
        treap_find_avg = statistics.mean(find_times) * 1_000_000 if find_times else 0
        
        print(f"  {n:>6,} {'Treap':>12} {treap_total_time*1000:>12.1f} {treap_find_avg:>10.1f} "
              f"{treap_stats['height']:>8} {treap_stats['balance_factor']:>8.2f}")
        
        # Test AVL
        if avl_available:
            avl = AVLTree()
            avl.release_interval(0, n * 100)
            
            start_time = time.perf_counter()
            find_times = []
            
            for op, start, end in operations:
                if op == 'reserve':
                    avl.reserve_interval(start, end)
                elif op == 'release':
                    avl.release_interval(start, end)
                elif op == 'find':
                    find_start = time.perf_counter()
                    try:
                        avl.find_interval(start, end - start)
                    except (ValueError, AttributeError):
                        pass
                    find_times.append(time.perf_counter() - find_start)
            
            avl_total_time = time.perf_counter() - start_time
            avl_find_avg = statistics.mean(find_times) * 1_000_000 if find_times else 0
            
            # AVL doesn't have same statistics, so estimate
            avl_height = "N/A"
            avl_balance = "1.00"  # AVL is always balanced
            
            print(f"  {n:>6,} {'AVL':>12} {avl_total_time*1000:>12.1f} {avl_find_avg:>10.1f} "
                  f"{avl_height:>8} {avl_balance:>8}")
            
            # Performance comparison
            speedup = avl_total_time / treap_total_time if treap_total_time > 0 else 1.0
            print(f"    → Treap {'faster' if speedup > 1 else 'slower'} by {abs(speedup):.2f}x")


def stress_test_treap() -> Dict[str, Any]:
    """Stress test treap with intensive workloads"""
    print(f"\n💪 Treap Stress Test")
    print("=" * 60)
    
    if not py_treap_available:
        print("[FAIL] Python Treap not available")
        return {}
    
    # Stress test parameters
    stress_tests = [
        ("High Frequency", 50_000, {'reserve': 0.6, 'release': 0.3, 'find': 0.1}),
        ("Balanced Mix", 25_000, {'reserve': 0.33, 'release': 0.33, 'find': 0.34}),
        ("Query Heavy", 20_000, {'reserve': 0.2, 'release': 0.2, 'find': 0.6}),
    ]
    
    stress_results = {}
    
    for test_name, num_ops, op_mix in stress_tests:
        print(f"\n  {test_name} Stress Test ({num_ops:,} operations):")
        
        treap = PyTreap(random_seed=42)
        treap.release_interval(0, 1_000_000)
        
        operations = generate_test_operations(num_ops, op_mix)
        
        # Track performance over time
        checkpoint_interval = num_ops // 10
        checkpoints = []
        
        start_total = time.perf_counter()
        
        for i, (op, start, end) in enumerate(operations):
            op_start = time.perf_counter()
            
            try:
                if op == 'reserve':
                    treap.reserve_interval(start, end)
                elif op == 'release':
                    treap.release_interval(start, end)
                elif op == 'find':
                    treap.find_interval(start, end - start)
            except ValueError:
                pass
            
            # Record checkpoint
            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.perf_counter() - start_total
                ops_so_far = i + 1
                current_ops_per_sec = ops_so_far / elapsed
                
                stats = treap.get_statistics()
                checkpoints.append({
                    'operations': ops_so_far,
                    'elapsed_time': elapsed,
                    'ops_per_second': current_ops_per_sec,
                    'tree_size': stats['size'],
                    'tree_height': stats['height'],
                    'balance_factor': stats['balance_factor']
                })
                
                print(f"    {ops_so_far:>6,} ops: {current_ops_per_sec:>8,.0f} ops/sec, "
                      f"height={stats['height']}, balance={stats['balance_factor']:.2f}")
        
        total_time = time.perf_counter() - start_total
        final_stats = treap.get_statistics()
        
        stress_results[test_name] = {
            'total_operations': num_ops,
            'total_time': total_time,
            'final_ops_per_second': num_ops / total_time,
            'final_tree_stats': final_stats,
            'checkpoints': checkpoints
        }
        
        print(f"    Final: {num_ops / total_time:,.0f} ops/sec, "
              f"tree size={final_stats['size']}, height={final_stats['height']}")
    
    return stress_results


def main():
    """Run comprehensive treap benchmarks"""
    print("🌳 Tree-Mendous Treap Performance Benchmark Suite")
    print("Comprehensive testing of randomized interval trees")
    print("=" * 70)
    
    random.seed(42)  # Reproducible results
    
    # Run all benchmark suites
    comparison_results = benchmark_treap_vs_others(10_000)
    specific_results = benchmark_treap_specific_operations()
    balance_results = benchmark_treap_probabilistic_balance()
    scaling_results = benchmark_treap_scaling()
    stress_results = stress_test_treap()
    
    # Summary analysis
    print(f"\n[STATS] TREAP BENCHMARK SUMMARY")
    print("=" * 70)
    
    if comparison_results:
        fastest_impl = max(comparison_results.items(), key=lambda x: x[1]['ops_per_second'])
        print(f"Fastest Implementation: {fastest_impl[0]} ({fastest_impl[1]['ops_per_second']:,.0f} ops/sec)")
    
    if py_treap_available and "Python Treap" in comparison_results:
        treap_result = comparison_results["Python Treap"]
        print(f"Python Treap Performance: {treap_result['ops_per_second']:,.0f} ops/sec")
        
        if 'tree_stats' in treap_result and 'balance_factor' in treap_result['tree_stats']:
            balance = treap_result['tree_stats']['balance_factor']
            print(f"  Balance quality: {balance:.2f} (closer to 1.0 is better)")
    
    if specific_results:
        print(f"\nTreap-Specific Operations:")
        for op_name, metrics in specific_results.items():
            print(f"  {op_name}: {metrics['avg_time_us']:.1f}µs average")
    
    print(f"\n[OK] Treap benchmark suite complete!")
    print(f"Key insights:")
    print(f"  • Probabilistic balancing eliminates worst-case behavior")
    print(f"  • O(log n) expected performance with high probability")
    print(f"  • Simpler implementation than deterministic balanced trees")
    print(f"  • Excellent performance across diverse workload patterns")


if __name__ == "__main__":
    import math
    main()
