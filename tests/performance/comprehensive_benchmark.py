#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for Tree-Mendous Implementation Hierarchy

Tests all implementations: Simple Python, Summary Python, Simple C++, Summary C++
Measures both basic operations and advanced summary-enhanced operations.
"""

from typing import List, Tuple, Dict, Any, Optional, Type, Union
from dataclasses import dataclass, field
from collections import defaultdict
import time
import random
import statistics
import sys
from pathlib import Path

# Add paths for import resolution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'examples'))

# Use unified backend system
print("🔄 Loading implementations via backend system...")

try:
    from common.backend_config import get_backend_manager, create_example_tree, get_tree_analytics
    backend_manager = get_backend_manager()
    available_backends = backend_manager.get_available_backends()
    BACKEND_SYSTEM_AVAILABLE = True
    
    print(f"✅ Backend system loaded with {len(available_backends)} implementations:")
    for backend_id, info in available_backends.items():
        print(f"  • {info.name} ({info.language}, {info.performance_tier})")
    
except ImportError as e:
    print(f"❌ Backend system not available: {e}")
    BACKEND_SYSTEM_AVAILABLE = False
    backend_manager = None
    available_backends = {}

print("📊 Implementation loading complete\n")

# Configuration
INITIAL_INTERVAL_SIZE: Tuple[int, int] = (0, 10_000_000)
BASIC_ITERATIONS: int = 50_000  # Reduced for more implementations
SUMMARY_ITERATIONS: int = 10_000  # Fewer for summary operations

@dataclass
class BenchmarkResult:
    name: str
    category: str  # "Simple" or "Summary"
    language: str  # "Python" or "C++"
    total_time: float
    operations_per_second: float
    op_times: Dict[str, float] = field(default_factory=dict)
    memory_estimate: Optional[float] = None
    intervals_count: int = 0
    summary_stats: Optional[Dict[str, Any]] = None


class BenchmarkSuite:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def register_implementations(self) -> Dict[str, Dict]:
        """Register all available implementations by category using backend system"""
        implementations = {
            "Simple Python": {},
            "Summary Python": {},
            "Simple C++": {},
            "Summary C++": {}
        }
        
        if BACKEND_SYSTEM_AVAILABLE:
            # Categorize backends by type and language
            for backend_id, backend_info in available_backends.items():
                # Determine category
                is_summary = ('summary' in backend_id.lower() or 
                            'O(1) analytics' in str(backend_info.features))
                is_cpp = backend_info.language == "C++"
                
                if is_cpp and is_summary:
                    category = "Summary C++"
                elif is_cpp:
                    category = "Simple C++"
                elif is_summary:
                    category = "Summary Python"
                else:
                    category = "Simple Python"
                
                # Use a clean name for the implementation
                if 'treap' in backend_id:
                    impl_name = "Treap"
                elif 'summary' in backend_id:
                    impl_name = "Summary Tree"
                elif 'boundary' in backend_id:
                    impl_name = "Boundary"
                elif 'avl' in backend_id:
                    impl_name = "AVL"
                elif 'ic' in backend_id:
                    impl_name = "IC"
                else:
                    impl_name = backend_info.name.split()[0]
                
                implementations[category][impl_name] = backend_id  # Store backend_id instead of class
        
        else:
            # Fallback to old system (should not happen in normal usage)
            print("⚠️  Using fallback implementation registration")
            implementations["Simple Python"]["Boundary"] = "py_boundary"
            implementations["Simple Python"]["AVL"] = "py_avl"
            
        return implementations
    
    def generate_operations(self, num_operations: int, operation_mix: Dict[str, float] = None) -> List[Tuple[str, int, int]]:
        """Generate randomized operations with configurable mix"""
        if operation_mix is None:
            operation_mix = {'reserve': 0.4, 'release': 0.4, 'find': 0.2}
            
        operations = []
        op_types = list(operation_mix.keys())
        op_weights = list(operation_mix.values())
        
        for _ in range(num_operations):
            op_type = random.choices(op_types, weights=op_weights)[0]
            start = random.randint(0, 9_999_900)
            length = random.randint(1, 1000)
            end = start + length
            operations.append((op_type, start, end))
            
        return operations
    
    def benchmark_basic_operations(self, backend_id: str, operations: List[Tuple[str, int, int]], name: str, category: str, language: str) -> BenchmarkResult:
        """Benchmark basic interval operations using backend system"""
        print(f"  📊 Testing {name}...")
        
        try:
            # Create tree using backend system
            if BACKEND_SYSTEM_AVAILABLE:
                tree = create_example_tree(backend_id, random_seed=42)
            else:
                # Fallback - this shouldn't happen in normal usage
                raise ValueError("Backend system not available")
            
            tree.release_interval(*INITIAL_INTERVAL_SIZE)
            
            op_times = defaultdict(list)
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
                            # Skip find operations for implementations without this method
                            continue
                            
                    op_times[op].append(time.perf_counter() - start_time)
                    
                except Exception as e:
                    # Silently continue for expected failures (no suitable interval, etc.)
                    continue
            
            total_time = time.perf_counter() - start_total
            
            # Calculate average operation times
            avg_times = {}
            total_ops = 0
            for op, times in op_times.items():
                if times:
                    avg_times[op] = statistics.mean(times)
                    total_ops += len(times)
            
            ops_per_second = total_ops / total_time if total_time > 0 else 0
            
            # Get final tree analytics
            analytics = get_tree_analytics(tree) if BACKEND_SYSTEM_AVAILABLE else {}
            intervals_count = analytics.get('tree_size', 0)
                
            return BenchmarkResult(
                name=name,
                category=category,
                language=language,
                total_time=total_time,
                operations_per_second=ops_per_second,
                op_times=avg_times,
                intervals_count=intervals_count
            )
            
        except Exception as e:
            print(f"    ❌ Failed to benchmark {name}: {e}")
            return BenchmarkResult(
                name=name,
                category=category,
                language=language,
                total_time=0.0,
                operations_per_second=0.0,
                intervals_count=0
            )
    
    def benchmark_summary_operations(self, backend_id: str, operations: List[Tuple[str, int, int]], name: str, language: str) -> BenchmarkResult:
        """Benchmark summary-enhanced operations using backend system"""
        print(f"  🌟 Testing {name} (Summary Operations)...")
        
        try:
            # Create tree using backend system
            tree = create_example_tree(backend_id, random_seed=42)
            tree.release_interval(*INITIAL_INTERVAL_SIZE)
            
            # Perform basic operations to create fragmentation
            basic_ops = operations[:len(operations)//2]
            for op, start, end in basic_ops:
                try:
                    if op == 'reserve':
                        tree.reserve_interval(start, end)
                    elif op == 'release':
                        tree.release_interval(start, end)
                except:
                    continue
            
            # Benchmark summary-specific operations
            summary_times = {}
            summary_stats = {}
            
            # Test get_availability_stats
            if hasattr(tree, 'get_availability_stats'):
                start_time = time.perf_counter()
                for _ in range(100):  # Multiple calls to get average
                    try:
                        stats = tree.get_availability_stats()
                    except:
                        break
                summary_times['get_stats'] = (time.perf_counter() - start_time) / 100
                try:
                    final_stats = tree.get_availability_stats()
                    summary_stats = dict(final_stats) if hasattr(final_stats, '__dict__') else final_stats
                except:
                    pass
            
            # Test find_best_fit
            if hasattr(tree, 'find_best_fit'):
                start_time = time.perf_counter()
                successful_calls = 0
                for _ in range(100):
                    try:
                        tree.find_best_fit(random.randint(50, 500))
                        successful_calls += 1
                    except (ValueError, AttributeError):
                        continue
                if successful_calls > 0:
                    summary_times['best_fit'] = (time.perf_counter() - start_time) / successful_calls
            
            # Test find_largest_available
            if hasattr(tree, 'find_largest_available'):
                start_time = time.perf_counter()
                for _ in range(100):
                    try:
                        tree.find_largest_available()
                    except:
                        break
                summary_times['largest'] = (time.perf_counter() - start_time) / 100
            
            # Test tree summary methods
            summary_methods = ['get_tree_summary', 'get_summary', 'get_statistics']
            for method in summary_methods:
                if hasattr(tree, method):
                    start_time = time.perf_counter()
                    for _ in range(100):
                        try:
                            getattr(tree, method)()
                        except:
                            break
                    summary_times['tree_summary'] = (time.perf_counter() - start_time) / 100
                    break
            
            total_summary_time = sum(summary_times.values())
            ops_per_second = len(summary_times) * 100 / total_summary_time if total_summary_time > 0 else 0
            
            # Get interval count
            analytics = get_tree_analytics(tree)
            intervals_count = analytics.get('tree_size', 0)
            
            return BenchmarkResult(
                name=name,
                category="Summary",
                language=language,
                total_time=total_summary_time,
                operations_per_second=ops_per_second,
                op_times=summary_times,
                intervals_count=intervals_count,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            print(f"    ❌ Failed to benchmark {name}: {e}")
            return BenchmarkResult(
                name=name,
                category="Summary",
                language=language,
                total_time=0.0,
                operations_per_second=0.0,
                intervals_count=0
            )
    
    def run_comprehensive_benchmark(self, seed: Optional[int] = None) -> None:
        """Run comprehensive benchmark across all implementations"""
        if seed:
            random.seed(seed)
            print(f"🎲 Using random seed: {seed}")
        
        print("🚀 Starting Comprehensive Tree-Mendous Benchmark")
        print("=" * 70)
        
        if not BACKEND_SYSTEM_AVAILABLE:
            print("❌ Backend system not available - cannot run comprehensive benchmark")
            return
        
        implementations = self.register_implementations()
        basic_operations = self.generate_operations(BASIC_ITERATIONS)
        summary_operations = self.generate_operations(SUMMARY_ITERATIONS)
        
        # Benchmark all implementations
        for category, impls in implementations.items():
            if not impls:
                continue
                
            print(f"\n📦 {category} Implementations:")
            is_summary = "Summary" in category
            is_cpp = "C++" in category
            language = "C++" if is_cpp else "Python"
            
            for impl_name, backend_id in impls.items():  # backend_id instead of impl_class
                try:
                    # Basic operations benchmark
                    ops_to_use = summary_operations if is_summary else basic_operations
                    result = self.benchmark_basic_operations(
                        backend_id, ops_to_use, impl_name, category, language
                    )
                    self.results.append(result)
                    
                    # Additional summary operations benchmark for summary implementations
                    if is_summary:
                        try:
                            # Test if backend supports summary operations
                            test_tree = create_example_tree(backend_id, random_seed=42)
                            if hasattr(test_tree, 'get_availability_stats') or hasattr(test_tree, 'get_statistics'):
                                summary_result = self.benchmark_summary_operations(
                                    backend_id, summary_operations, impl_name, language
                                )
                                summary_result.name = f"{impl_name} (Advanced)"
                                self.results.append(summary_result)
                        except:
                            pass
                    
                except Exception as e:
                    print(f"    ❌ Failed to benchmark {impl_name}: {e}")
                    continue
    
    def print_results(self) -> None:
        """Print comprehensive benchmark results"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 70)
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Print results by category
        for category, results in categories.items():
            print(f"\n🏷️  {category} Results:")
            print("-" * 50)
            
            # Sort by total time (fastest first)
            results.sort(key=lambda x: x.total_time)
            
            for result in results:
                print(f"  {result.name:20} ({result.language:6}): {result.total_time:.4f}s")
                print(f"    {'':20} Ops/sec: {result.operations_per_second:,.0f}")
                print(f"    {'':20} Intervals: {result.intervals_count}")
                
                # Print operation breakdown
                if result.op_times:
                    print(f"    {'':20} Op times (ms):", end="")
                    for op, time_ms in result.op_times.items():
                        print(f" {op}: {time_ms*1000:.3f}", end="")
                    print()
                
                # Print summary stats for summary implementations
                if result.summary_stats and result.category == "Summary":
                    stats = result.summary_stats
                    if isinstance(stats, dict):
                        utilization = stats.get('utilization', 0) * 100
                        fragmentation = stats.get('fragmentation', 0) * 100
                        print(f"    {'':20} Utilization: {utilization:.1f}%, Fragmentation: {fragmentation:.1f}%")
                
                print()
        
        # Performance comparisons
        self.print_performance_analysis()
    
    def print_performance_analysis(self) -> None:
        """Print detailed performance analysis"""
        print("\n🔬 PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Find baseline (fastest simple Python implementation)
        simple_python = [r for r in self.results if r.category == "Simple Python"]
        if simple_python:
            baseline = min(simple_python, key=lambda x: x.total_time)
            baseline_time = baseline.total_time
            
            print(f"📏 Baseline: {baseline.name} (Python) - {baseline_time:.4f}s")
            print("\nSpeedup Analysis:")
            print("-" * 30)
            
            # Group by category for comparison
            cpp_simple = [r for r in self.results if r.category == "Simple C++"]
            python_summary = [r for r in self.results if r.category == "Summary Python"]  
            cpp_summary = [r for r in self.results if r.category == "Summary C++"]
            
            def print_speedups(results, category_name):
                if results:
                    print(f"\n{category_name}:")
                    for result in sorted(results, key=lambda x: x.total_time):
                        speedup = baseline_time / result.total_time
                        slowdown = result.total_time / baseline_time
                        if speedup > 1:
                            print(f"  {result.name:20}: {speedup:.2f}x FASTER")
                        else:
                            print(f"  {result.name:20}: {slowdown:.2f}x slower")
            
            print_speedups(cpp_simple, "Simple C++")
            print_speedups(python_summary, "Summary Python")  
            print_speedups(cpp_summary, "Summary C++")
        
        # Summary-enhanced features analysis
        summary_results = [r for r in self.results if "Advanced" in r.name]
        if summary_results:
            print(f"\n🌟 Summary Operations Performance:")
            print("-" * 40)
            
            for result in sorted(summary_results, key=lambda x: x.total_time):
                print(f"  {result.name:25}: {result.total_time*1000:.2f}ms total")
                if 'get_stats' in result.op_times:
                    stats_time = result.op_times['get_stats'] * 1000000  # Convert to microseconds
                    print(f"    {'':25}   get_stats: {stats_time:.1f}µs")
                if 'best_fit' in result.op_times:
                    bf_time = result.op_times['best_fit'] * 1000000
                    print(f"    {'':25}   best_fit: {bf_time:.1f}µs")
        
        # Memory usage estimates (simplified)
        print(f"\n💾 Estimated Memory Efficiency:")
        print("-" * 30)
        
        for category in ["Simple Python", "Summary Python", "Simple C++", "Summary C++"]:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                avg_intervals = statistics.mean([r.intervals_count for r in category_results])
                if "Python" in category:
                    est_memory = avg_intervals * (256 if "Simple" in category else 320)  # bytes per interval
                else:
                    est_memory = avg_intervals * (96 if "Simple" in category else 128)   # C++ is more efficient
                    
                print(f"  {category:15}: ~{est_memory/1024:.1f} KB ({est_memory/avg_intervals:.0f} bytes/interval)")


def main():
    """Main benchmark execution"""
    print("🌟 Tree-Mendous Comprehensive Performance Benchmark")
    print("Testing complete implementation hierarchy")
    print("=" * 70)
    
    # Parse command line arguments
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print("⚠️  Invalid seed, using random")
    
    # Create and run benchmark suite
    suite = BenchmarkSuite()
    suite.run_comprehensive_benchmark(seed)
    suite.print_results()
    
    # Save results for further analysis
    print("\n💾 Benchmark complete!")
    print("Results can be used for performance optimization and scaling decisions.")


if __name__ == "__main__":
    main()
