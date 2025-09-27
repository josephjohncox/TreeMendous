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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all implementations following the hierarchy
print("🔄 Loading implementations...")

# Python Simple Implementations
from treemendous.basic.boundary import IntervalManager as PySimpleBoundary
from treemendous.basic.avl_earliest import EarliestIntervalTree as PySimpleAVL

# Python Randomized Implementation
try:
    sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))
    from treap import IntervalTreap as PyRandomizedTreap
    py_treap_available = True
except ImportError as e:
    print(f"⚠️  Python Treap not available: {e}")
    PyRandomizedTreap = None
    py_treap_available = False

# Python Summary Implementation 
try:
    from treemendous.basic.summary import SummaryIntervalTree as PySummaryTree
    py_summary_available = True
except ImportError as e:
    print(f"⚠️  Python Summary not available: {e}")
    PySummaryTree = None
    py_summary_available = False

# C++ Implementations
try:
    from treemendous.cpp.boundary import SimpleIntervalManager as CppSimpleBoundary
    from treemendous.cpp.boundary import IntervalManager as CppSummaryBoundary
    cpp_available = True
    print("✅ C++ implementations loaded")
except ImportError as e:
    print(f"⚠️  C++ implementations not available: {e}")
    CppSimpleBoundary = None
    CppSummaryBoundary = None
    cpp_available = False

try:
    from treemendous.cpp.boundary import SimpleICIntervalManager as CppSimpleIC
    from treemendous.cpp.boundary import ICIntervalManager as CppSummaryIC
    cpp_ic_available = True
    print("✅ C++ ICL implementations loaded")
except ImportError as e:
    print(f"⚠️  C++ ICL implementations not available: {e}")
    CppSimpleIC = None
    CppSummaryIC = None
    cpp_ic_available = False

# C++ Treap Implementation
try:
    from treemendous.cpp.treap import IntervalTreap as CppTreap
    cpp_treap_available = True
    print("✅ C++ Treap implementation loaded")
except ImportError as e:
    print(f"⚠️  C++ Treap implementation not available: {e}")
    CppTreap = None
    cpp_treap_available = False

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
        """Register all available implementations by category"""
        implementations = {
            "Simple Python": {},
            "Summary Python": {},
            "Simple C++": {},
            "Summary C++": {}
        }
        
        # Simple Python implementations
        implementations["Simple Python"]["Boundary"] = PySimpleBoundary
        implementations["Simple Python"]["AVL Earliest"] = PySimpleAVL
        if py_treap_available:
            implementations["Simple Python"]["Treap"] = PyRandomizedTreap
        
        # Summary Python implementations
        if py_summary_available:
            implementations["Summary Python"]["Summary Tree"] = PySummaryTree
        
        # Simple C++ implementations
        if cpp_available:
            implementations["Simple C++"]["Boundary"] = CppSimpleBoundary
        if cpp_ic_available:
            implementations["Simple C++"]["IC Boundary"] = CppSimpleIC
        if cpp_treap_available:
            implementations["Simple C++"]["Treap"] = CppTreap
            
        # Summary C++ implementations  
        if cpp_available:
            implementations["Summary C++"]["Boundary"] = CppSummaryBoundary
        if cpp_ic_available:
            implementations["Summary C++"]["IC Boundary"] = CppSummaryIC
            
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
    
    def benchmark_basic_operations(self, manager_class, operations: List[Tuple[str, int, int]], name: str, category: str, language: str) -> BenchmarkResult:
        """Benchmark basic interval operations"""
        print(f"  📊 Testing {name}...")
        
        manager = manager_class()
        manager.release_interval(*INITIAL_INTERVAL_SIZE)
        
        op_times = defaultdict(list)
        start_total = time.perf_counter()
        
        for op, start, end in operations:
            start_time = time.perf_counter()
            
            try:
                if op == 'reserve':
                    manager.reserve_interval(start, end)
                elif op == 'release':
                    manager.release_interval(start, end)
                elif op == 'find':
                    if hasattr(manager, 'find_interval'):
                        manager.find_interval(start, end - start)
                    else:
                        # Skip find operations for implementations without this method
                        continue
                        
                op_times[op].append(time.perf_counter() - start_time)
                
            except Exception as e:
                print(f"    ⚠️  Error in {op} operation: {e}")
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
        
        # Get final intervals count
        try:
            intervals = manager.get_intervals()
            intervals_count = len(intervals)
        except:
            intervals_count = 0
            
        return BenchmarkResult(
            name=name,
            category=category,
            language=language,
            total_time=total_time,
            operations_per_second=ops_per_second,
            op_times=avg_times,
            intervals_count=intervals_count
        )
    
    def benchmark_summary_operations(self, manager_class, operations: List[Tuple[str, int, int]], name: str, language: str) -> BenchmarkResult:
        """Benchmark summary-enhanced operations"""
        print(f"  🌟 Testing {name} (Summary Operations)...")
        
        manager = manager_class()
        manager.release_interval(*INITIAL_INTERVAL_SIZE)
        
        # Perform basic operations to create fragmentation
        basic_ops = operations[:len(operations)//2]
        for op, start, end in basic_ops:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            elif op == 'release':
                manager.release_interval(start, end)
        
        # Benchmark summary-specific operations
        summary_times = {}
        summary_stats = {}
        
        # Test get_availability_stats
        if hasattr(manager, 'get_availability_stats'):
            start_time = time.perf_counter()
            for _ in range(100):  # Multiple calls to get average
                stats = manager.get_availability_stats()
            summary_times['get_stats'] = (time.perf_counter() - start_time) / 100
            summary_stats = dict(stats) if hasattr(stats, '__dict__') else stats
        
        # Test find_best_fit
        if hasattr(manager, 'find_best_fit'):
            start_time = time.perf_counter()
            for _ in range(100):
                manager.find_best_fit(random.randint(50, 500))
            summary_times['best_fit'] = (time.perf_counter() - start_time) / 100
        
        # Test find_largest_available
        if hasattr(manager, 'find_largest_available'):
            start_time = time.perf_counter()
            for _ in range(100):
                manager.find_largest_available()
            summary_times['largest'] = (time.perf_counter() - start_time) / 100
        
        # Test get_tree_summary (Python) or get_summary (C++)
        if hasattr(manager, 'get_tree_summary'):
            start_time = time.perf_counter()
            for _ in range(100):
                manager.get_tree_summary()
            summary_times['tree_summary'] = (time.perf_counter() - start_time) / 100
        elif hasattr(manager, 'get_summary'):
            start_time = time.perf_counter()
            for _ in range(100):
                manager.get_summary()
            summary_times['tree_summary'] = (time.perf_counter() - start_time) / 100
        
        total_summary_time = sum(summary_times.values())
        ops_per_second = len(summary_times) * 100 / total_summary_time if total_summary_time > 0 else 0
        
        return BenchmarkResult(
            name=name,
            category="Summary",
            language=language,
            total_time=total_summary_time,
            operations_per_second=ops_per_second,
            op_times=summary_times,
            intervals_count=len(manager.get_intervals()) if hasattr(manager, 'get_intervals') else 0,
            summary_stats=summary_stats
        )
    
    def run_comprehensive_benchmark(self, seed: Optional[int] = None) -> None:
        """Run comprehensive benchmark across all implementations"""
        if seed:
            random.seed(seed)
            print(f"🎲 Using random seed: {seed}")
        
        print("🚀 Starting Comprehensive Tree-Mendous Benchmark")
        print("=" * 70)
        
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
            
            for impl_name, impl_class in impls.items():
                try:
                    # Basic operations benchmark
                    ops_to_use = summary_operations if is_summary else basic_operations
                    result = self.benchmark_basic_operations(
                        impl_class, ops_to_use, impl_name, category, language
                    )
                    self.results.append(result)
                    
                    # Additional summary operations benchmark for summary implementations
                    if is_summary and hasattr(impl_class(), 'get_availability_stats'):
                        summary_result = self.benchmark_summary_operations(
                            impl_class, summary_operations, impl_name, language
                        )
                        summary_result.name = f"{impl_name} (Advanced)"
                        self.results.append(summary_result)
                    
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
