#!/usr/bin/env python3
"""
Backend Configuration for Tree-Mendous Examples

Provides unified interface for switching between Python and C++ implementations
while maintaining identical functionality through common protocols.
"""

import sys
import os
import argparse
from typing import Optional, Dict, Any, Type, Union, List
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

# Add Tree-Mendous to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))


class Backend(Enum):
    PYTHON = "python"
    CPP = "cpp"
    AUTO = "auto"


@dataclass
class BackendInfo:
    """Information about available backend implementations"""
    name: str
    language: str
    available: bool
    performance_tier: str  # "baseline", "optimized", "high_performance"
    features: list
    estimated_speedup: float = 1.0


class BackendManager:
    """Manages backend selection and implementation loading"""
    
    def __init__(self):
        self.available_backends = {}
        self.current_backend = Backend.AUTO
        self.implementations = {}
        self._load_implementations()
    
    def _load_implementations(self):
        """Load and register available implementations"""
        
        # Python implementations
        from treemendous.basic.summary import SummaryIntervalTree as PySummaryTree
        self.available_backends['py_summary'] = BackendInfo(
            name="Python Summary Tree",
            language="Python",
            available=True,
            performance_tier="baseline", 
            features=["O(1) analytics", "comprehensive summaries", "best-fit queries"],
            estimated_speedup=1.0
        )
        self.implementations['py_summary'] = PySummaryTree
        
        from treemendous.basic.treap import IntervalTreap as PyTreap
        self.available_backends['py_treap'] = BackendInfo(
            name="Python Treap",
            language="Python", 
            available=True,
            performance_tier="optimized",
            features=["probabilistic balance", "O(log n) expected", "random sampling"],
            estimated_speedup=1.2
        )
        self.implementations['py_treap'] = PyTreap
        
        from treemendous.basic.boundary import IntervalManager as PyBoundary
        self.available_backends['py_boundary'] = BackendInfo(
            name="Python Boundary Manager",
            language="Python",
            available=True,
            performance_tier="baseline",
            features=["simple implementation", "SortedDict-based"],
            estimated_speedup=0.8
        )
        self.implementations['py_boundary'] = PyBoundary
        
        from treemendous.basic.avl_earliest import EarliestIntervalTree as PyAVL
        self.available_backends['py_avl'] = BackendInfo(
            name="Python AVL Earliest",
            language="Python",
            available=True,
            performance_tier="optimized",
            features=["self-balancing", "earliest-fit optimization"],
            estimated_speedup=0.9
        )
        self.implementations['py_avl'] = PyAVL
        
        from treemendous.basic.boundary_summary import BoundarySummaryManager as PyBoundarySummary
        self.available_backends['py_boundary_summary'] = BackendInfo(
            name="Python Boundary Summary",
            language="Python",
            available=True,
            performance_tier="optimized",
            features=["boundary-based", "O(1) analytics", "caching", "best-fit allocation"],
            estimated_speedup=1.1
        )
        self.implementations['py_boundary_summary'] = PyBoundarySummary
        
        # C++ implementations (may not be available if not compiled)
        try:
            from treemendous.cpp.boundary import IntervalManager as CppBoundaryManager
            
            # Test actual capabilities
            test_tree = CppBoundaryManager()
            test_tree.release_interval(0, 100)
            
            features = ["native performance", "interval operations"]
            if hasattr(test_tree, 'get_availability_stats'):
                features.extend(["O(1) analytics", "summary statistics"])
            if hasattr(test_tree, 'find_best_fit'):
                features.append("best-fit allocation")
            
            backend_name = "C++ Boundary Manager"
            self.available_backends['cpp_boundary'] = BackendInfo(
                name=backend_name,
                language="C++",
                available=True,
                performance_tier="optimized",
                features=features,
                estimated_speedup=2.5
            )
            self.implementations['cpp_boundary'] = CppBoundaryManager
            
        except ImportError:
            pass  # C++ boundary not compiled
        
        # C++ optimized implementations (with micro-optimizations)
        try:
            from treemendous.cpp.boundary_optimized import IntervalManager as CppBoundaryOptimized
            self.available_backends['cpp_boundary_optimized'] = BackendInfo(
                name="C++ Boundary (Optimized)",
                language="C++",
                available=True,
                performance_tier="high_performance",
                features=["native performance", "small-vector opt", "branch hints"],
                estimated_speedup=2.5
            )
            self.implementations['cpp_boundary_optimized'] = CppBoundaryOptimized
        except ImportError:
            pass
        
        try:
            from treemendous.cpp.boundary_summary_optimized import BoundarySummaryManager as CppBoundarySummaryOptimized
            self.available_backends['cpp_boundary_summary_optimized'] = BackendInfo(
                name="C++ Boundary Summary (Optimized)",
                language="C++",
                available=True,
                performance_tier="high_performance",
                features=["native performance", "O(1) analytics", "caching", "small-vector opt"],
                estimated_speedup=3.0
            )
            self.implementations['cpp_boundary_summary_optimized'] = CppBoundarySummaryOptimized
        except ImportError:
            pass
        
        # C++ Treap (optional - only if compiled)
        try:
            from treemendous.cpp.treap import IntervalTreap as CppTreap
            self.available_backends['cpp_treap'] = BackendInfo(
                name="C++ Treap",
                language="C++",
                available=True,
                performance_tier="high_performance",
                features=["probabilistic balance", "native performance", "random sampling", "split/merge"],
                estimated_speedup=5.0
            )
            self.implementations['cpp_treap'] = CppTreap
        except ImportError:
            pass  # C++ treap not compiled
        
        # C++ Boundary Summary (optional - only if compiled)
        try:
            from treemendous.cpp.boundary_summary import BoundarySummaryManager as CppBoundarySummary
            self.available_backends['cpp_boundary_summary'] = BackendInfo(
                name="C++ Boundary Summary",
                language="C++",
                available=True,
                performance_tier="high_performance",
                features=["boundary-based", "native performance", "O(1) analytics", "caching"],
                estimated_speedup=6.0
            )
            self.implementations['cpp_boundary_summary'] = CppBoundarySummary
        except ImportError:
            pass  # C++ boundary summary not compiled
    
    def get_available_backends(self) -> Dict[str, BackendInfo]:
        """Get information about all available backends"""
        return {k: v for k, v in self.available_backends.items() if v.available}
    
    def select_backend(self, preference: str = "auto") -> str:
        """Select best available backend based on preference"""
        available = self.get_available_backends()
        
        if not available:
            raise RuntimeError("No Tree-Mendous implementations available")
        
        if preference == "auto":
            # Auto-select best available implementation
            # Priority: C++ Boundary Summary > C++ Summary > C++ Treap > Python Boundary Summary > Python Summary > Others
            priorities = [
                'cpp_boundary_summary',  # Highest: C++ Boundary + Summary (best of both worlds)
                'cpp_summary',           # High: C++ Summary (best analytics + performance) 
                'cpp_treap',             # High: C++ Treap (probabilistic balance + performance)
                'py_boundary_summary',   # Medium-High: Python Boundary + Summary (efficient + analytics)
                'py_summary',            # Medium: Python Summary (best analytics)
                'py_treap',              # Medium: Python Treap (probabilistic balance)
                'cpp_boundary',          # Lower: C++ Simple (performance only)
                'py_avl',                # Lower: Python AVL (deterministic balance)
                'py_boundary'            # Lowest: Python Boundary (simple)
            ]
            
            for backend_id in priorities:
                if backend_id in available:
                    return backend_id
            
            # Fallback to first available
            return list(available.keys())[0]
        
        elif preference in available:
            return preference
        
        elif preference in self.available_backends:
            # Backend exists but not available
            backend_info = self.available_backends[preference]
            available_names = [info.name for info in available.values()]
            raise ValueError(f"Backend '{backend_info.name}' not available. Available: {available_names}")
        
        else:
            # Try to find closest match
            preference_lower = preference.lower()
            
            for backend_id in available:
                if preference_lower in backend_id.lower():
                    return backend_id
            
            # No match found
            available_names = [info.name for info in available.values()]
            raise ValueError(f"Backend '{preference}' not available. Available: {available_names}")
    
    def create_tree(self, backend_id: Optional[str] = None, **kwargs) -> Any:
        """Create interval tree using specified backend"""
        if backend_id is None:
            backend_id = self.select_backend("auto")
        
        if backend_id not in self.implementations:
            raise ValueError(f"Backend '{backend_id}' not available")
        
        implementation_class = self.implementations[backend_id]
        
        # Handle different constructor signatures
        try:
            if backend_id == 'py_treap' and 'random_seed' in kwargs:
                # Python treap with seed
                return implementation_class(random_seed=kwargs['random_seed'])
            elif backend_id == 'cpp_treap' and 'random_seed' in kwargs:
                # C++ treap with seed
                return implementation_class(kwargs['random_seed'])
            else:
                # Default constructor for most implementations
                return implementation_class()
        except TypeError as e:
            # Fallback: try with no arguments
            try:
                return implementation_class()
            except TypeError:
                # If still fails, try with seed if available
                if 'random_seed' in kwargs:
                    try:
                        return implementation_class(kwargs['random_seed'])
                    except TypeError:
                        pass
                raise ValueError(f"Cannot create {backend_id} with available parameters: {e}")
    
    def print_backend_info(self, backend_id: str) -> None:
        """Print information about specified backend"""
        if backend_id not in self.available_backends:
            print(f"❌ Unknown backend: {backend_id}")
            return
        
        info = self.available_backends[backend_id]
        status = "✅ Available" if info.available else "❌ Not Available"
        
        print(f"{info.name} ({info.language}): {status}")
        if info.available:
            print(f"  Performance tier: {info.performance_tier}")
            print(f"  Estimated speedup: {info.estimated_speedup:.1f}x")
            print(f"  Features: {', '.join(info.features)}")
    
    def benchmark_available_backends(self, num_operations: int = 5000) -> Dict[str, Dict]:
        """Benchmark all available backends for comparison"""
        import time
        import random
        
        results = {}
        available = self.get_available_backends()
        
        print(f"🏁 Benchmarking {len(available)} available backends ({num_operations:,} operations)...")
        
        # Generate consistent test data
        random.seed(42)
        operations = []
        for _ in range(num_operations):
            op = random.choice(['reserve', 'release', 'find'])
            start = random.randint(0, num_operations * 10)
            length = random.randint(1, 100)
            operations.append((op, start, start + length))
        
        for backend_id, info in available.items():
            print(f"  Testing {info.name}...")
            
            try:
                # Create tree
                tree = self.create_tree(backend_id, random_seed=42)
                tree.release_interval(0, num_operations * 20)  # Initialize
                
                # Benchmark
                start_time = time.perf_counter()
                successful_ops = 0
                
                for op, start, end in operations:
                    try:
                        if op == 'reserve':
                            tree.reserve_interval(start, end)
                            successful_ops += 1
                        elif op == 'release':
                            tree.release_interval(start, end)
                            successful_ops += 1
                        elif op == 'find':
                            if hasattr(tree, 'find_interval'):
                                tree.find_interval(start, end - start)
                                successful_ops += 1
                            elif hasattr(tree, 'find_best_fit'):
                                tree.find_best_fit(end - start)
                                successful_ops += 1
                    except (ValueError, AttributeError):
                        continue
                
                total_time = time.perf_counter() - start_time
                ops_per_second = successful_ops / total_time if total_time > 0 else 0
                
                # Get final statistics
                final_stats = {}
                if hasattr(tree, 'get_statistics'):
                    final_stats = tree.get_statistics()
                elif hasattr(tree, 'get_availability_stats'):
                    stats = tree.get_availability_stats()
                    final_stats = {
                        'utilization': stats['utilization'],
                        'fragmentation': stats['fragmentation'],
                        'intervals': len(tree.get_intervals()) if hasattr(tree, 'get_intervals') else 0
                    }
                
                results[backend_id] = {
                    'name': info.name,
                    'ops_per_second': ops_per_second,
                    'total_time': total_time,
                    'successful_ops': successful_ops,
                    'final_stats': final_stats
                }
                
                print(f"    {ops_per_second:>8,.0f} ops/sec")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results[backend_id] = {'name': info.name, 'error': str(e)}
        
        return results


# Global backend manager instance
_backend_manager = BackendManager()


def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance"""
    return _backend_manager


def create_interval_tree(backend: str = "auto", **kwargs) -> Any:
    """Create interval tree with specified backend"""
    return _backend_manager.create_tree(backend, **kwargs)


def parse_backend_args(description: str = "Tree-Mendous Example") -> argparse.Namespace:
    """Parse command line arguments for backend selection"""
    parser = argparse.ArgumentParser(description=description)
    
    available_backends = list(_backend_manager.get_available_backends().keys())
    
    parser.add_argument(
        '--backend', '-b',
        choices=['auto'] + available_backends,
        default='auto',
        help='Backend implementation to use'
    )
    
    parser.add_argument(
        '--list-backends',
        action='store_true',
        help='List available backends and exit'
    )
    
    parser.add_argument(
        '--benchmark-backends',
        action='store_true', 
        help='Benchmark all available backends'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def handle_backend_args(args: argparse.Namespace) -> Optional[str]:
    """Handle backend-related command line arguments"""
    
    if args.list_backends:
        print("📋 Available Tree-Mendous Backends:")
        print("=" * 40)
        
        available = _backend_manager.get_available_backends()
        if not available:
            print("❌ No backends available")
            return None
        
        for backend_id, info in available.items():
            _backend_manager.print_backend_info(backend_id)
            print()
        
        return None
    
    if args.benchmark_backends:
        print("🚀 Backend Performance Comparison")
        print("=" * 40)
        
        results = _backend_manager.benchmark_available_backends()
        
        print(f"\nBenchmark Results:")
        print(f"{'Backend':25} {'Ops/sec':>10} {'Language':>8} {'Status':>10}")
        print("-" * 55)
        
        for backend_id, result in results.items():
            if 'error' in result:
                print(f"{result['name']:25} {'ERROR':>10} {'N/A':>8} {'FAILED':>10}")
            else:
                backend_info = _backend_manager.available_backends[backend_id]
                print(f"{result['name']:25} {result['ops_per_second']:>10,.0f} "
                      f"{backend_info.language:>8} {'OK':>10}")
        
        return None
    
    # Select backend
    try:
        selected_backend = _backend_manager.select_backend(args.backend)
        backend_info = _backend_manager.available_backends[selected_backend]
        
        if args.verbose:
            print(f"🔧 Using backend: {backend_info.name}")
            print(f"   Language: {backend_info.language}")
            print(f"   Performance tier: {backend_info.performance_tier}")
            print(f"   Estimated speedup: {backend_info.estimated_speedup:.1f}x")
            print()
        
        return selected_backend
        
    except (ValueError, RuntimeError) as e:
        print(f"❌ Backend selection failed: {e}")
        return None


def create_example_tree(backend_id: str, **kwargs) -> Any:
    """Create interval tree for examples with error handling"""
    try:
        tree = _backend_manager.create_tree(backend_id, **kwargs)
        
        # Initialize with standard test space if not specified
        if 'initial_space' not in kwargs:
            tree.release_interval(0, 1000)
        
        return tree
        
    except Exception as e:
        print(f"❌ Failed to create tree with backend '{backend_id}': {e}")
        
        # Try fallback to auto selection
        try:
            fallback_backend = _backend_manager.select_backend("auto")
            if fallback_backend != backend_id:
                print(f"🔄 Falling back to {_backend_manager.available_backends[fallback_backend].name}")
                tree = _backend_manager.create_tree(fallback_backend, **kwargs)
                if 'initial_space' not in kwargs:
                    tree.release_interval(0, 1000)
                return tree
        except Exception:
            pass
        
        raise RuntimeError(f"Failed to create any interval tree implementation")


def get_tree_analytics(tree: Any) -> Dict[str, Any]:
    """Get analytics from tree regardless of implementation"""
    analytics = {}
    
    # Try different analytics methods
    if hasattr(tree, 'get_availability_stats'):
        try:
            stats = tree.get_availability_stats()
            if isinstance(stats, dict):
                analytics.update({
                    'utilization': stats.get('utilization', 0),
                    'fragmentation': stats.get('fragmentation', 0),
                    'total_free': stats.get('total_free', 0),
                    'free_chunks': stats.get('free_chunks', 0),
                    'largest_chunk': stats.get('largest_chunk', 0)
                })
            else:
                # C++ AvailabilityStats object
                analytics.update({
                    'utilization': getattr(stats, 'utilization', 0),
                    'fragmentation': getattr(stats, 'fragmentation', 0),
                    'total_free': getattr(stats, 'total_free', 0),
                    'free_chunks': getattr(stats, 'free_chunks', 0),
                    'largest_chunk': getattr(stats, 'largest_chunk', 0)
                })
        except (AttributeError, TypeError):
            pass
    
    if hasattr(tree, 'get_statistics'):
        try:
            stats = tree.get_statistics()
            if isinstance(stats, dict):
                analytics.update(stats)
            else:
                # C++ statistics object
                analytics.update({
                    'size': getattr(stats, 'size', 0),
                    'height': getattr(stats, 'height', 0),
                    'balance_factor': getattr(stats, 'balance_factor', 1.0),
                    'total_length': getattr(stats, 'total_length', 0)
                })
        except (AttributeError, TypeError):
            pass
    
    # Basic tree properties
    if hasattr(tree, 'get_total_available_length'):
        try:
            analytics['total_available'] = tree.get_total_available_length()
        except:
            pass
    
    if hasattr(tree, 'get_tree_size'):
        try:
            analytics['tree_size'] = tree.get_tree_size()
        except:
            pass
    elif hasattr(tree, 'get_intervals'):
        try:
            intervals = tree.get_intervals()
            analytics['tree_size'] = len(intervals) if intervals else 0
        except:
            pass
    
    return analytics


def detect_tree_features(tree: Any) -> List[str]:
    """Detect available features for a tree implementation"""
    features = []
    
    # Test each feature with actual method calls
    feature_tests = [
        ('get_intervals', 'basic_intervals'),
        ('find_interval', 'find_operations'),
        ('reserve_interval', 'basic_operations'),
        ('release_interval', 'basic_operations'),
        ('find_best_fit', 'best_fit_allocation'),
        ('get_availability_stats', 'summary_statistics'),
        ('sample_random_interval', 'random_sampling'),
        ('get_statistics', 'tree_statistics'),
        ('find_overlapping_intervals', 'overlap_queries'),
        ('split', 'split_operations'),
        ('merge_treap', 'merge_operations'),
        ('verify_treap_properties', 'property_verification'),
        ('get_rank', 'rank_operations'),
        ('find_largest_available', 'largest_block_finder'),
        ('get_tree_summary', 'comprehensive_summaries'),
    ]
    
    for method_name, feature_name in feature_tests:
        if hasattr(tree, method_name):
            # Additional validation for some features
            if method_name == 'find_best_fit':
                try:
                    # Test if method actually works
                    tree.find_best_fit(10)
                    features.append(feature_name)
                except (ValueError, AttributeError, TypeError):
                    # Method exists but might not be implemented
                    pass
            elif method_name == 'get_availability_stats':
                try:
                    stats = tree.get_availability_stats()
                    if stats:  # Has actual statistics
                        features.append(feature_name)
                except (AttributeError, TypeError):
                    pass
            else:
                features.append(feature_name)
    
    return features


# Example usage functions
def demo_backend_switching():
    """Demonstrate backend switching functionality"""
    print("🔄 Backend Switching Demo")
    print("=" * 30)
    
    manager = get_backend_manager()
    available = manager.get_available_backends()
    
    if len(available) < 2:
        print("⚠️  Need at least 2 backends for meaningful comparison")
        return
    
    # Test same operations on different backends
    operations = [
        ('release', 0, 1000),
        ('reserve', 100, 200),
        ('reserve', 300, 400),
        ('release', 150, 350),
    ]
    
    results = {}
    
    for backend_id, info in available.items():
        print(f"\nTesting {info.name}:")
        
        try:
            tree = manager.create_tree(backend_id, random_seed=42)
            
            # Apply operations
            for op, start, end in operations:
                if op == 'reserve':
                    tree.reserve_interval(start, end)
                else:
                    tree.release_interval(start, end)
            
            # Get analytics
            analytics = get_tree_analytics(tree)
            results[backend_id] = analytics
            
            print(f"  Final intervals: {len(tree.get_intervals()) if hasattr(tree, 'get_intervals') else 'N/A'}")
            print(f"  Total available: {analytics.get('total_available', 'N/A')}")
            if 'utilization' in analytics:
                print(f"  Utilization: {analytics['utilization']:.1%}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Compare results
    print(f"\n📊 Backend Comparison:")
    if len(results) > 1:
        # Check if results are consistent
        total_available_values = [r.get('total_available', 0) for r in results.values()]
        if len(set(total_available_values)) == 1:
            print("✅ All backends produced identical results")
        else:
            print("⚠️  Results differ between backends:")
            for backend_id, result in results.items():
                info = available[backend_id]
                print(f"  {info.name}: {result.get('total_available', 'N/A')} total available")


if __name__ == "__main__":
    # Demo backend management
    args = parse_backend_args("Backend Configuration Demo")
    
    # Handle special args
    selected_backend = handle_backend_args(args)
    if selected_backend is None:
        sys.exit(0)
    
    # Demo functionality
    demo_backend_switching()
    
    print(f"\n✅ Backend configuration system ready!")
    print(f"Use --backend=<name> to select specific implementations")
    print(f"Use --list-backends to see available options")
    print(f"Use --benchmark-backends to compare performance")
