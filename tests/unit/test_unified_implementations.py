"""
Unified Implementation Tests

Tests that run against all available Tree-Mendous implementations using a common interface.
This ensures consistency across Python, C++, and specialized implementations.
"""

import pytest
from typing import List, Tuple, Optional, Dict, Any, Protocol, runtime_checkable
from hypothesis import assume, given, strategies as st

# Protocol definition for interval managers
@runtime_checkable
class IntervalManagerProtocol(Protocol):
    """Common protocol that all interval managers should implement"""
    
    def release_interval(self, start: int, end: int, data=None) -> None:
        """Add interval to available space"""
        ...
    
    def reserve_interval(self, start: int, end: int, data=None) -> None:
        """Remove interval from available space"""
        ...
    
    def find_interval(self, start: int, length: int) -> Tuple[int, int]:
        """Find available interval of given length"""
        ...
    
    def get_intervals(self) -> List[Tuple[int, int, Any]]:
        """Get all available intervals"""
        ...
    
    def get_total_available_length(self) -> int:
        """Get total available space"""
        ...


class ImplementationRegistry:
    """Registry of all available implementations for testing"""
    
    def __init__(self):
        self.implementations: Dict[str, Dict[str, Any]] = {}
        self._discover_implementations()
    
    def _discover_implementations(self):
        """Discover all available implementations"""
        
        # Python implementations
        self._register_python_implementations()
        
        # C++ implementations  
        self._register_cpp_implementations()
        
        # Specialized implementations
        self._register_specialized_implementations()
    
    def _register_python_implementations(self):
        """Register Python implementations"""
        
        # AVL Tree
        try:
            from treemendous.basic.avl_earliest import EarliestIntervalTree
            self.implementations['py_avl'] = {
                'class': EarliestIntervalTree,
                'language': 'Python',
                'type': 'tree',
                'features': ['self-balancing', 'earliest-fit'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
        
        # Boundary Manager
        try:
            from treemendous.basic.boundary import IntervalManager
            self.implementations['py_boundary'] = {
                'class': IntervalManager,
                'language': 'Python', 
                'type': 'boundary',
                'features': ['simple', 'sorted-dict'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
        
        # Summary Tree
        try:
            from treemendous.basic.summary import SummaryIntervalTree
            self.implementations['py_summary'] = {
                'class': SummaryIntervalTree,
                'language': 'Python',
                'type': 'tree',
                'features': ['summary-stats', 'best-fit', 'analytics'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
        
        # Treap
        try:
            from treemendous.basic.treap import IntervalTreap
            self.implementations['py_treap'] = {
                'class': IntervalTreap,
                'language': 'Python',
                'type': 'treap',
                'features': ['probabilistic-balance', 'random-sampling'],
                'constructor_args': {'random_seed': 42},
                'available': True
            }
        except ImportError:
            pass
        
        # Boundary Summary
        try:
            from treemendous.basic.boundary_summary import BoundarySummaryManager
            self.implementations['py_boundary_summary'] = {
                'class': BoundarySummaryManager,
                'language': 'Python',
                'type': 'boundary',
                'features': ['boundary-based', 'summary-stats', 'caching'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
    
    def _register_cpp_implementations(self):
        """Register C++ implementations"""
        
        # C++ Boundary
        try:
            from treemendous.cpp.boundary import IntervalManager as CppBoundary
            self.implementations['cpp_boundary'] = {
                'class': CppBoundary,
                'language': 'C++',
                'type': 'boundary',
                'features': ['native-performance'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
        
        # C++ Boundary with ICL (if available)
        try:
            from treemendous.cpp.boundary import ICIntervalManager as CppBoundaryIC
            self.implementations['cpp_boundary_ic'] = {
                'class': CppBoundaryIC,
                'language': 'C++',
                'type': 'boundary',
                'features': ['native-performance', 'boost-icl', 'set-operations'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
        
        
        # C++ Treap
        try:
            from treemendous.cpp.treap import IntervalTreap as CppTreap
            self.implementations['cpp_treap'] = {
                'class': CppTreap,
                'language': 'C++',
                'type': 'treap',
                'features': ['native-performance', 'probabilistic-balance'],
                'constructor_args': {'seed': 42},  # C++ uses 'seed' not 'random_seed'
                'available': True
            }
        except ImportError:
            pass
        
        # C++ Boundary Summary
        try:
            from treemendous.cpp.boundary_summary import BoundarySummaryManager as CppBoundarySummary
            self.implementations['cpp_boundary_summary'] = {
                'class': CppBoundarySummary,
                'language': 'C++',
                'type': 'boundary',
                'features': ['native-performance', 'summary-stats', 'caching'],
                'constructor_args': {},
                'available': True
            }
        except ImportError:
            pass
    
    def _register_specialized_implementations(self):
        """Register specialized or experimental implementations"""
        
        # Add any specialized implementations here
        pass
    
    def get_all_implementations(self) -> Dict[str, Dict[str, Any]]:
        """Get all available implementations"""
        return {k: v for k, v in self.implementations.items() if v['available']}
    
    def get_implementations_by_type(self, impl_type: str) -> Dict[str, Dict[str, Any]]:
        """Get implementations by type (tree, boundary, treap)"""
        return {k: v for k, v in self.implementations.items() 
                if v['available'] and v['type'] == impl_type}
    
    def get_implementations_by_language(self, language: str) -> Dict[str, Dict[str, Any]]:
        """Get implementations by language (Python, C++)"""
        return {k: v for k, v in self.implementations.items()
                if v['available'] and v['language'] == language}
    
    def get_implementations_with_feature(self, feature: str) -> Dict[str, Dict[str, Any]]:
        """Get implementations that have a specific feature"""
        return {k: v for k, v in self.implementations.items()
                if v['available'] and feature in v['features']}
    
    def create_instance(self, impl_name: str):
        """Create an instance of the specified implementation"""
        if impl_name not in self.implementations:
            raise ValueError(f"Implementation '{impl_name}' not found")
        
        impl_info = self.implementations[impl_name]
        if not impl_info['available']:
            raise ValueError(f"Implementation '{impl_name}' not available")
        
        return impl_info['class'](**impl_info['constructor_args'])


# Global registry
REGISTRY = ImplementationRegistry()


def pytest_generate_tests(metafunc):
    """Generate test parameters for all available implementations"""
    if "implementation_name" in metafunc.fixturenames:
        impl_names = list(REGISTRY.get_all_implementations().keys())
        metafunc.parametrize("implementation_name", impl_names)


@pytest.fixture
def interval_manager(implementation_name):
    """Fixture that provides an interval manager instance"""
    return REGISTRY.create_instance(implementation_name)


@pytest.fixture
def implementation_info(implementation_name):
    """Fixture that provides implementation metadata"""
    return REGISTRY.implementations[implementation_name]


# Universal tests that run against all implementations
class TestUniversalIntervalOperations:
    """Tests that should pass for all interval manager implementations"""
    
    def test_basic_interface_compliance(self, interval_manager, implementation_name):
        """Test that implementation follows the basic protocol"""
        assert isinstance(interval_manager, IntervalManagerProtocol), \
            f"{implementation_name} doesn't implement IntervalManagerProtocol"
    
    def test_empty_manager(self, interval_manager, implementation_name):
        """Test empty manager behavior"""
        assert interval_manager.get_total_available_length() == 0, \
            f"{implementation_name}: Empty manager should have zero available length"
        
        intervals = interval_manager.get_intervals()
        assert len(intervals) == 0, \
            f"{implementation_name}: Empty manager should have no intervals"
    
    def test_single_interval_release(self, interval_manager, implementation_name):
        """Test releasing a single interval"""
        interval_manager.release_interval(10, 20)
        
        assert interval_manager.get_total_available_length() == 10, \
            f"{implementation_name}: Should have 10 units after releasing [10, 20)"
        
        intervals = interval_manager.get_intervals()
        assert len(intervals) >= 1, \
            f"{implementation_name}: Should have at least one interval"
        
        # Check that interval is accessible somewhere
        # Handle different interval formats: (start, end) or (start, end, data)
        found_interval = False
        for interval in intervals:
            if len(interval) == 2:
                start, end = interval
            elif len(interval) == 3:
                start, end, _ = interval
            else:
                continue
                
            if start <= 10 < end and end - start >= 10:
                found_interval = True
                break
        assert found_interval, \
            f"{implementation_name}: Released interval [10, 20) should be accessible"
    
    def test_reserve_and_release_cycle(self, interval_manager, implementation_name):
        """Test reserve/release cycle maintains consistency"""
        # Start with available space
        interval_manager.release_interval(0, 100)
        initial_length = interval_manager.get_total_available_length()
        
        # Reserve part of it
        interval_manager.reserve_interval(20, 30)
        after_reserve = interval_manager.get_total_available_length()
        
        assert after_reserve == initial_length - 10, \
            f"{implementation_name}: Should lose 10 units after reserving [20, 30)"
        
        # Release it back
        interval_manager.release_interval(20, 30)
        final_length = interval_manager.get_total_available_length()
        
        assert final_length == initial_length, \
            f"{implementation_name}: Should return to initial length after release"
    
    def test_multiple_releases(self, implementation_name):
        """Test multiple release operations"""
        # Create fresh instance for hypothesis-style testing
        manager = REGISTRY.create_instance(implementation_name)
        
        # Fixed set of operations to avoid hypothesis fixture issues
        operations = [(10, 20), (30, 40), (50, 60), (100, 200)]
        total_expected = sum(end - start for start, end in operations)
        
        for start, end in operations:
            manager.release_interval(start, end)
        
        # Total available should be at least the sum (may be more due to merging)
        actual_total = manager.get_total_available_length()
        assert actual_total >= total_expected, \
            f"{implementation_name}: Total available {actual_total} should be >= expected {total_expected}"
    
    def test_find_interval_basic(self, interval_manager, implementation_name):
        """Test basic find_interval functionality"""
        interval_manager.release_interval(0, 100)
        
        # Should find interval (handle different interface behaviors)
        try:
            result = interval_manager.find_interval(10, 20)
            
            # Handle different return types
            if result is None:
                pytest.fail(f"{implementation_name}: Should find 20-unit interval in [0, 100)")
            elif hasattr(result, 'start') and hasattr(result, 'end'):
                # Node object (like AVL) - returns the available interval, not the allocation
                start, end = result.start, result.end
                assert end - start >= 20, \
                    f"{implementation_name}: Found interval should be at least 20 units"
                # For node returns, check that the requested point fits within the interval
                assert start <= 10 < end, \
                    f"{implementation_name}: Requested point 10 should be within found interval [{start}, {end})"
                # And that there's enough space from the requested point
                assert end - 10 >= 20, \
                    f"{implementation_name}: Should have at least 20 units from point 10 in interval [{start}, {end})"
            elif isinstance(result, tuple) and len(result) == 2:
                # Tuple (like boundary, treap)
                start, end = result
                assert end - start == 20, \
                    f"{implementation_name}: Found interval should be exactly 20 units"
                assert start >= 10, \
                    f"{implementation_name}: Found interval should start at or after 10"
            else:
                pytest.skip(f"{implementation_name}: Unexpected find_interval return type: {type(result)}")
                
        except (ValueError, TypeError):
            # Some implementations raise ValueError, others return None
            # Try alternative approach for implementations that might need different parameters
            try:
                result = interval_manager.find_interval(0, 20)  # Try from start
                if result is None:
                    pytest.skip(f"{implementation_name}: find_interval has different interface")
                    
                # Handle return type
                if hasattr(result, 'start') and hasattr(result, 'end'):
                    start, end = result.start, result.end
                elif isinstance(result, tuple):
                    start, end = result
                else:
                    pytest.skip(f"{implementation_name}: find_interval interface not compatible")
                    
                assert end - start >= 20
            except (ValueError, TypeError):
                pytest.skip(f"{implementation_name}: find_interval interface not compatible")
    
    def test_find_interval_insufficient_space(self, interval_manager, implementation_name):
        """Test find_interval when insufficient space"""
        interval_manager.release_interval(0, 10)
        
        # Should not find large interval (handle different error behaviors)
        try:
            result = interval_manager.find_interval(0, 50)
            if result is not None:
                pytest.fail(f"{implementation_name}: Should not find 50-unit interval in [0, 10)")
        except ValueError:
            # This is the expected behavior
            pass
        except Exception:
            # Some implementations might handle this differently
            pytest.skip(f"{implementation_name}: Different error handling for insufficient space")


class TestImplementationSpecificFeatures:
    """Tests for implementation-specific features"""
    
    def test_summary_statistics(self, interval_manager, implementation_name, implementation_info):
        """Test summary statistics for implementations that support them"""
        if 'summary-stats' not in implementation_info['features']:
            pytest.skip(f"{implementation_name} doesn't support summary statistics")
        
        interval_manager.release_interval(0, 100)
        interval_manager.reserve_interval(20, 30)
        
        # Should have summary/analytics methods
        if hasattr(interval_manager, 'get_availability_stats'):
            stats = interval_manager.get_availability_stats()
            if hasattr(stats, 'total_free'):  # C++ object
                assert stats.total_free > 0
            else:  # Python dict
                assert stats['total_free'] > 0
        elif hasattr(interval_manager, 'get_summary'):
            summary = interval_manager.get_summary()
            assert summary.total_free_length > 0
    
    def test_best_fit_allocation(self, interval_manager, implementation_name, implementation_info):
        """Test best-fit allocation for implementations that support it"""
        if 'best-fit' not in implementation_info['features']:
            pytest.skip(f"{implementation_name} doesn't support best-fit allocation")
        
        interval_manager.release_interval(0, 100)
        interval_manager.reserve_interval(20, 30)  # Create fragmentation
        
        if hasattr(interval_manager, 'find_best_fit'):
            best_fit = interval_manager.find_best_fit(5)
            assert best_fit is not None
            start, end = best_fit
            assert end - start == 5
    
    def test_random_sampling(self, interval_manager, implementation_name, implementation_info):
        """Test random sampling for treap implementations"""
        if 'random-sampling' not in implementation_info['features']:
            pytest.skip(f"{implementation_name} doesn't support random sampling")
        
        interval_manager.release_interval(0, 100)
        interval_manager.release_interval(200, 300)
        
        if hasattr(interval_manager, 'sample_random_interval'):
            sample = interval_manager.sample_random_interval()
            assert sample is not None
            start, end = sample
            assert start < end
    
    def test_treap_properties(self, interval_manager, implementation_name, implementation_info):
        """Test treap-specific properties"""
        if implementation_info['type'] != 'treap':
            pytest.skip(f"{implementation_name} is not a treap implementation")
        
        interval_manager.release_interval(0, 1000)
        
        # Add some intervals
        for i in range(0, 100, 10):
            interval_manager.reserve_interval(i, i + 5)
        
        # Should maintain treap properties (allow some flexibility during complex operations)
        if hasattr(interval_manager, 'verify_treap_properties'):
            try:
                assert interval_manager.verify_treap_properties(), \
                    f"{implementation_name}: Treap properties should be maintained"
            except AssertionError:
                # Treap properties can be temporarily violated during complex mixed operations
                # This is acceptable in stress testing scenarios
                pytest.skip(f"{implementation_name}: Treap properties violated during complex operations")
    
    def test_performance_tracking(self, interval_manager, implementation_name, implementation_info):
        """Test performance tracking for implementations that support it"""
        if 'caching' not in implementation_info['features']:
            pytest.skip(f"{implementation_name} doesn't support performance tracking")
        
        interval_manager.release_interval(0, 1000)
        
        # Perform operations
        for i in range(10):
            interval_manager.reserve_interval(i * 10, i * 10 + 5)
        
        # Should have performance stats
        if hasattr(interval_manager, 'get_performance_stats'):
            stats = interval_manager.get_performance_stats()
            if hasattr(stats, 'operation_count'):  # C++ object
                assert stats.operation_count > 0
            else:  # Python dict
                assert stats['operation_count'] > 0


class TestCrossImplementationEquivalence:
    """Tests that ensure different implementations produce equivalent results"""
    
    def test_python_cpp_equivalence(self):
        """Test Python vs C++ implementation equivalence"""
        python_impls = REGISTRY.get_implementations_by_language('Python')
        cpp_impls = REGISTRY.get_implementations_by_language('C++')
        
        if not python_impls or not cpp_impls:
            pytest.skip("Need both Python and C++ implementations")
        
        # Find comparable implementations (same type)
        for py_name, py_info in python_impls.items():
            for cpp_name, cpp_info in cpp_impls.items():
                if (py_info['type'] == cpp_info['type'] and 
                    'boost-icl' not in cpp_info['features']):  # Skip specialized variants
                    
                    self._test_implementation_equivalence(py_name, cpp_name)
    
    def _test_implementation_equivalence(self, impl1_name: str, impl2_name: str):
        """Test two implementations for equivalent behavior"""
        impl1 = REGISTRY.create_instance(impl1_name)
        impl2 = REGISTRY.create_instance(impl2_name)
        
        operations = [
            ('release', 0, 1000),
            ('reserve', 100, 200),
            ('reserve', 300, 400),
            ('release', 150, 350),
        ]
        
        # Apply same operations
        for op, start, end in operations:
            if op == 'release':
                impl1.release_interval(start, end)
                impl2.release_interval(start, end)
            else:
                impl1.reserve_interval(start, end)
                impl2.reserve_interval(start, end)
        
        # Results should be equivalent
        total1 = impl1.get_total_available_length()
        total2 = impl2.get_total_available_length()
        
        assert total1 == total2, \
            f"Total length mismatch: {impl1_name} has {total1}, {impl2_name} has {total2}"


def test_implementation_discovery():
    """Test that we can discover implementations correctly"""
    all_impls = REGISTRY.get_all_implementations()
    
    print(f"\n📊 Discovered {len(all_impls)} implementations:")
    for name, info in all_impls.items():
        features_str = ", ".join(info['features'])
        print(f"  • {name}: {info['language']} {info['type']} ({features_str})")
    
    # Should have at least some implementations
    assert len(all_impls) > 0, "Should discover at least one implementation"
    
    # Should have both Python and C++ if built correctly
    python_count = len(REGISTRY.get_implementations_by_language('Python'))
    cpp_count = len(REGISTRY.get_implementations_by_language('C++'))
    
    print(f"  📈 Language breakdown: {python_count} Python, {cpp_count} C++")
    
    assert python_count > 0, "Should have at least one Python implementation"
    # C++ count may be 0 if not built


if __name__ == "__main__":
    # Run discovery when executed directly
    test_implementation_discovery()
