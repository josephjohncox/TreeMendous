"""
Property-based tests for Boundary Summary interval managers.

Tests verify that boundary-based summary managers maintain correctness
while providing comprehensive O(1) analytics.
"""

from hypothesis import assume, given, strategies as st
from typing import List, Tuple, Optional, Dict, Any
import random
import math

# Import Python implementation
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))
    
    from boundary_summary import BoundarySummaryManager as PyBoundarySummary, BoundarySummary
    PY_BOUNDARY_SUMMARY_AVAILABLE = True
    print("✅ Python Boundary Summary loaded")
except ImportError as e:
    print(f"❌ Python Boundary Summary failed: {e}")
    PyBoundarySummary = None
    PY_BOUNDARY_SUMMARY_AVAILABLE = False

# Import C++ implementation
try:
    from treemendous.cpp.boundary_summary import BoundarySummaryManager as CppBoundarySummary
    CPP_BOUNDARY_SUMMARY_AVAILABLE = True
    print("✅ C++ Boundary Summary loaded")
except ImportError as e:
    print(f"⚠️  C++ Boundary Summary not available: {e}")
    CppBoundarySummary = None
    CPP_BOUNDARY_SUMMARY_AVAILABLE = False


def validate_boundary_summary_invariants(manager) -> None:
    """Verify boundary summary manager invariants"""
    if not hasattr(manager, 'get_summary'):
        return  # Skip validation for implementations without summary
    
    summary = manager.get_summary()
    
    # Basic invariants
    assert summary.total_free_length >= 0, "Total free length cannot be negative"
    assert summary.interval_count >= 0, "Interval count cannot be negative"
    assert summary.largest_interval_length >= 0, "Largest interval length cannot be negative"
    
    # Consistency checks
    if summary.interval_count > 0:
        assert summary.total_free_length > 0, "If intervals exist, total free should be positive"
        assert summary.largest_interval_length > 0, "If intervals exist, largest should be positive"
        assert summary.avg_interval_length > 0, "If intervals exist, average should be positive"
    
    # Bounds checks
    assert 0.0 <= summary.fragmentation_index <= 1.0, f"Fragmentation should be [0,1], got {summary.fragmentation_index}"
    assert 0.0 <= summary.utilization <= 1.0, f"Utilization should be [0,1], got {summary.utilization}"
    
    # Verify statistics match actual intervals
    if hasattr(manager, 'get_intervals'):
        intervals = manager.get_intervals()
        # Handle both legacy tuples and new IntervalResult objects
        if intervals and hasattr(intervals[0], 'start'):
            # New IntervalResult format
            actual_intervals = [(interval.start, interval.end) for interval in intervals]
        elif intervals and isinstance(intervals[0], tuple):
            if len(intervals[0]) == 3:
                # Format: (start, end, data)
                actual_intervals = [(start, end) for start, end, _ in intervals]
            else:
                # Format: (start, end)
                actual_intervals = intervals
        else:
            actual_intervals = []
        
        actual_count = len(actual_intervals)
        actual_total = sum(end - start for start, end in actual_intervals)
    
        assert summary.interval_count == actual_count, f"Count mismatch: summary {summary.interval_count} vs actual {actual_count}"
        assert summary.total_free_length == actual_total, f"Length mismatch: summary {summary.total_free_length} vs actual {actual_total}"
        
        if actual_intervals:
            actual_largest = max(end - start for start, end in actual_intervals)
            assert summary.largest_interval_length == actual_largest, f"Largest mismatch: summary {summary.largest_interval_length} vs actual {actual_largest}"


@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted))
def test_py_boundary_summary_operations(operations: List[Tuple[int, int]]) -> None:
    """Test Python boundary summary basic operations"""
    if not PY_BOUNDARY_SUMMARY_AVAILABLE:
        return
    
    manager = PyBoundarySummary()
    
    # Apply release operations
    for start, end in operations:
        manager.release_interval(start, end)
        validate_boundary_summary_invariants(manager)
    
    # Test availability stats
    stats = manager.get_availability_stats()
    assert stats['total_free'] >= 0
    assert stats['free_chunks'] >= 0
    assert stats['fragmentation'] >= 0.0
    
    # Test advanced queries
    if stats['largest_chunk'] > 0:
        largest = manager.find_largest_available()
        assert largest is not None, "Should find largest interval when one exists"
        
        assert largest.length == stats['largest_chunk'], "Largest interval size should match summary"


@given(st.lists(st.tuples(
    st.sampled_from(['reserve', 'release']),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[1] < x[2])))
def test_py_boundary_summary_mixed_operations(operations: List[Tuple[str, int, int]]) -> None:
    """Test Python boundary summary with mixed operations"""
    if not PY_BOUNDARY_SUMMARY_AVAILABLE:
        return
    
    manager = PyBoundarySummary()
    manager.release_interval(0, 10000)  # Start with available space
    
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        else:
            manager.release_interval(start, end)
        
        validate_boundary_summary_invariants(manager)
        
        # Test that summary is consistent
        summary = manager.get_summary()
        stats = manager.get_availability_stats()
        
        assert stats['total_free'] == summary.total_free_length
        assert stats['free_chunks'] == summary.interval_count


@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted))
def test_cpp_boundary_summary_operations(operations: List[Tuple[int, int]]) -> None:
    """Test C++ boundary summary basic operations"""
    if not CPP_BOUNDARY_SUMMARY_AVAILABLE:
        return
    
    manager = CppBoundarySummary()
    
    # Apply release operations
    for start, end in operations:
        manager.release_interval(start, end)
        validate_boundary_summary_invariants(manager)
    
    # Test availability stats
    stats = manager.get_availability_stats()
    assert stats.total_free >= 0
    assert stats.free_chunks >= 0
    assert stats.fragmentation >= 0.0


def test_py_cpp_boundary_summary_equivalence():
    """Test that Python and C++ boundary summary produce equivalent results"""
    if not (PY_BOUNDARY_SUMMARY_AVAILABLE and CPP_BOUNDARY_SUMMARY_AVAILABLE):
        print("⚠️  Cannot test equivalence - missing implementation")
        return
    
    py_manager = PyBoundarySummary()
    cpp_manager = CppBoundarySummary()
    
    # Apply same operations to both
    operations = [
        ('release', 0, 1000),
        ('reserve', 100, 200),
        ('reserve', 300, 400),
        ('release', 150, 350),
        ('reserve', 600, 700),
    ]
    
    for op, start, end in operations:
        if op == 'reserve':
            py_manager.reserve_interval(start, end)
            cpp_manager.reserve_interval(start, end)
        else:
            py_manager.release_interval(start, end)
            cpp_manager.release_interval(start, end)
    
    # Compare results
    py_stats = py_manager.get_availability_stats()
    cpp_stats = cpp_manager.get_availability_stats()
    
    # Core metrics should match
    assert py_stats['total_free'] == cpp_stats.total_free, f"Total free mismatch: Python {py_stats['total_free']} vs C++ {cpp_stats.total_free}"
    assert py_stats['free_chunks'] == cpp_stats.free_chunks, f"Chunk count mismatch: Python {py_stats['free_chunks']} vs C++ {cpp_stats.free_chunks}"
    assert py_stats['largest_chunk'] == cpp_stats.largest_chunk, f"Largest chunk mismatch: Python {py_stats['largest_chunk']} vs C++ {cpp_stats.largest_chunk}"
    
    # Fragmentation should be close (may have minor floating point differences)
    frag_diff = abs(py_stats['fragmentation'] - cpp_stats.fragmentation)
    assert frag_diff < 0.001, f"Fragmentation differs too much: {frag_diff}"


def test_boundary_summary_caching():
    """Test summary caching performance optimization"""
    if not PY_BOUNDARY_SUMMARY_AVAILABLE:
        return
    
    manager = PyBoundarySummary()
    manager.release_interval(0, 1000)
    
    # First access should compute summary
    summary1 = manager.get_summary()
    perf1 = manager.get_performance_stats()
    
    # Second access should use cache
    summary2 = manager.get_summary()
    perf2 = manager.get_performance_stats()
    
    # Cache hits should increase
    assert perf2.cache_hits > perf1.cache_hits, "Cache hits should increase on repeated access"
    
    # Summaries should be identical
    assert summary1.total_free_length == summary2.total_free_length
    assert summary1.interval_count == summary2.interval_count
    
    # Modify tree to invalidate cache
    manager.reserve_interval(100, 200)
    
    # Next access should recompute
    summary3 = manager.get_summary()
    assert summary3.total_free_length != summary1.total_free_length, "Summary should update after modification"


def test_boundary_summary_advanced_queries():
    """Test advanced query functionality"""
    implementations = []
    
    if PY_BOUNDARY_SUMMARY_AVAILABLE:
        implementations.append(("Python", PyBoundarySummary()))
    if CPP_BOUNDARY_SUMMARY_AVAILABLE:
        implementations.append(("C++", CppBoundarySummary()))
    
    for impl_name, manager in implementations:
        print(f"\nTesting {impl_name} boundary summary advanced queries...")
        
        # Setup test scenario
        manager.release_interval(0, 1000)
        manager.reserve_interval(100, 200)  # Creates gaps: [0,100), [200,1000)
        manager.reserve_interval(300, 400)  # Creates: [0,100), [200,300), [400,1000)
        
        # Test best fit queries
        best_fit_50 = manager.find_best_fit(50)
        assert best_fit_50 is not None, f"{impl_name}: Should find 50-unit interval"
        
        start, end = best_fit_50.start, best_fit_50.end
        assert end - start == 50, f"{impl_name}: Best fit should be exactly 50 units"
        
        # Test largest available
        largest = manager.find_largest_available()
        assert largest is not None, f"{impl_name}: Should find largest interval"
        
        expected_largest = 600  # [400, 1000) should be largest
        assert largest.length == expected_largest, f"{impl_name}: Largest should be {expected_largest}, got {largest.length}"
        
        # Test summary statistics
        stats = manager.get_availability_stats()
        if hasattr(stats, 'total_free'):  # C++ object
            total_free = stats.total_free
            free_chunks = stats.free_chunks
        else:  # Python dict
            total_free = stats['total_free']
            free_chunks = stats['free_chunks']
        
        expected_free = 100 + 100 + 600  # Three intervals
        assert total_free == expected_free, f"{impl_name}: Total free should be {expected_free}, got {total_free}"
        assert free_chunks == 3, f"{impl_name}: Should have 3 chunks, got {free_chunks}"


def test_boundary_summary_performance():
    """Test boundary summary performance characteristics"""
    implementations = []
    
    if PY_BOUNDARY_SUMMARY_AVAILABLE:
        implementations.append(("Python", PyBoundarySummary()))
    if CPP_BOUNDARY_SUMMARY_AVAILABLE:
        implementations.append(("C++", CppBoundarySummary()))
    
    if not implementations:
        print("⚠️  No boundary summary implementations available for performance testing")
        return
    
    print(f"\n⚡ Boundary Summary Performance Test")
    print("=" * 50)
    
    for impl_name, manager in implementations:
        print(f"\nTesting {impl_name} implementation:")
        
        # Initialize
        manager.release_interval(0, 100000)
        
        # Generate operations
        num_operations = 2000  # Moderate size for property-based test
        operations = []
        for i in range(num_operations):
            op = 'reserve' if i % 2 == 0 else 'release'
            start = (i * 17) % 90000
            end = start + ((i * 13) % 1000) + 1
            operations.append((op, start, end))
        
        # Time operations
        import time
        start_time = time.perf_counter()
        
        for op, start, end in operations:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        
        operation_time = time.perf_counter() - start_time
        
        # Time summary access (O(1) with caching)
        summary_start = time.perf_counter()
        for _ in range(100):
            if hasattr(manager, 'get_availability_stats'):
                manager.get_availability_stats()
            else:
                manager.get_summary()
        summary_time = time.perf_counter() - summary_start
        
        ops_per_second = num_operations / operation_time if operation_time > 0 else 0
        avg_summary_time = summary_time / 100 * 1_000_000  # Convert to microseconds
        
        print(f"  Operations: {ops_per_second:,.0f} ops/sec")
        print(f"  Summary access: {avg_summary_time:.1f}µs average")
        
        # Get final statistics
        if hasattr(manager, 'get_availability_stats'):
            stats = manager.get_availability_stats()
            if hasattr(stats, 'fragmentation'):  # C++ object
                fragmentation = stats.fragmentation
                free_chunks = stats.free_chunks
            else:  # Python dict
                fragmentation = stats['fragmentation']
                free_chunks = stats['free_chunks']
            
            print(f"  Final state: {free_chunks} chunks, {fragmentation:.1%} fragmentation")
        
        # Performance stats
        if hasattr(manager, 'get_performance_stats'):
            perf = manager.get_performance_stats()
            if hasattr(perf, 'cache_hit_rate'):  # C++ object
                print(f"  Cache performance: {perf.cache_hit_rate:.1%} hit rate")
            else:  # Python dict
                print(f"  Cache performance: {perf['cache_hit_rate']:.1%} hit rate")


def test_boundary_summary_vs_regular_boundary():
    """Compare boundary summary with regular boundary manager"""
    if not PY_BOUNDARY_SUMMARY_AVAILABLE:
        return
    
    try:
        from boundary import IntervalManager as RegularBoundary
        regular_available = True
    except ImportError:
        print("⚠️  Regular boundary manager not available for comparison")
        return
    
    # Test same operations on both implementations
    operations = [
        ('release', 0, 1000),
        ('reserve', 100, 200),
        ('reserve', 300, 400),
        ('release', 150, 350),
    ]
    
    # Regular boundary manager
    regular_manager = RegularBoundary()
    for op, start, end in operations:
        if op == 'reserve':
            regular_manager.reserve_interval(start, end)
        else:
            regular_manager.release_interval(start, end)
    
    # Boundary summary manager
    summary_manager = PyBoundarySummary()
    for op, start, end in operations:
        if op == 'reserve':
            summary_manager.reserve_interval(start, end)
        else:
            summary_manager.release_interval(start, end)
    
    # Compare basic results
    regular_total = regular_manager.get_total_available_length()
    summary_total = summary_manager.get_total_available_length()
    
    assert regular_total == summary_total, f"Total available mismatch: regular {regular_total} vs summary {summary_total}"
    
    # Summary manager should provide additional analytics
    stats = summary_manager.get_availability_stats()
    assert 'fragmentation' in stats, "Summary manager should provide fragmentation analysis"
    assert 'utilization' in stats, "Summary manager should provide utilization analysis"


if __name__ == "__main__":
    # Run basic tests without hypothesis
    print("🧪 Boundary Summary Tests")
    print("=" * 40)
    
    test_boundary_summary_advanced_queries()
    test_boundary_summary_performance()
    
    if PY_BOUNDARY_SUMMARY_AVAILABLE and CPP_BOUNDARY_SUMMARY_AVAILABLE:
        test_py_cpp_boundary_summary_equivalence()
        print("✓ Python/C++ equivalence test passed")
    
    if PY_BOUNDARY_SUMMARY_AVAILABLE:
        test_boundary_summary_caching()
        print("✓ Caching test passed")
        
        test_boundary_summary_vs_regular_boundary()
        print("✓ Regular boundary comparison passed")
    
    print(f"\n✅ Boundary summary tests complete!")
    print(f"Run with pytest for comprehensive property-based testing.")
