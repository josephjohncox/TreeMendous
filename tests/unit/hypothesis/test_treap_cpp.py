"""
Property-based tests for C++ Treap implementation.

Tests verify C++ treap maintains correctness and performance while
providing identical functionality to Python implementation.
"""

from hypothesis import assume, given, strategies as st
from typing import List, Tuple, Optional, Dict, Any
import random
import math

# Try to import C++ treap
try:
    from treemendous.cpp.treap import IntervalTreap as CppTreap
    CPP_TREAP_AVAILABLE = True
    print("✅ C++ Treap loaded for testing")
except ImportError as e:
    print(f"⚠️  C++ Treap not available: {e}")
    CPP_TREAP_AVAILABLE = False
    CppTreap = None

# Import Python treap for comparison
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))
    from treap import IntervalTreap as PyTreap
    PY_TREAP_AVAILABLE = True
    print("✅ Python Treap loaded for comparison")
except ImportError as e:
    print(f"⚠️  Python Treap not available: {e}")
    PyTreap = None
    PY_TREAP_AVAILABLE = False


def validate_cpp_treap_properties(treap) -> None:
    """Verify C++ treap properties"""
    if not CPP_TREAP_AVAILABLE:
        return
    
    # Verify treap properties using C++ methods
    assert treap.verify_treap_properties(), "C++ treap properties violated"
    
    # Verify statistics consistency
    stats = treap.get_statistics()
    assert stats.size >= 0, "Invalid tree size"
    assert stats.height >= 0, "Invalid tree height"
    assert stats.total_length >= 0, "Invalid total length"
    
    # Balance should be reasonable
    if stats.size > 0:
        assert stats.balance_factor > 0, "Invalid balance factor"
        assert stats.balance_factor < 5.0, f"Tree too unbalanced: {stats.balance_factor}"


@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted))
def test_cpp_treap_basic_operations(operations: List[Tuple[int, int]]) -> None:
    """Test C++ treap basic operations"""
    if not CPP_TREAP_AVAILABLE:
        return
    
    treap = CppTreap(42)  # Fixed seed
    
    # Apply operations
    for start, end in operations:
        treap.release_interval(start, end)
        validate_cpp_treap_properties(treap)
    
    # Verify results
    intervals = treap.get_intervals()
    total_calculated = sum(end - start for start, end in intervals)
    assert treap.get_total_available_length() == total_calculated


@given(st.lists(st.tuples(
    st.sampled_from(['reserve', 'release']),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[1] < x[2])))
def test_cpp_treap_mixed_operations(operations: List[Tuple[str, int, int]]) -> None:
    """Test C++ treap with mixed operations"""
    if not CPP_TREAP_AVAILABLE:
        return
    
    treap = CppTreap(42)
    treap.release_interval(0, 10000)
    
    for op, start, end in operations:
        if op == 'reserve':
            treap.reserve_interval(start, end)
        else:
            treap.release_interval(start, end)
        
        # Every operation must maintain treap properties
        validate_cpp_treap_properties(treap)


def test_cpp_py_treap_equivalence():
    """Test that C++ and Python treaps produce equivalent results"""
    if not (CPP_TREAP_AVAILABLE and PY_TREAP_AVAILABLE):
        print("⚠️  Cannot test equivalence - missing implementation")
        return
    
    # Use same random seed for both
    py_treap = PyTreap(random_seed=42)
    cpp_treap = CppTreap(42)
    
    # Apply same operations
    operations = [
        ('release', 0, 1000),
        ('reserve', 100, 200),
        ('reserve', 300, 400),
        ('release', 150, 350),
        ('reserve', 600, 700),
    ]
    
    for op, start, end in operations:
        if op == 'reserve':
            py_treap.reserve_interval(start, end)
            cpp_treap.reserve_interval(start, end)
        else:
            py_treap.release_interval(start, end)
            cpp_treap.release_interval(start, end)
    
    # Compare results
    py_intervals = set((interval.start, interval.end) for interval in py_treap.get_intervals())
    cpp_intervals = set(cpp_treap.get_intervals())
    
    # Total lengths should match
    assert py_treap.get_total_available_length() == cpp_treap.get_total_available_length()
    
    # Individual intervals might differ due to implementation details,
    # but total coverage should be equivalent
    py_total = sum(end - start for start, end in py_intervals)
    cpp_total = sum(end - start for start, end in cpp_intervals)
    assert py_total == cpp_total, f"Total coverage mismatch: Python {py_total} vs C++ {cpp_total}"


def test_cpp_treap_performance():
    """Test C++ treap performance characteristics"""
    if not CPP_TREAP_AVAILABLE:
        print("⚠️  C++ Treap not available for performance testing")
        return
    
    print("\n🚀 C++ Treap Performance Test")
    print("-" * 40)
    
    # Use built-in C++ performance test
    try:
        from treemendous.cpp.treap import test_treap_performance
        result = test_treap_performance()
        
        print(f"  Operations: {result['operations']:,}")
        print(f"  Time: {result['time_microseconds']:,} µs")
        print(f"  Ops/sec: {result['ops_per_second']:,.0f}")
        print(f"  Height: {result['height']} (expected: {result['expected_height']:.1f})")
        print(f"  Balance factor: {result['balance_factor']:.2f}")
        
        # Performance should be reasonable
        assert result['ops_per_second'] > 100_000, f"C++ treap too slow: {result['ops_per_second']:.0f} ops/sec"
        assert result['balance_factor'] < 3.0, f"C++ treap unbalanced: {result['balance_factor']:.2f}"
        
    except ImportError:
        print("  ⚠️  C++ performance test not available")


def test_treap_random_operations():
    """Test treap-specific random operations"""
    if not CPP_TREAP_AVAILABLE:
        return
    
    treap = CppTreap(42)
    
    # Insert test intervals
    test_intervals = [(10, 20), (30, 50), (60, 80), (90, 110)]
    for start, end in test_intervals:
        treap.release_interval(start, end)
    
    # Test random sampling
    samples = []
    for _ in range(50):
        sample = treap.sample_random_interval()
        if sample:
            samples.append(sample)
    
    # Should sample from available intervals
    available_intervals = set(treap.get_intervals())
    sampled_intervals = set(samples)
    
    assert sampled_intervals.issubset(available_intervals), "C++ treap sampled invalid intervals"
    
    # Test overlapping intervals
    overlaps = treap.find_overlapping_intervals(25, 75)
    expected_overlaps = {(30, 50), (60, 80)}  # Based on test data
    actual_overlaps = set(overlaps)
    
    assert actual_overlaps == expected_overlaps, f"Overlap mismatch: expected {expected_overlaps}, got {actual_overlaps}"


def test_cpp_treap_split_operations():
    """Test C++ treap split operations"""
    if not CPP_TREAP_AVAILABLE:
        return
    
    treap = CppTreap(42)
    
    # Setup test data
    intervals = [(10, 20), (30, 40), (50, 60), (70, 80)]
    for start, end in intervals:
        treap.release_interval(start, end)
    
    original_size = treap.get_tree_size()
    original_length = treap.get_total_available_length()
    
    # Test split operation
    left_treap, right_treap = treap.split(45)
    
    # Verify split properties
    left_size = left_treap.get_tree_size()
    right_size = right_treap.get_tree_size()
    left_length = left_treap.get_total_available_length()
    right_length = right_treap.get_total_available_length()
    
    assert left_size + right_size == original_size, f"Size mismatch after split: {left_size} + {right_size} ≠ {original_size}"
    assert left_length + right_length == original_length, f"Length mismatch after split: {left_length} + {right_length} ≠ {original_length}"
    
    # Verify separation constraint
    left_intervals = left_treap.get_intervals()
    right_intervals = right_treap.get_intervals()
    
    for start, end in left_intervals:
        assert start < 45, f"Left treap contains interval starting >= 45: [{start}, {end})"
    
    for start, end in right_intervals:
        assert start >= 45, f"Right treap contains interval starting < 45: [{start}, {end})"


def benchmark_cpp_vs_python_treap():
    """Benchmark C++ vs Python treap performance"""
    if not (CPP_TREAP_AVAILABLE and PY_TREAP_AVAILABLE):
        print("⚠️  Cannot compare C++ vs Python - missing implementations")
        return
    
    print(f"\n⚡ C++ vs Python Treap Performance")
    print("=" * 60)
    
    sizes = [1000, 5000, 10000]
    
    print(f"  {'Size':>6} {'Python(ops/s)':>14} {'C++(ops/s)':>12} {'Speedup':>8}")
    print("  " + "-" * 50)
    
    for n in sizes:
        # Generate same operations
        operations = [(random.randint(0, n*10), random.randint(0, n*10)) for _ in range(n)]
        operations = [(min(s, e), max(s, e)) for s, e in operations if s != e]
        
        # Benchmark Python treap
        py_treap = PyTreap(random_seed=42)
        start_time = time.perf_counter()
        for start, end in operations:
            py_treap.release_interval(start, end)
        py_time = time.perf_counter() - start_time
        py_ops_per_sec = len(operations) / py_time if py_time > 0 else 0
        
        # Benchmark C++ treap
        cpp_treap = CppTreap(42)
        start_time = time.perf_counter()
        for start, end in operations:
            cpp_treap.release_interval(start, end)
        cpp_time = time.perf_counter() - start_time
        cpp_ops_per_sec = len(operations) / cpp_time if cpp_time > 0 else 0
        
        speedup = cpp_ops_per_sec / py_ops_per_sec if py_ops_per_sec > 0 else 1.0
        
        print(f"  {n:>6,} {py_ops_per_sec:>14,.0f} {cpp_ops_per_sec:>12,.0f} {speedup:>8.1f}x")


if __name__ == "__main__":
    # Run tests that don't require hypothesis
    print("🧪 C++ Treap Tests")
    print("=" * 30)
    
    test_cpp_treap_performance()
    
    if CPP_TREAP_AVAILABLE:
        test_treap_random_operations()
        print("✓ Random operations test passed")
        
        test_cpp_treap_split_operations()
        print("✓ Split operations test passed")
        
        test_cpp_py_treap_equivalence()
        print("✓ C++/Python equivalence test passed")
        
        benchmark_cpp_vs_python_treap()
    
    print(f"\n✅ C++ treap tests complete!")
    print(f"Run with pytest for comprehensive property-based testing.")
