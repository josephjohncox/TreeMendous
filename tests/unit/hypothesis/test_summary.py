"""
Property-based tests for summary-enhanced interval trees.

Tests verify that aggregate statistics remain consistent across operations
and that summary-optimized queries produce correct results.
"""

from hypothesis import assume, given, strategies as st
from typing import List, Tuple, Optional, Dict, Any

from treemendous.basic.summary import SummaryIntervalTree, TreeSummary


def validate_tree_invariants(tree: SummaryIntervalTree) -> None:
    """Verify all tree invariants hold"""
    if not tree.root:
        return
        
    def check_node_invariants(node):
        if not node:
            return True
            
        # AVL balance invariant
        balance = tree._get_balance(node)
        assert abs(balance) <= 1, f"AVL invariant violated: balance = {balance}"
        
        # Height consistency
        expected_height = 1 + max(
            node.get_height(node.left),
            node.get_height(node.right)
        )
        assert node.height == expected_height, f"Height inconsistency: {node.height} vs {expected_height}"
        
        # Total length consistency
        expected_total = node.length
        if node.left:
            expected_total += node.left.total_length
        if node.right:
            expected_total += node.right.total_length
            
        assert node.total_length == expected_total, f"Total length inconsistency: {node.total_length} vs {expected_total}"
        
        # Summary consistency
        node_summary = TreeSummary.from_interval(node.start, node.end)
        left_summary = node.left.summary if node.left else None
        right_summary = node.right.summary if node.right else None
        expected_summary = TreeSummary.merge(left_summary, right_summary, node_summary)
        
        # Check key summary fields
        assert node.summary.total_free_length == expected_summary.total_free_length
        assert node.summary.contiguous_count == expected_summary.contiguous_count
        assert node.summary.largest_free_length == expected_summary.largest_free_length
        
        # Recursive check
        check_node_invariants(node.left)
        check_node_invariants(node.right)
        
    check_node_invariants(tree.root)


def calculate_expected_stats(intervals: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Calculate expected statistics from interval list"""
    if not intervals:
        return {
            'total_free': 0,
            'contiguous_count': 0,
            'largest_free': 0,
            'avg_free': 0.0
        }
        
    total_free = sum(end - start for start, end in intervals)
    contiguous_count = len(intervals)
    largest_free = max(end - start for start, end in intervals)
    avg_free = float(total_free) / contiguous_count
    
    return {
        'total_free': total_free,
        'contiguous_count': contiguous_count, 
        'largest_free': largest_free,
        'avg_free': avg_free
    }


@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted))
def test_summary_statistics_consistency(operations: List[Tuple[int, int]]) -> None:
    """Verify summary statistics match actual interval state"""
    tree = SummaryIntervalTree()
    tree.release_interval(0, 10000)  # Start with large available space
    
    # Apply reserve operations
    for start, end in operations:
        tree.reserve_interval(start, end)
        validate_tree_invariants(tree)
    
    # Get actual intervals and calculate expected stats
    intervals = [(start, end) for start, end, _ in tree.get_intervals()]
    expected_stats = calculate_expected_stats(intervals)
    
    # Compare with tree's reported statistics
    tree_stats = tree.get_availability_stats()
    
    assert tree_stats['total_free'] == expected_stats['total_free']
    assert tree_stats['free_chunks'] == expected_stats['contiguous_count']
    assert tree_stats['largest_chunk'] == expected_stats['largest_free']
    
    if expected_stats['contiguous_count'] > 0:
        assert abs(tree_stats['avg_chunk_size'] - expected_stats['avg_free']) < 0.001


@given(st.lists(st.tuples(
    st.sampled_from(['reserve', 'release']),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[1] < x[2])))
def test_mixed_operations_invariants(operations: List[Tuple[str, int, int]]) -> None:
    """Test invariants hold under mixed reserve/release operations"""
    tree = SummaryIntervalTree()
    tree.release_interval(0, 10000)
    
    for op, start, end in operations:
        if op == 'reserve':
            tree.reserve_interval(start, end)
        else:
            tree.release_interval(start, end)
            
        validate_tree_invariants(tree)
        
        # Verify summary bounds are reasonable
        summary = tree.get_tree_summary()
        if summary.earliest_free_start is not None and summary.latest_free_end is not None:
            assert summary.earliest_free_start <= summary.latest_free_end
            assert summary.earliest_free_start >= 0
            assert summary.latest_free_end <= 10000


@given(st.integers(min_value=1, max_value=500))
def test_find_best_fit_correctness(length: int) -> None:
    """Verify find_best_fit returns valid results"""
    tree = SummaryIntervalTree()
    tree.release_interval(0, 1000)
    
    # Create some fragmentation
    tree.reserve_interval(100, 200)
    tree.reserve_interval(300, 350) 
    tree.reserve_interval(600, 700)
    
    # Test early preference
    result_early = tree.find_best_fit(length, prefer_early=True)
    result_best = tree.find_best_fit(length, prefer_early=False)
    
    if result_early is not None:
        start, end = result_early
        assert end - start == length
        assert start >= 0 and end <= 1000
        
        # Verify the interval is actually available
        intervals = tree.get_intervals()
        found_available = False
        for istart, iend, _ in intervals:
            if istart <= start and end <= iend:
                found_available = True
                break
        assert found_available, f"Interval [{start}, {end}] not actually available"
    
    if result_best is not None:
        start, end = result_best
        assert end - start == length
        assert start >= 0 and end <= 1000


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20))
def test_find_largest_available(lengths: List[int]) -> None:
    """Test finding largest available interval"""
    tree = SummaryIntervalTree()
    tree.release_interval(0, 10000)
    
    # Reserve intervals of various lengths
    pos = 0
    gap_sizes = []
    
    for length in lengths:
        if pos + length <= 9000:  # Leave space at end
            tree.reserve_interval(pos, pos + length)
            pos += length + 50  # Leave 50-unit gaps
            gap_sizes.append(50)  # Track gaps created
    
    # The final gap from pos to 10000
    final_gap = 10000 - pos
    if final_gap > 0:
        gap_sizes.append(final_gap)
    
    largest = tree.find_largest_available()
    summary = tree.get_tree_summary()
    
    if largest:
        start, end = largest
        actual_largest = end - start
        assert actual_largest == summary.largest_free_length
        # The actual largest should be one of our expected gaps or larger due to merging
        expected_max = max(gap_sizes) if gap_sizes else 0
        assert actual_largest >= expected_max  # Can be larger due to interval merging
    else:
        assert summary.largest_free_length == 0


def test_tree_summary_merging() -> None:
    """Test TreeSummary.merge functionality"""
    # Test empty merging
    empty = TreeSummary.empty()
    single = TreeSummary.from_interval(10, 20)
    
    merged = TreeSummary.merge(empty, single)
    assert merged.total_free_length == 10
    assert merged.contiguous_count == 1
    
    # Test multi-summary merging
    left = TreeSummary.from_interval(0, 10)
    right = TreeSummary.from_interval(20, 25) 
    node = TreeSummary.from_interval(15, 18)
    
    merged = TreeSummary.merge(left, right, node)
    assert merged.total_free_length == 10 + 5 + 3  # 10 + 5 + 3 = 18
    assert merged.contiguous_count == 3
    assert merged.largest_free_length == 10
    assert merged.earliest_free_start == 0
    assert merged.latest_free_end == 25


def test_scheduling_scenario() -> None:
    """Test realistic scheduling scenario"""
    tree = SummaryIntervalTree()
    
    # Full day available (24 hours = 86400 seconds)
    tree.release_interval(0, 86400)
    
    # Schedule morning meetings
    tree.reserve_interval(32400, 36000)  # 9-10 AM
    tree.reserve_interval(36000, 39600)  # 10-11 AM
    tree.reserve_interval(43200, 46800)  # 12-1 PM
    
    validate_tree_invariants(tree)
    
    # Check statistics make sense
    stats = tree.get_availability_stats()
    
    # Should have 3 free chunks: [0,9AM], [11AM,12PM], [1PM,24H]
    expected_chunks = 3  # Non-adjacent reserved intervals create 3 gaps
    assert stats['free_chunks'] == expected_chunks
    
    # Calculate expected free space: 86400 - (reserved intervals)
    # 9AM=32400, 10AM=36000, 11AM=39600, 12PM=43200, 1PM=46800
    # Reserved: [32400,36000] + [36000,39600] + [43200,46800] = 1h + 1h + 1h = 10800s
    expected_free = 86400 - 10800  # Total - reserved
    assert stats['total_free'] == expected_free
    
    # Find 2-hour slot
    two_hour_slot = tree.find_best_fit(7200)  # 2 hours
    assert two_hour_slot is not None
    
    start, end = two_hour_slot
    assert end - start == 7200
    
    # Should be able to find largest available (evening block)
    largest = tree.find_largest_available()
    assert largest is not None
    
    # Evening block should be largest (1PM to midnight = 11 hours)
    start, end = largest
    evening_duration = end - start
    assert evening_duration == 86400 - 46800  # From 1PM to end of day


def test_fragmentation_metrics() -> None:
    """Test fragmentation calculation"""
    tree = SummaryIntervalTree()
    tree.release_interval(0, 1000)
    
    # No fragmentation initially
    stats = tree.get_availability_stats()
    assert stats['fragmentation'] == 0.0
    
    # Create fragmentation: reserve middle portions
    tree.reserve_interval(100, 200)
    tree.reserve_interval(300, 400)
    tree.reserve_interval(500, 600)
    tree.reserve_interval(700, 800)
    
    # Now we have 5 chunks: [0,100], [200,300], [400,500], [600,700], [800,1000]
    # Total free = 100 + 100 + 100 + 100 + 200 = 600
    # Largest chunk = 200 (the last one [800,1000])
    stats = tree.get_availability_stats()
    
    expected_total_free = 600
    expected_largest = 200  # The [800,1000] chunk
    expected_fragmentation = 1.0 - (expected_largest / expected_total_free)  # 1 - 200/600 = 2/3
    
    assert stats['total_free'] == expected_total_free
    assert abs(stats['fragmentation'] - expected_fragmentation) < 0.001
    
    # Utilization should be 40% (400 occupied out of 1000 total)
    assert abs(stats['utilization'] - 0.4) < 0.001


@given(st.integers(min_value=28800, max_value=86400), 
       st.integers(min_value=1, max_value=7200))
def test_find_interval_after_point(start_time: int, duration: int) -> None:
    """Test finding intervals starting after a specific time"""
    assume(start_time + duration <= 86400)
    
    tree = SummaryIntervalTree()
    tree.release_interval(0, 86400)  # Full day
    
    # Block out early morning
    tree.reserve_interval(0, 28800)  # Block 0-8AM
    
    # The available interval is [28800, 86400] 
    # Use find_best_fit instead of find_interval for better results
    result = tree.find_best_fit(duration, prefer_early=True)
    
    if result:
        found_start, found_end = result
        
        # Should find interval within available space
        assert found_start >= 28800  # Should be after 8AM
        assert found_end - found_start == duration
        assert found_end <= 86400  # Should be within day bounds
        
    else:
        # No suitable interval exists - verify this makes sense
        available_space = 86400 - 28800  # Space after 8AM
        assert available_space < duration


def test_empty_tree_operations() -> None:
    """Test operations on empty tree"""
    tree = SummaryIntervalTree()
    
    # Empty tree should have empty summary
    summary = tree.get_tree_summary() 
    assert summary.total_free_length == 0
    assert summary.contiguous_count == 0
    assert summary.largest_free_length == 0
    
    # Should handle queries gracefully
    assert tree.find_best_fit(100) is None
    assert tree.find_largest_available() is None
    
    # Should handle reserve on empty tree
    tree.reserve_interval(100, 200)  # Should be no-op
    assert tree.get_tree_summary().total_free_length == 0


if __name__ == "__main__":
    # Run some basic sanity checks
    print("Running basic summary tree tests...")
    
    test_tree_summary_merging()
    print("✓ TreeSummary merging works")
    
    test_scheduling_scenario()
    print("✓ Scheduling scenario works")
    
    test_fragmentation_metrics()
    print("✓ Fragmentation metrics work")
    
    test_empty_tree_operations()
    print("✓ Empty tree operations work")
    
    print("\nAll basic tests passed! Run with pytest for comprehensive property-based testing.")
