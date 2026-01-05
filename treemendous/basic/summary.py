"""
Enhanced AVL Interval Tree with Summary Statistics

This implementation extends the basic AVL tree with comprehensive aggregate statistics
to enable efficient scheduling queries. Each node maintains summary information about
the free space distribution in its subtree, allowing for fast "best fit" operations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, List, Tuple, TypeVar, cast, Protocol
try:
    from treemendous.basic.base import IntervalNodeBase, IntervalNodeProtocol, IntervalTreeBase
    from treemendous.basic.protocols import CoreIntervalManagerProtocol, EnhancedIntervalManagerProtocol, IntervalResult, AvailabilityStats
except ImportError:
    from base import IntervalNodeBase, IntervalNodeProtocol, IntervalTreeBase
    from protocols import CoreIntervalManagerProtocol, EnhancedIntervalManagerProtocol, IntervalResult, AvailabilityStats


@dataclass(frozen=True)
class TreeSummary:
    """Aggregate statistics for subtree free space distribution"""
    
    # Core metrics
    total_free_length: int = 0           # Sum of all free space in subtree
    total_occupied_length: int = 0       # Sum of all occupied space in subtree
    contiguous_count: int = 0            # Number of separate free intervals
    
    # Largest available chunk
    largest_free_length: int = 0         # Size of largest contiguous free space
    largest_free_start: Optional[int] = None   # Start of largest free interval
    
    # Bounds of free space distribution  
    earliest_free_start: Optional[int] = None  # Earliest available start time
    latest_free_end: Optional[int] = None      # Latest available end time
    
    # Statistical distribution
    avg_free_length: float = 0.0         # Average size of free intervals
    free_density: float = 0.0            # Ratio of free to total space
    
    @classmethod
    def empty(cls) -> 'TreeSummary':
        """Create empty summary for leaf nodes"""
        return cls()
    
    @classmethod
    def from_interval(cls, start: int, end: int) -> 'TreeSummary':
        """Create summary for single free interval"""
        length = end - start
        return cls(
            total_free_length=length,
            total_occupied_length=0,
            contiguous_count=1,
            largest_free_length=length,
            largest_free_start=start,
            earliest_free_start=start,
            latest_free_end=end,
            avg_free_length=float(length),
            free_density=1.0
        )
    
    @classmethod  
    def merge(cls, left: Optional['TreeSummary'], right: Optional['TreeSummary'], 
              node_summary: Optional['TreeSummary'] = None) -> 'TreeSummary':
        """Merge summaries from left subtree, right subtree, and current node"""
        
        summaries = [s for s in [left, right, node_summary] if s is not None]
        if not summaries:
            return cls.empty()
            
        # Aggregate basic metrics
        total_free = sum(s.total_free_length for s in summaries)
        total_occupied = sum(s.total_occupied_length for s in summaries)
        total_count = sum(s.contiguous_count for s in summaries)
        
        # Find largest free interval across all summaries
        largest_intervals = [(s.largest_free_length, s.largest_free_start) 
                           for s in summaries if s.largest_free_length > 0]
        if largest_intervals:
            largest_length, largest_start = max(largest_intervals, key=lambda x: x[0])
        else:
            largest_length, largest_start = 0, None
            
        # Determine bounds of free space
        starts = [s.earliest_free_start for s in summaries if s.earliest_free_start is not None]
        ends = [s.latest_free_end for s in summaries if s.latest_free_end is not None]
        
        earliest_start = min(starts) if starts else None
        latest_end = max(ends) if ends else None
        
        # Calculate statistical measures
        avg_length = float(total_free) / total_count if total_count > 0 else 0.0
        total_space = total_free + total_occupied
        density = float(total_free) / total_space if total_space > 0 else 0.0
        
        return cls(
            total_free_length=total_free,
            total_occupied_length=total_occupied, 
            contiguous_count=total_count,
            largest_free_length=largest_length,
            largest_free_start=largest_start,
            earliest_free_start=earliest_start,
            latest_free_end=latest_end,
            avg_free_length=avg_length,
            free_density=density
        )


class SummaryIntervalNode(IntervalNodeBase['SummaryIntervalNode', Any]):
    """AVL tree node with comprehensive summary statistics"""
    
    def __init__(self, start: int, end: int, data: Optional[Any] = None) -> None:
        super().__init__(start, end, data)
        self.summary: TreeSummary = TreeSummary.from_interval(start, end)
        self.height: int = 1
        self.total_length: int = end - start  # For compatibility
        
    def update_stats(self) -> None:
        """Update height, total_length, and summary statistics"""
        self.update_length()
        
        # Update height (AVL tree invariant)
        self.height = 1 + max(
            self.get_height(self.left),
            self.get_height(self.right)
        )
        
        # Update total length for compatibility  
        self.total_length = self.length
        if self.left:
            self.total_length += self.left.total_length
        if self.right:
            self.total_length += self.right.total_length
            
        # Update comprehensive summary
        node_summary = TreeSummary.from_interval(self.start, self.end)
        left_summary = self.left.summary if self.left else None
        right_summary = self.right.summary if self.right else None
        
        self.summary = TreeSummary.merge(left_summary, right_summary, node_summary)
    
    @staticmethod
    def get_height(node: Optional['SummaryIntervalNode']) -> int:
        return node.height if node else 0


class SummaryIntervalTree(IntervalTreeBase[SummaryIntervalNode, Any], EnhancedIntervalManagerProtocol[Any]):
    """AVL interval tree with summary statistics for efficient scheduling"""
    
    def __init__(self, merge_fn: Optional[Callable[[Any, Any], Any]] = None) -> None:
        super().__init__(merge_fn=merge_fn)
        self.root: Optional[SummaryIntervalNode] = None
        self._managed_space_start: Optional[int] = None
        self._managed_space_end: Optional[int] = None
        
    def _print_node(self, node: SummaryIntervalNode, indent: str, prefix: str) -> None:
        s = node.summary
        print(f"{indent}{prefix}{node.start}-{node.end} "
              f"(free={s.total_free_length}, occupied={s.total_occupied_length}, "
              f"chunks={s.contiguous_count}, largest={s.largest_free_length})")
    
    def get_tree_summary(self) -> TreeSummary:
        """Get comprehensive summary of entire tree"""
        if not self.root:
            return TreeSummary.empty()
            
        base_summary = self.root.summary
        
        # Calculate occupied space if we're tracking managed bounds
        if self._managed_space_start is not None and self._managed_space_end is not None:
            total_managed = self._managed_space_end - self._managed_space_start
            occupied_space = total_managed - base_summary.total_free_length
            
            # Create enhanced summary with occupied space
            return TreeSummary(
                total_free_length=base_summary.total_free_length,
                total_occupied_length=max(0, occupied_space),
                contiguous_count=base_summary.contiguous_count,
                largest_free_length=base_summary.largest_free_length,
                largest_free_start=base_summary.largest_free_start,
                earliest_free_start=base_summary.earliest_free_start,
                latest_free_end=base_summary.latest_free_end,
                avg_free_length=base_summary.avg_free_length,
                free_density=base_summary.free_density
            )
            
        return base_summary
    
    # Implement required abstract methods
    def reserve_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Mark interval as occupied (remove from free space)"""
        self.root = self._delete_interval(self.root, start, end)
        
    def release_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Mark interval as free (add to available space)"""
        # Track managed space bounds
        self._update_managed_bounds(start, end)
        
        overlapping_nodes: List[SummaryIntervalNode] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        
        # Merge with overlapping intervals
        merged_data = data
        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)
            merged_data = self.merge_data(merged_data, node.data)
            
        # Insert merged interval
        new_node = SummaryIntervalNode(start, end, merged_data)
        self.root = self._insert(self.root, new_node)
    
    def _update_managed_bounds(self, start: int, end: int) -> None:
        """Update the bounds of space being managed by this tree"""
        if self._managed_space_start is None:
            self._managed_space_start = start
            self._managed_space_end = end
        else:
            self._managed_space_start = min(self._managed_space_start, start)
            self._managed_space_end = max(self._managed_space_end, end)
    
    def find_interval(self, start: int, length: int) -> Tuple[int, int]:
        """Find available interval using summary statistics for optimization"""
        result = self._find_interval_optimized(self.root, start, length)
        if result:
            return (result.start, result.start + length)
        raise ValueError(f"No interval of length {length} available starting from {start}")
    
    def find_best_fit(self, length: int, prefer_early: bool = True) -> Optional[Tuple[int, int]]:
        """Find best available interval of given length using summary optimization"""
        if not self.root:
            return None
            
        # Quick elimination: check if any subtree can satisfy request
        if self.root.summary.largest_free_length < length:
            return None
            
        best_node = self._find_best_fit_node(self.root, length, prefer_early)
        if best_node and (best_node.end - best_node.start) >= length:
            start_pos = best_node.start
            return (start_pos, start_pos + length)
            
        return None
    
    def find_largest_available(self) -> Optional[Tuple[int, int]]:
        """Find the largest available free interval"""
        if not self.root:
            return None
            
        summary = self.root.summary
        if summary.largest_free_length == 0:
            return None
            
        # Find the node containing the largest interval
        largest_node = self._find_node_with_largest(self.root)
        if largest_node:
            return (largest_node.start, largest_node.end)
            
        return None
    
    def get_availability_stats(self) -> dict:
        """Get comprehensive availability statistics"""
        summary = self.get_tree_summary()
        
        total_space = summary.total_free_length + summary.total_occupied_length
        
        return {
            'total_free': summary.total_free_length,
            'total_occupied': summary.total_occupied_length,
            'total_space': total_space,
            'free_chunks': summary.contiguous_count,
            'largest_chunk': summary.largest_free_length,
            'avg_chunk_size': summary.avg_free_length,
            'utilization': float(summary.total_occupied_length) / total_space if total_space > 0 else 0.0,
            'fragmentation': 1.0 - (float(summary.largest_free_length) / summary.total_free_length) 
                           if summary.total_free_length > 0 else 0.0,
            'free_density': summary.free_density,
            'bounds': (summary.earliest_free_start, summary.latest_free_end)
        }
    
    def get_intervals(self) -> List[Tuple[int, int, Optional[Any]]]:
        """Get all free intervals"""
        intervals: List[Tuple[int, int, Optional[Any]]] = []
        self._collect_intervals(self.root, intervals)
        return intervals
        
    def _collect_intervals(self, node: Optional[SummaryIntervalNode], 
                          intervals: List[Tuple[int, int, Optional[Any]]]) -> None:
        """Collect all intervals via in-order traversal"""
        if not node:
            return
            
        self._collect_intervals(node.left, intervals)
        intervals.append((node.start, node.end, node.data))
        self._collect_intervals(node.right, intervals)
    
    def _find_interval_optimized(self, node: Optional[SummaryIntervalNode], 
                                start: int, length: int) -> Optional[SummaryIntervalNode]:
        """Find interval using summary statistics to prune search space"""
        if not node:
            return None
            
        # Quick elimination: if this subtree's largest interval is too small, skip
        if node.summary.largest_free_length < length:
            return None
            
        # Quick elimination: if this subtree has no free space after start, skip  
        if node.summary.latest_free_end and node.summary.latest_free_end <= start:
            return None
            
        # Check current node
        if node.start >= start and (node.end - node.start) >= length:
            # Look for earlier interval in left subtree first
            left_result = self._find_interval_optimized(node.left, start, length) 
            return left_result if left_result else node
            
        # Search appropriate subtree
        if node.start < start:
            return self._find_interval_optimized(node.right, start, length)
        else:
            return self._find_interval_optimized(node.right, start, length)
    
    def _find_best_fit_node(self, node: Optional[SummaryIntervalNode], 
                           length: int, prefer_early: bool) -> Optional[SummaryIntervalNode]:
        """Find best fit interval using summary-guided search"""
        if not node or node.summary.largest_free_length < length:
            return None
            
        candidates = []
        
        # Check current node
        if (node.end - node.start) >= length:
            candidates.append(node)
            
        # Check subtrees that might have suitable intervals
        if node.left and node.left.summary.largest_free_length >= length:
            left_candidate = self._find_best_fit_node(node.left, length, prefer_early)
            if left_candidate:
                candidates.append(left_candidate)
                
        if node.right and node.right.summary.largest_free_length >= length:
            right_candidate = self._find_best_fit_node(node.right, length, prefer_early)
            if right_candidate:
                candidates.append(right_candidate)
        
        if not candidates:
            return None
            
        # Select best candidate based on preference
        if prefer_early:
            return min(candidates, key=lambda n: n.start)
        else:
            # Prefer best fit (smallest interval that satisfies requirement)
            return min(candidates, key=lambda n: (n.end - n.start, n.start))
    
    def _find_node_with_largest(self, node: Optional[SummaryIntervalNode]) -> Optional[SummaryIntervalNode]:
        """Find node containing the largest free interval"""
        if not node:
            return None
            
        largest_length = node.summary.largest_free_length
        
        # Check if current node has the largest interval
        if (node.end - node.start) == largest_length:
            return node
            
        # Search subtrees
        if node.left and node.left.summary.largest_free_length == largest_length:
            return self._find_node_with_largest(node.left)
            
        if node.right and node.right.summary.largest_free_length == largest_length:
            return self._find_node_with_largest(node.right)
            
        return node  # Fallback
    
    # AVL tree operations (adapted from existing implementation)
    def _insert(self, node: Optional[SummaryIntervalNode], 
                new_node: SummaryIntervalNode) -> SummaryIntervalNode:
        if not node:
            return new_node
            
        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
        else:
            node.right = self._insert(node.right, new_node)
            
        node.update_stats()
        return self._rebalance(node)
    
    def _delete_interval(self, node: Optional[SummaryIntervalNode], 
                        start: int, end: int) -> Optional[SummaryIntervalNode]:
        """Delete interval from tree (for reserve operations)"""
        if not node:
            return None
            
        if node.end <= start:
            node.right = self._delete_interval(node.right, start, end)
        elif node.start >= end:
            node.left = self._delete_interval(node.left, start, end)
        else:
            # Node overlaps with interval to delete
            nodes_to_insert = []
            
            # Create left remainder if exists
            if node.start < start:
                left_node = SummaryIntervalNode(node.start, start, node.data)
                nodes_to_insert.append(left_node)
                
            # Create right remainder if exists  
            if node.end > end:
                right_node = SummaryIntervalNode(end, node.end, node.data)
                nodes_to_insert.append(right_node)
                
            # Remove current node and process subtrees
            node = self._merge_subtrees(
                self._delete_interval(node.left, start, end),
                self._delete_interval(node.right, start, end)
            )
            
            # Insert remainder nodes
            for n in nodes_to_insert:
                node = self._insert(node, n)
                
        if node:
            node.update_stats()
            node = self._rebalance(node)
            
        return node
    
    def _delete_overlaps(self, node: Optional[SummaryIntervalNode], 
                        start: int, end: int, 
                        overlapping_nodes: List[SummaryIntervalNode]) -> Optional[SummaryIntervalNode]:
        """Find and remove overlapping intervals (for release operations)"""
        if not node:
            return None
            
        if node.end <= start:
            node.right = self._delete_overlaps(node.right, start, end, overlapping_nodes)
        elif node.start >= end:
            node.left = self._delete_overlaps(node.left, start, end, overlapping_nodes)
        else:
            # Overlap detected
            overlapping_nodes.append(node)
            node = self._merge_subtrees(
                self._delete_overlaps(node.left, start, end, overlapping_nodes),
                self._delete_overlaps(node.right, start, end, overlapping_nodes)
            )
            return node
            
        if node:
            node.update_stats()
            node = self._rebalance(node)
            
        return node
    
    def _merge_subtrees(self, left: Optional[SummaryIntervalNode], 
                       right: Optional[SummaryIntervalNode]) -> Optional[SummaryIntervalNode]:
        """Merge two subtrees"""
        if not left:
            return right
        if not right:
            return left
            
        # Find minimum node in right subtree
        min_node = self._get_min(right)
        right = self._delete_min(right)
        
        min_node.left = left
        min_node.right = right
        min_node.update_stats()
        
        return self._rebalance(min_node)
    
    def _get_min(self, node: SummaryIntervalNode) -> SummaryIntervalNode:
        """Find minimum node in subtree"""
        while node.left:
            node = node.left
        return node
    
    def _delete_min(self, node: SummaryIntervalNode) -> Optional[SummaryIntervalNode]:
        """Delete minimum node from subtree"""
        if node.left is None:
            return node.right
            
        node.left = self._delete_min(node.left)
        node.update_stats()
        return self._rebalance(node)
    
    def _rebalance(self, node: SummaryIntervalNode) -> SummaryIntervalNode:
        """Rebalance AVL tree"""
        balance = self._get_balance(node)
        
        if balance > 1:
            # Left heavy
            if self._get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            node = self._rotate_right(node)
        elif balance < -1:
            # Right heavy  
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            node = self._rotate_left(node)
            
        return node
    
    def _get_balance(self, node: Optional[SummaryIntervalNode]) -> int:
        """Get balance factor for AVL tree"""
        if not node:
            return 0
        return SummaryIntervalNode.get_height(node.left) - SummaryIntervalNode.get_height(node.right)
    
    def _rotate_left(self, z: Optional[SummaryIntervalNode]) -> Optional[SummaryIntervalNode]:
        """Left rotation for AVL rebalancing"""
        if not z or not z.right:
            return z
            
        y = z.right
        subtree = y.left
        
        y.left = z
        z.right = subtree
        
        z.update_stats()
        y.update_stats()
        
        return y
    
    def _rotate_right(self, z: Optional[SummaryIntervalNode]) -> Optional[SummaryIntervalNode]:
        """Right rotation for AVL rebalancing"""
        if not z or not z.left:
            return z
            
        y = z.left
        subtree = y.right
        
        y.right = z  
        z.left = subtree
        
        z.update_stats()
        y.update_stats()
        
        return y


# Example usage and demonstration
if __name__ == "__main__":
    tree = SummaryIntervalTree()
    
    print("=== Enhanced Interval Tree with Summary Statistics ===")
    
    # Initialize with full day available
    tree.release_interval(0, 86400)  # 24 hours in seconds
    print(f"\nInitial state:")
    tree.print_tree()
    
    stats = tree.get_availability_stats()
    print(f"\nAvailability Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Schedule some meetings
    print(f"\n=== Scheduling Operations ===")
    
    # 9 AM - 10 AM meeting
    tree.reserve_interval(32400, 36000)  
    print(f"\nAfter reserving 9-10 AM:")
    
    # 2 PM - 4 PM meeting  
    tree.reserve_interval(50400, 57600)
    print(f"After reserving 2-4 PM:")
    
    stats = tree.get_availability_stats()
    print(f"\nUpdated Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Find best available slots
    print(f"\n=== Intelligent Scheduling Queries ===")
    
    # Find 2-hour slot
    best_slot = tree.find_best_fit(7200)  # 2 hours
    if best_slot:
        start_hour = best_slot[0] // 3600
        print(f"Best 2-hour slot: {start_hour}:00-{start_hour+2}:00")
    
    # Find largest available block
    largest = tree.find_largest_available()
    if largest:
        duration_hours = (largest[1] - largest[0]) // 3600
        start_hour = largest[0] // 3600
        print(f"Largest available block: {duration_hours} hours starting at {start_hour}:00")
    
    print(f"\nTree structure with summaries:")
    tree.print_tree()
