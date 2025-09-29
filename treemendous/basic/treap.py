"""
Treap (Tree + Heap) Implementation for Interval Trees

A treap combines the properties of a binary search tree and a heap,
using random priorities to maintain probabilistic balance without
complex rotation logic. This provides O(log n) expected performance
for all operations with high probability.
"""

import random
import math
from typing import Optional, List, Tuple, TypeVar, Generic, Dict

try:
    from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase
    from treemendous.basic.protocols import CoreIntervalManagerProtocol, RandomizedProtocol, IntervalResult, PerformanceStats
except ImportError:
    from base import IntervalNodeBase, IntervalTreeBase
    from protocols import CoreIntervalManagerProtocol, RandomizedProtocol, IntervalResult, PerformanceStats


class TreapNode(IntervalNodeBase['TreapNode', None]):
    """Treap node combining BST ordering with heap priorities"""
    
    def __init__(self, start: int, end: int, priority: Optional[float] = None):
        super().__init__(start, end)
        self.priority = priority if priority is not None else random.random()
        self.height = 1
        self.total_length = self.length
        
        # Treap-specific statistics
        self.subtree_size = 1
        
    def update_stats(self) -> None:
        """Update height, total length, and subtree size"""
        self.update_length()
        self.total_length = self.length
        self.subtree_size = 1
        
        if self.left:
            self.total_length += self.left.total_length
            self.subtree_size += self.left.subtree_size
        if self.right:
            self.total_length += self.right.total_length
            self.subtree_size += self.right.subtree_size
            
        self.height = 1 + max(
            self.get_height(self.left),
            self.get_height(self.right)
        )
    
    def length(self) -> int:
        """Get interval length"""
        return self.end - self.start
    
    @staticmethod
    def get_height(node: Optional['TreapNode']) -> int:
        return node.height if node else 0
    
    @staticmethod
    def get_size(node: Optional['TreapNode']) -> int:
        return node.subtree_size if node else 0


class IntervalTreap(IntervalTreeBase[TreapNode, None], CoreIntervalManagerProtocol[None], RandomizedProtocol):
    """Randomized interval tree using treap structure"""
    
    def __init__(self, random_seed: Optional[int] = None):
        super().__init__()
        self.root: Optional[TreapNode] = None
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def _print_node(self, node: TreapNode, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}[{node.start},{node.end}) p={node.priority:.3f} "
              f"(h={node.height}, size={node.subtree_size})")
    
    def reserve_interval(self, start: int, end: int, data=None) -> None:
        """Remove interval from available space (mark as occupied)"""
        # Delete overlapping intervals and split as needed
        self.root = self._delete_range(self.root, start, end)
    
    def release_interval(self, start: int, end: int, data=None) -> None:
        """Add interval to available space (mark as free)"""
        # First remove any overlapping intervals, then insert merged interval
        overlapping_intervals = self._find_and_remove_overlapping(start, end)
        
        # Merge with overlapping intervals
        merged_start = start
        merged_end = end
        
        for interval_start, interval_end in overlapping_intervals:
            merged_start = min(merged_start, interval_start)
            merged_end = max(merged_end, interval_end)
        
        # Insert merged interval
        new_node = TreapNode(merged_start, merged_end)
        self.root = self._insert(self.root, new_node)
    
    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        """Find available interval of given length starting at or after start time"""
        result = self._find_interval(self.root, start, length)
        if result:
            # If the requested start point is within the found interval, use it
            if result.start <= start and start + length <= result.end:
                return IntervalResult(start=start, end=start + length, length=length)
            else:
                # Otherwise, allocate from the beginning of the found interval
                return IntervalResult(start=result.start, end=result.start + length, length=length)
        return None
    
    def get_intervals(self) -> List[IntervalResult]:
        """Get all available intervals in sorted order"""
        intervals = []
        self._inorder_traversal(self.root, intervals)
        return [IntervalResult(start=start, end=end, data=data) for start, end, data in intervals]
    
    def get_total_available_length(self) -> int:
        """Get total available space"""
        return self.root.total_length if self.root else 0
    
    def get_tree_size(self) -> int:
        """Get number of intervals in treap"""
        return self.root.subtree_size if self.root else 0
    
    def get_expected_height(self) -> float:
        """Get expected height based on treap theory"""
        size = self.get_tree_size()
        return math.log2(size + 1) if size > 0 else 0
    
    def verify_treap_properties(self) -> bool:
        """Verify BST and heap properties are maintained"""
        return self._verify_bst_property(self.root) and self._verify_heap_property(self.root)
    
    def verify_properties(self) -> bool:
        """Verify implementation-specific properties (RandomizedProtocol requirement)"""
        return self.verify_treap_properties()
    
    # Core treap operations
    def _insert(self, node: Optional[TreapNode], new_node: TreapNode) -> TreapNode:
        """Insert maintaining BST and heap properties"""
        if not node:
            return new_node
        
        # BST insertion by start time
        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
            # Rotate right if heap property violated
            if node.left and node.left.priority > node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert(node.right, new_node)
            # Rotate left if heap property violated
            if node.right and node.right.priority > node.priority:
                node = self._rotate_left(node)
        
        # Update stats and return
        if node:
            node.update_stats()
        return node
    
    def _delete(self, node: Optional[TreapNode], start: int, end: int) -> Tuple[Optional[TreapNode], bool]:
        """Delete interval maintaining treap properties"""
        if not node:
            return None, False
        
        if start < node.start or (start == node.start and end < node.end):
            node.left, deleted = self._delete(node.left, start, end)
            if node:
                node.update_stats()
            return node, deleted
        elif start > node.start or end > node.end:
            node.right, deleted = self._delete(node.right, start, end)
            if node:
                node.update_stats()
            return node, deleted
        else:
            # Found exact match - delete this node
            return self._delete_node(node), True
    
    def _delete_node(self, node: TreapNode) -> Optional[TreapNode]:
        """Delete specific node using priority-based rotations"""
        if not node.left:
            return node.right
        elif not node.right:
            return node.left
        else:
            # Rotate with child having higher priority, then delete recursively
            if node.left.priority > node.right.priority:
                node = self._rotate_right(node)
                node.right = self._delete_node(node.right)
            else:
                node = self._rotate_left(node)
                node.left = self._delete_node(node.left)
            
            if node:
                node.update_stats()
            return node
    
    def _delete_range(self, node: Optional[TreapNode], start: int, end: int) -> Optional[TreapNode]:
        """Delete all intervals overlapping with [start, end)"""
        if not node:
            return None
        
        # Check for overlap
        if node.end <= start:
            # No overlap, search right
            node.right = self._delete_range(node.right, start, end)
        elif node.start >= end:
            # No overlap, search left
            node.left = self._delete_range(node.left, start, end)
        else:
            # Overlap - need to handle this node
            nodes_to_insert = []
            
            # Create left remainder if needed
            if node.start < start:
                # Use a priority lower than the original node to maintain heap property
                left_remainder = TreapNode(node.start, start, priority=node.priority * 0.5)
                nodes_to_insert.append(left_remainder)
            
            # Create right remainder if needed
            if node.end > end:
                # Use a priority lower than the original node to maintain heap property
                right_remainder = TreapNode(end, node.end, priority=node.priority * 0.5)
                nodes_to_insert.append(right_remainder)
            
            # Delete current node and process subtrees
            node = self._merge_subtrees(
                self._delete_range(node.left, start, end),
                self._delete_range(node.right, start, end)
            )
            
            # Insert remainders
            for remainder in nodes_to_insert:
                node = self._insert(node, remainder)
        
        if node:
            node.update_stats()
        return node
    
    def _find_and_remove_overlapping(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Find and remove all overlapping intervals, return their ranges"""
        overlapping = []
        self.root = self._collect_and_remove_overlapping(self.root, start, end, overlapping)
        return overlapping
    
    def _collect_and_remove_overlapping(self, node: Optional[TreapNode], start: int, end: int,
                                       overlapping: List[Tuple[int, int]]) -> Optional[TreapNode]:
        """Helper for finding and removing overlapping intervals"""
        if not node:
            return None
        
        if node.end <= start:
            node.right = self._collect_and_remove_overlapping(node.right, start, end, overlapping)
        elif node.start >= end:
            node.left = self._collect_and_remove_overlapping(node.left, start, end, overlapping)
        else:
            # Overlap found
            overlapping.append((node.start, node.end))
            # Remove this node and continue in subtrees
            return self._merge_subtrees(
                self._collect_and_remove_overlapping(node.left, start, end, overlapping),
                self._collect_and_remove_overlapping(node.right, start, end, overlapping)
            )
        
        if node:
            node.update_stats()
        return node
    
    def _find_interval(self, node: Optional[TreapNode], start: int, length: int) -> Optional[TreapNode]:
        """Find interval that can accommodate request"""
        if not node:
            return None
        
        # Check if current node works
        node_length = node.end - node.start
        if node_length >= length:
            # Node is large enough - check if it can accommodate the request
            if node.end >= start + length:  # Can fit the request starting at 'start' or later
                # Check left subtree for earlier option
                left_result = self._find_interval(node.left, start, length)
                return left_result if left_result else node
        
        # Search both subtrees if current node doesn't work
        # Try left subtree first (earlier intervals)
        left_result = self._find_interval(node.left, start, length)
        if left_result:
            return left_result
        
        # Try right subtree
        return self._find_interval(node.right, start, length)
    
    def _merge_subtrees(self, left: Optional[TreapNode], right: Optional[TreapNode]) -> Optional[TreapNode]:
        """Merge two treap subtrees"""
        if not left:
            return right
        if not right:
            return left
        
        # Choose root based on priority (heap property)
        if left.priority > right.priority:
            left.right = self._merge_subtrees(left.right, right)
            left.update_stats()
            return left
        else:
            right.left = self._merge_subtrees(left, right.left)
            right.update_stats()
            return right
    
    def _rotate_left(self, node: TreapNode) -> TreapNode:
        """Left rotation maintaining treap properties"""
        if not node.right:
            return node
        
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        
        node.update_stats()
        new_root.update_stats()
        return new_root
    
    def _rotate_right(self, node: TreapNode) -> TreapNode:
        """Right rotation maintaining treap properties"""
        if not node.left:
            return node
        
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        
        node.update_stats()
        new_root.update_stats()
        return new_root
    
    def _inorder_traversal(self, node: Optional[TreapNode], result: List[Tuple[int, int, None]]) -> None:
        """In-order traversal for sorted interval collection"""
        if not node:
            return
        
        self._inorder_traversal(node.left, result)
        result.append((node.start, node.end, None))
        self._inorder_traversal(node.right, result)
    
    def _verify_bst_property(self, node: Optional[TreapNode]) -> bool:
        """Verify binary search tree property"""
        if not node:
            return True
        
        if node.left and node.left.start >= node.start:
            return False
        if node.right and node.right.start < node.start:
            return False
        
        return (self._verify_bst_property(node.left) and 
                self._verify_bst_property(node.right))
    
    def _verify_heap_property(self, node: Optional[TreapNode]) -> bool:
        """Verify heap property (parent priority ≥ child priorities)"""
        if not node:
            return True
        
        if node.left and node.left.priority > node.priority:
            return False
        if node.right and node.right.priority > node.priority:
            return False
        
        return (self._verify_heap_property(node.left) and 
                self._verify_heap_property(node.right))
    
    # Additional treap-specific operations
    def split(self, key: int) -> Tuple['IntervalTreap', 'IntervalTreap']:
        """Split treap at given key into two treaps"""
        left_treap = IntervalTreap()
        right_treap = IntervalTreap()
        
        left_treap.root, right_treap.root = self._split_at_key(self.root, key)
        
        return left_treap, right_treap
    
    def _split_at_key(self, node: Optional[TreapNode], key: int) -> Tuple[Optional[TreapNode], Optional[TreapNode]]:
        """Split treap at key"""
        if not node:
            return None, None
        
        if node.start < key:
            left, right = self._split_at_key(node.right, key)
            node.right = left
            node.update_stats()
            return node, right
        else:
            left, right = self._split_at_key(node.left, key)
            node.left = right
            node.update_stats()
            return left, node
    
    def merge_treap(self, other: 'IntervalTreap') -> 'IntervalTreap':
        """Merge this treap with another treap"""
        result = IntervalTreap()
        result.root = self._merge_subtrees(self.root, other.root)
        return result
    
    def sample_random_interval(self) -> Optional[IntervalResult]:
        """Sample random interval from treap with uniform probability"""
        if not self.root:
            return None
        
        target_index = random.randint(0, self.root.subtree_size - 1)
        node = self._select_kth_interval(self.root, target_index)
        
        return IntervalResult(start=node.start, end=node.end) if node else None
    
    def _select_kth_interval(self, node: Optional[TreapNode], k: int) -> Optional[TreapNode]:
        """Select k-th interval in sorted order (0-indexed)"""
        if not node:
            return None
        
        left_size = TreapNode.get_size(node.left)
        
        if k < left_size:
            return self._select_kth_interval(node.left, k)
        elif k == left_size:
            return node
        else:
            return self._select_kth_interval(node.right, k - left_size - 1)
    
    def get_rank(self, start: int, end: int) -> int:
        """Get rank (position) of interval in sorted order"""
        return self._get_rank(self.root, start, end)
    
    def _get_rank(self, node: Optional[TreapNode], start: int, end: int) -> int:
        """Helper for rank computation"""
        if not node:
            return 0
        
        if start < node.start or (start == node.start and end < node.end):
            return self._get_rank(node.left, start, end)
        elif start == node.start and end == node.end:
            return TreapNode.get_size(node.left)
        else:
            return TreapNode.get_size(node.left) + 1 + self._get_rank(node.right, start, end)
    
    def find_overlapping_intervals(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Find all intervals overlapping with query range"""
        result = []
        self._find_overlapping(self.root, start, end, result)
        return result
    
    def _find_overlapping(self, node: Optional[TreapNode], start: int, end: int, 
                         result: List[Tuple[int, int]]) -> None:
        """Collect overlapping intervals"""
        if not node:
            return
        
        # Check overlap with current node
        if node.start < end and node.end > start:
            result.append((node.start, node.end))
        
        # Search subtrees
        if node.left:
            self._find_overlapping(node.left, start, end, result)
        if node.right:
            self._find_overlapping(node.right, start, end, result)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get treap-specific performance statistics"""
        if not self.root:
            return {
                'size': 0,
                'height': 0,
                'expected_height': 0,
                'balance_factor': 1.0,
                'total_length': 0
            }
        
        size = self.root.subtree_size
        actual_height = self.root.height
        expected_height = math.log2(size + 1)
        balance_factor = actual_height / expected_height if expected_height > 0 else 1.0
        
        return {
            'size': size,
            'height': actual_height,
            'expected_height': expected_height,
            'balance_factor': balance_factor,
            'total_length': self.root.total_length,
            'avg_interval_length': self.root.total_length / size if size > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    import math
    
    print("🌳 Treap Interval Tree Demonstration")
    print("=" * 40)
    
    treap = IntervalTreap(random_seed=42)
    
    # Test basic operations
    print("Inserting intervals...")
    intervals = [(10, 20), (30, 40), (15, 25), (50, 60), (5, 15)]
    
    for start, end in intervals:
        treap.release_interval(start, end)
    
    print(f"Tree size: {treap.get_tree_size()}")
    print(f"Total length: {treap.get_total_available_length()}")
    
    stats = treap.get_statistics()
    print(f"Height: {stats['height']} (expected: {stats['expected_height']:.1f})")
    print(f"Balance factor: {stats['balance_factor']:.2f}")
    
    print(f"\nTree structure:")
    treap.print_tree()
    
    # Test properties
    print(f"\nTreap properties verified: {treap.verify_treap_properties()}")
    
    # Test operations
    print(f"\nOverlapping with [12, 35]: {treap.find_overlapping_intervals(12, 35)}")
    
    # Test random sampling
    print(f"\nRandom samples:")
    for _ in range(3):
        sample = treap.sample_random_interval()
        print(f"  {sample}")
    
    # Test split operation
    print(f"\nSplitting at key 25:")
    left_treap, right_treap = treap.split(25)
    print(f"  Left treap: {left_treap.get_tree_size()} intervals")
    print(f"  Right treap: {right_treap.get_tree_size()} intervals")
    
    print(f"\n✅ Treap demonstration complete!")
