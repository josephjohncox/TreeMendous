"""
Treap (Tree + Heap) Implementation for Interval Trees

A treap combines the properties of a binary search tree and a heap,
using random priorities to maintain probabilistic balance without
complex rotation logic. This provides O(log n) expected performance
for all operations with high probability.
"""

import math
import random
from collections.abc import Callable
from typing import Any, Optional

from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase
from treemendous.basic.protocols import IntervalResult
from treemendous.domain import Span, validate_coordinate, validate_length


class TreapNode(IntervalNodeBase["TreapNode", Any]):
    """Treap node combining BST ordering with heap priorities"""

    def __init__(
        self,
        start: int,
        end: int,
        data: Any | None = None,
        priority: float | None = None,
    ):
        super().__init__(start, end, data)
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

        self.height = 1 + max(self.get_height(self.left), self.get_height(self.right))

    @staticmethod
    def get_height(node: Optional["TreapNode"]) -> int:
        return node.height if node else 0

    @staticmethod
    def get_size(node: Optional["TreapNode"]) -> int:
        return node.subtree_size if node else 0


class IntervalTreap(IntervalTreeBase[TreapNode, Any]):
    """Randomized interval tree using treap structure"""

    def __init__(
        self,
        random_seed: int | None = None,
        merge_fn: Callable[[Any, Any], Any] | None = None,
        split_fn: Callable[[Any, int, int, int, int], Any] | None = None,
        can_merge: Callable[[Any | None, Any | None], bool] | None = None,
        merge_idempotent: bool = False,
        split_idempotent: bool = False,
    ):
        super().__init__(
            merge_fn=merge_fn,
            split_fn=split_fn,
            can_merge=can_merge,
            merge_idempotent=merge_idempotent,
            split_idempotent=split_idempotent,
        )
        self.root: TreapNode | None = None
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)

    def _print_node(self, node: TreapNode, indent: str, prefix: str) -> None:
        print(
            f"{indent}{prefix}[{node.start},{node.end}) p={node.priority:.3f} "
            f"(h={node.height}, size={node.subtree_size})"
        )

    def reserve_interval(self, start: int, end: int, data: Any | None = None) -> None:
        """Remove interval from available space (mark as occupied)"""
        Span(start, end)
        # Delete overlapping intervals and split as needed
        self.root = self._delete_range(self.root, start, end)

    def release_interval(self, start: int, end: int, data: Any | None = None) -> None:
        """Add interval to available space (mark as free)"""
        Span(start, end)
        # First remove any overlapping intervals, then insert merged interval
        overlapping_intervals = self._find_and_remove_overlapping(start, end)

        # Merge with overlapping intervals
        merged_start = start
        merged_end = end
        merged_data = data

        for interval_start, interval_end, interval_data in overlapping_intervals:
            merged_start = min(merged_start, interval_start)
            merged_end = max(merged_end, interval_end)
            merged_data = self.merge_data(merged_data, interval_data)

        # Insert merged interval
        new_node = TreapNode(
            merged_start, merged_end, merged_data, priority=self._rng.random()
        )
        self.root = self._insert(self.root, new_node)

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
        """Find available interval of given length starting at or after start time"""
        validate_coordinate(start, "start")
        validate_length(length)
        result = self._find_interval(self.root, start, length)
        if result:
            # If the requested start point is within the found interval, use it
            if result.start <= start and start + length <= result.end:
                return IntervalResult(
                    start=start, end=start + length, length=length, data=result.data
                )
            else:
                # Otherwise, allocate from the beginning of the found interval
                return IntervalResult(
                    start=result.start,
                    end=result.start + length,
                    length=length,
                    data=result.data,
                )
        return None

    def get_intervals(self) -> list[IntervalResult]:
        """Get all available intervals in sorted order"""
        intervals: list[tuple[int, int, Any | None]] = []
        self._inorder_traversal(self.root, intervals)
        return [
            IntervalResult(start=start, end=end, data=data)
            for start, end, data in intervals
        ]

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
        """Verify ordering, heap, non-overlap, and cached aggregates recursively."""
        return self._verify_node(self.root, None, None)[0]

    def verify_properties(self) -> bool:
        """Verify implementation-specific properties (RandomizedProtocol requirement)"""
        return self.verify_treap_properties()

    # Core treap operations
    def _insert(self, node: TreapNode | None, new_node: TreapNode) -> TreapNode:
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

    def _delete(
        self, node: TreapNode | None, start: int, end: int
    ) -> tuple[TreapNode | None, bool]:
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

    def _delete_node(self, node: TreapNode) -> TreapNode | None:
        """Delete specific node using priority-based rotations"""
        if not node.left:
            return node.right
        elif not node.right:
            return node.left
        else:
            # Rotate with child having higher priority, then delete recursively
            if node.left.priority > node.right.priority:
                node = self._rotate_right(node)
                assert node.right is not None
                node.right = self._delete_node(node.right)
            else:
                node = self._rotate_left(node)
                assert node.left is not None
                node.left = self._delete_node(node.left)

            if node:
                node.update_stats()
            return node

    def _delete_range(
        self, node: TreapNode | None, start: int, end: int
    ) -> TreapNode | None:
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
                left_data = self.split_data(
                    node.data, node.start, node.end, node.start, start
                )
                left_remainder = TreapNode(
                    node.start, start, left_data, priority=node.priority * 0.5
                )
                nodes_to_insert.append(left_remainder)

            # Create right remainder if needed
            if node.end > end:
                # Use a priority lower than the original node to maintain heap property
                right_data = self.split_data(
                    node.data, node.start, node.end, end, node.end
                )
                right_remainder = TreapNode(
                    end, node.end, right_data, priority=node.priority * 0.5
                )
                nodes_to_insert.append(right_remainder)

            # Delete current node and process subtrees
            node = self._merge_subtrees(
                self._delete_range(node.left, start, end),
                self._delete_range(node.right, start, end),
            )

            # Insert remainders
            for remainder in nodes_to_insert:
                node = self._insert(node, remainder)

        if node:
            node.update_stats()
        return node

    def _find_and_remove_overlapping(
        self, start: int, end: int
    ) -> list[tuple[int, int, Any | None]]:
        """Find and remove all overlapping intervals, return their ranges"""
        overlapping: list[tuple[int, int, Any | None]] = []
        self.root = self._collect_and_remove_overlapping(
            self.root, start, end, overlapping
        )
        return overlapping

    def _collect_and_remove_overlapping(
        self,
        node: TreapNode | None,
        start: int,
        end: int,
        overlapping: list[tuple[int, int, Any | None]],
    ) -> TreapNode | None:
        """Helper for finding and removing overlapping intervals"""
        if not node:
            return None

        if node.end <= start:
            node.right = self._collect_and_remove_overlapping(
                node.right, start, end, overlapping
            )
        elif node.start >= end:
            node.left = self._collect_and_remove_overlapping(
                node.left, start, end, overlapping
            )
        else:
            # Overlap found
            overlapping.append((node.start, node.end, node.data))
            # Remove this node and continue in subtrees
            return self._merge_subtrees(
                self._collect_and_remove_overlapping(
                    node.left, start, end, overlapping
                ),
                self._collect_and_remove_overlapping(
                    node.right, start, end, overlapping
                ),
            )

        if node:
            node.update_stats()
        return node

    def _find_interval(
        self, node: TreapNode | None, start: int, length: int
    ) -> TreapNode | None:
        """Find interval that can accommodate request"""
        if not node:
            return None

        # Check if current node works
        node_length = node.end - node.start
        allocation_start = max(start, node.start)
        if node_length >= length and allocation_start + length <= node.end:
            left_result = self._find_interval(node.left, start, length)
            return left_result if left_result else node

        # Search both subtrees if current node doesn't work
        # Try left subtree first (earlier intervals)
        left_result = self._find_interval(node.left, start, length)
        if left_result:
            return left_result

        # Try right subtree
        return self._find_interval(node.right, start, length)

    def _merge_subtrees(
        self, left: TreapNode | None, right: TreapNode | None
    ) -> TreapNode | None:
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

    def _inorder_traversal(
        self, node: TreapNode | None, result: list[tuple[int, int, Any | None]]
    ) -> None:
        """In-order traversal for sorted interval collection"""
        if not node:
            return

        self._inorder_traversal(node.left, result)
        result.append((node.start, node.end, node.data))
        self._inorder_traversal(node.right, result)

    def _verify_bst_property(self, node: TreapNode | None) -> bool:
        """Compatibility wrapper for the recursive propagated-bounds verifier."""
        return self._verify_node(node, None, None)[0]

    def _verify_node(
        self, node: TreapNode | None, lower: int | None, upper: int | None
    ) -> tuple[bool, int, int, int, int | None, int | None]:
        if node is None:
            return True, 0, 0, 0, None, None
        if (lower is not None and node.start <= lower) or (
            upper is not None and node.start >= upper
        ):
            return False, 0, 0, 0, None, None
        if node.start >= node.end:
            return False, 0, 0, 0, None, None
        (
            left_ok,
            left_size,
            left_height,
            left_total,
            left_min_start,
            left_max_end,
        ) = self._verify_node(node.left, lower, node.start)
        (
            right_ok,
            right_size,
            right_height,
            right_total,
            right_min_start,
            right_max_end,
        ) = self._verify_node(node.right, node.start, upper)
        # Propagated endpoint bounds catch a deep descendant that crosses an
        # ancestor even when the immediate child does not overlap it.
        if left_max_end is not None and left_max_end > node.start:
            left_ok = False
        if right_min_start is not None and node.end > right_min_start:
            right_ok = False
        expected_size = 1 + left_size + right_size
        expected_height = 1 + max(left_height, right_height)
        expected_total = node.end - node.start + left_total + right_total
        valid = (
            left_ok
            and right_ok
            and (not node.left or node.left.priority <= node.priority)
            and (not node.right or node.right.priority <= node.priority)
            and node.subtree_size == expected_size
            and node.height == expected_height
            and node.total_length == expected_total
        )
        minimum_start = left_min_start if left_min_start is not None else node.start
        maximum_end = right_max_end if right_max_end is not None else node.end
        return (
            valid,
            expected_size,
            expected_height,
            expected_total,
            minimum_start,
            maximum_end,
        )

    def _verify_heap_property(self, node: TreapNode | None) -> bool:
        """Verify heap property (parent priority ≥ child priorities)"""
        if not node:
            return True

        if node.left and node.left.priority > node.priority:
            return False
        if node.right and node.right.priority > node.priority:
            return False

        return self._verify_heap_property(node.left) and self._verify_heap_property(
            node.right
        )

    # Additional treap-specific operations
    def _new_related(self) -> "IntervalTreap":
        result = IntervalTreap(
            random_seed=self._random_seed,
            merge_fn=self.merge_fn,
            split_fn=self.split_fn,
            can_merge=self.can_merge_fn,
            merge_idempotent=self.merge_idempotent,
            split_idempotent=self.split_idempotent,
        )
        result._rng.setstate(self._rng.getstate())
        return result

    def split(self, key: int) -> tuple["IntervalTreap", "IntervalTreap"]:
        """Return deterministic, independent treaps split at a start key."""
        validate_coordinate(key, "key")
        left_treap = self._new_related()
        right_treap = self._new_related()
        for interval in self.get_intervals():
            target = left_treap if interval.start < key else right_treap
            target.release_interval(interval.start, interval.end, interval.data)
        return left_treap, right_treap

    def _split_at_key(
        self, node: TreapNode | None, key: int
    ) -> tuple[TreapNode | None, TreapNode | None]:
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

    def merge_treap(self, other: "IntervalTreap") -> "IntervalTreap":
        """Return a non-mutating ordered merge of two disjoint treaps."""
        left = self.get_intervals()
        right = other.get_intervals()
        if left and right and left[-1].end > right[0].start:
            raise ValueError(
                "treap merge requires all left intervals to precede right intervals"
            )
        result = self._new_related()
        for interval in (*left, *right):
            result.release_interval(interval.start, interval.end, interval.data)
        return result

    def sample_random_interval(self) -> IntervalResult | None:
        """Sample random interval from treap with uniform probability"""
        if not self.root:
            return None

        target_index = self._rng.randint(0, self.root.subtree_size - 1)
        node = self._select_kth_interval(self.root, target_index)

        return (
            IntervalResult(start=node.start, end=node.end, data=node.data)
            if node
            else None
        )

    def _select_kth_interval(self, node: TreapNode | None, k: int) -> TreapNode | None:
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

    def _get_rank(self, node: TreapNode | None, start: int, end: int) -> int:
        """Helper for rank computation"""
        if not node:
            return 0

        if start < node.start or (start == node.start and end < node.end):
            return self._get_rank(node.left, start, end)
        elif start == node.start and end == node.end:
            return TreapNode.get_size(node.left)
        else:
            return (
                TreapNode.get_size(node.left)
                + 1
                + self._get_rank(node.right, start, end)
            )

    def find_overlapping_intervals(self, start: int, end: int) -> list[tuple[int, int]]:
        """Find all intervals overlapping with query range"""
        Span(start, end)
        result: list[tuple[int, int]] = []
        self._find_overlapping(self.root, start, end, result)
        return result

    def _find_overlapping(
        self,
        node: TreapNode | None,
        start: int,
        end: int,
        result: list[tuple[int, int]],
    ) -> None:
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

    def get_statistics(self) -> dict[str, float]:
        """Get treap-specific performance statistics"""
        if not self.root:
            return {
                "size": 0,
                "height": 0,
                "expected_height": 0,
                "balance_factor": 1.0,
                "total_length": 0,
            }

        size = self.root.subtree_size
        actual_height = self.root.height
        expected_height = math.log2(size + 1)
        balance_factor = actual_height / expected_height if expected_height > 0 else 1.0

        return {
            "size": size,
            "height": actual_height,
            "expected_height": expected_height,
            "balance_factor": balance_factor,
            "total_length": self.root.total_length,
            "avg_interval_length": self.root.total_length / size if size > 0 else 0,
        }


# Example usage and testing
