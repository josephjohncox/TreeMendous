"""
Enhanced AVL Interval Tree with Summary Statistics

This implementation extends the basic AVL tree with comprehensive aggregate statistics
to enable efficient scheduling queries. Each node maintains summary information about
the free space distribution in its subtree, allowing for fast "best fit" operations.
"""

from dataclasses import dataclass
from typing import Optional

from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase
from treemendous.domain import ManagedDomain, Span, validate_coordinate, validate_length


@dataclass(frozen=True)
class TreeSummary:
    """Aggregate statistics for subtree free space distribution"""

    # Core metrics
    total_free_length: int = 0  # Sum of all free space in subtree
    total_occupied_length: int = 0  # Sum of all occupied space in subtree
    contiguous_count: int = 0  # Number of separate free intervals

    # Largest available chunk
    largest_free_length: int = 0  # Size of largest contiguous free space
    largest_free_start: int | None = None  # Start of largest free interval

    # Bounds of free space distribution
    earliest_free_start: int | None = None  # Earliest available start time
    latest_free_end: int | None = None  # Latest available end time

    # Statistical distribution
    avg_free_length: float = 0.0  # Average size of free intervals
    free_density: float = 0.0  # Ratio of free to total space

    @classmethod
    def empty(cls) -> "TreeSummary":
        """Create empty summary for leaf nodes"""
        return cls()

    @classmethod
    def from_interval(cls, start: int, end: int) -> "TreeSummary":
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
            avg_free_length=length,
            free_density=1.0,
        )

    @classmethod
    def merge(
        cls,
        left: Optional["TreeSummary"],
        right: Optional["TreeSummary"],
        node_summary: Optional["TreeSummary"] = None,
    ) -> "TreeSummary":
        """Merge summaries from left subtree, right subtree, and current node"""

        summaries = [s for s in [left, right, node_summary] if s is not None]
        if not summaries:
            return cls.empty()

        # Aggregate basic metrics
        total_free = sum(s.total_free_length for s in summaries)
        total_occupied = sum(s.total_occupied_length for s in summaries)
        total_count = sum(s.contiguous_count for s in summaries)

        # Find largest free interval across all summaries
        largest_intervals = [
            (s.largest_free_length, s.largest_free_start)
            for s in summaries
            if s.largest_free_length > 0
        ]
        if largest_intervals:
            largest_length, largest_start = max(largest_intervals, key=lambda x: x[0])
        else:
            largest_length, largest_start = 0, None

        # Determine bounds of free space
        starts = [
            s.earliest_free_start
            for s in summaries
            if s.earliest_free_start is not None
        ]
        ends = [s.latest_free_end for s in summaries if s.latest_free_end is not None]

        earliest_start = min(starts) if starts else None
        latest_end = max(ends) if ends else None

        # Calculate statistical measures
        avg_length = total_free / total_count if total_count > 0 else 0.0
        total_space = total_free + total_occupied
        density = total_free / total_space if total_space > 0 else 0.0

        return cls(
            total_free_length=total_free,
            total_occupied_length=total_occupied,
            contiguous_count=total_count,
            largest_free_length=largest_length,
            largest_free_start=largest_start,
            earliest_free_start=earliest_start,
            latest_free_end=latest_end,
            avg_free_length=avg_length,
            free_density=density,
        )


class SummaryIntervalNode(IntervalNodeBase["SummaryIntervalNode"]):
    """AVL tree node with comprehensive summary statistics"""

    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)
        self.summary: TreeSummary = TreeSummary.from_interval(start, end)
        self.height: int = 1
        self.total_length: int = end - start

    def update_stats(self) -> None:
        """Update height, total_length, and summary statistics"""
        self.update_length()

        # Update height (AVL tree invariant)
        self.height = 1 + max(self.get_height(self.left), self.get_height(self.right))

        # Update the subtree measure.
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
    def get_height(node: Optional["SummaryIntervalNode"]) -> int:
        return node.height if node else 0


class SummaryIntervalTree(IntervalTreeBase[SummaryIntervalNode]):
    """AVL interval tree with summary statistics for efficient scheduling"""

    def __init__(
        self,
        managed_domain: ManagedDomain | Span | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.root: SummaryIntervalNode | None = None
        self._managed_domain = (
            managed_domain
            if isinstance(managed_domain, ManagedDomain)
            else (ManagedDomain(managed_domain) if managed_domain is not None else None)
        )
        self._domain_explicit = managed_domain is not None

    def _print_node(self, node: SummaryIntervalNode, indent: str, prefix: str) -> None:
        s = node.summary
        print(
            f"{indent}{prefix}{node.start}-{node.end} "
            f"(free={s.total_free_length}, occupied={s.total_occupied_length}, "
            f"chunks={s.contiguous_count}, largest={s.largest_free_length})"
        )

    def get_tree_summary(self) -> TreeSummary:
        """Derive analytics from primitive free totals and a normalized domain."""
        base_summary = self.root.summary if self.root else TreeSummary.empty()
        if self._managed_domain is None:
            return base_summary
        total_managed = self._managed_domain.measure
        occupied = total_managed - base_summary.total_free_length
        bounds = self._managed_domain.bounds
        return TreeSummary(
            total_free_length=base_summary.total_free_length,
            total_occupied_length=occupied,
            contiguous_count=base_summary.contiguous_count,
            largest_free_length=base_summary.largest_free_length,
            largest_free_start=base_summary.largest_free_start,
            earliest_free_start=(
                base_summary.earliest_free_start
                if base_summary.earliest_free_start is not None
                else bounds[0]
            ),
            latest_free_end=(
                base_summary.latest_free_end
                if base_summary.latest_free_end is not None
                else bounds[1]
            ),
            avg_free_length=base_summary.avg_free_length,
            free_density=base_summary.total_free_length / total_managed,
        )

    # Implement required abstract methods
    def reserve_interval(self, start: int, end: int) -> None:
        """Mark interval as occupied (remove from free space)"""
        span = Span(start, end)
        if self._domain_explicit:
            assert self._managed_domain is not None
            if not self._managed_domain.contains(span):
                raise ValueError("span must be contained in the managed domain")
        self.root = self._delete_interval(self.root, start, end)

    def release_interval(self, start: int, end: int) -> None:
        """Mark interval as free (add to available space)"""
        span = Span(start, end)
        if self._domain_explicit:
            assert self._managed_domain is not None
            if not self._managed_domain.contains(span):
                raise ValueError("span must be contained in the managed domain")

        overlapping_nodes: list[SummaryIntervalNode] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)

        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)

        new_node = SummaryIntervalNode(start, end)
        self.root = self._insert(self.root, new_node)
        if not self._domain_explicit:
            self._managed_domain = (
                self._managed_domain.extended(span)
                if self._managed_domain is not None
                else ManagedDomain(span)
            )

    def find_interval(self, start: int, length: int) -> tuple[int, int] | None:
        """Find the earliest available interval at or after ``start``."""
        validate_coordinate(start, "start")
        validate_length(length)
        node = self._find_interval_optimized(self.root, start, length)
        if node is None:
            return None
        allocation_start = max(start, node.start)
        return allocation_start, allocation_start + length

    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> tuple[int, int] | None:
        """Find best available interval of given length using summary optimization"""
        validate_length(length)
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

    def find_largest_available(self) -> tuple[int, int] | None:
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
            "total_free": summary.total_free_length,
            "total_occupied": summary.total_occupied_length,
            "total_space": total_space,
            "free_chunks": summary.contiguous_count,
            "largest_chunk": summary.largest_free_length,
            "avg_chunk_size": summary.avg_free_length,
            "utilization": summary.total_occupied_length / total_space
            if total_space > 0
            else 0.0,
            "fragmentation": 1.0
            - (summary.largest_free_length / summary.total_free_length)
            if summary.total_free_length > 0
            else 0.0,
            "free_density": summary.free_density,
            "bounds": (summary.earliest_free_start, summary.latest_free_end),
        }

    def get_intervals(self) -> list[tuple[int, int]]:
        """Get all free intervals"""
        intervals: list[tuple[int, int]] = []
        self._collect_intervals(self.root, intervals)
        return intervals

    def _collect_intervals(
        self,
        node: SummaryIntervalNode | None,
        intervals: list[tuple[int, int]],
    ) -> None:
        """Collect all intervals via in-order traversal"""
        if not node:
            return

        self._collect_intervals(node.left, intervals)
        intervals.append((node.start, node.end))
        self._collect_intervals(node.right, intervals)

    def _find_interval_optimized(
        self, node: SummaryIntervalNode | None, start: int, length: int
    ) -> SummaryIntervalNode | None:
        """Find the earliest fit while pruning impossible subtrees."""
        if (
            node is None
            or node.summary.largest_free_length < length
            or node.summary.latest_free_end is None
            or node.summary.latest_free_end <= start
        ):
            return None

        left_result = self._find_interval_optimized(node.left, start, length)
        if left_result is not None:
            return left_result

        allocation_start = max(start, node.start)
        if allocation_start + length <= node.end:
            return node

        return self._find_interval_optimized(node.right, start, length)

    def _find_best_fit_node(
        self, node: SummaryIntervalNode | None, length: int, prefer_early: bool
    ) -> SummaryIntervalNode | None:
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

    def _find_node_with_largest(
        self, node: SummaryIntervalNode | None
    ) -> SummaryIntervalNode | None:
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
    def _insert(
        self, node: SummaryIntervalNode | None, new_node: SummaryIntervalNode
    ) -> SummaryIntervalNode:
        if not node:
            return new_node

        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
        else:
            node.right = self._insert(node.right, new_node)

        node.update_stats()
        return self._rebalance(node)

    def _delete_interval(
        self, node: SummaryIntervalNode | None, start: int, end: int
    ) -> SummaryIntervalNode | None:
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
                nodes_to_insert.append(SummaryIntervalNode(node.start, start))

            # Create right remainder if exists
            if node.end > end:
                nodes_to_insert.append(SummaryIntervalNode(end, node.end))

            # Remove current node and process subtrees
            node = self._merge_subtrees(
                self._delete_interval(node.left, start, end),
                self._delete_interval(node.right, start, end),
            )

            # Insert remainder nodes
            for n in nodes_to_insert:
                node = self._insert(node, n)

        if node:
            node.update_stats()
            node = self._rebalance(node)

        return node

    def _delete_overlaps(
        self,
        node: SummaryIntervalNode | None,
        start: int,
        end: int,
        overlapping_nodes: list[SummaryIntervalNode],
    ) -> SummaryIntervalNode | None:
        """Find and remove overlapping intervals (for release operations)"""
        if not node:
            return None

        if node.end < start:
            node.right = self._delete_overlaps(
                node.right, start, end, overlapping_nodes
            )
        elif node.start > end:
            node.left = self._delete_overlaps(node.left, start, end, overlapping_nodes)
        else:
            # Overlap detected
            overlapping_nodes.append(node)
            node = self._merge_subtrees(
                self._delete_overlaps(node.left, start, end, overlapping_nodes),
                self._delete_overlaps(node.right, start, end, overlapping_nodes),
            )
            return node

        if node:
            node.update_stats()
            node = self._rebalance(node)

        return node

    def _merge_subtrees(
        self, left: SummaryIntervalNode | None, right: SummaryIntervalNode | None
    ) -> SummaryIntervalNode | None:
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

    def _delete_min(self, node: SummaryIntervalNode) -> SummaryIntervalNode | None:
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
            rotated = self._rotate_right(node)
            assert rotated is not None
            node = rotated
        elif balance < -1:
            # Right heavy
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            rotated = self._rotate_left(node)
            assert rotated is not None
            node = rotated

        return node

    def _get_balance(self, node: SummaryIntervalNode | None) -> int:
        """Get balance factor for AVL tree"""
        if not node:
            return 0
        return SummaryIntervalNode.get_height(
            node.left
        ) - SummaryIntervalNode.get_height(node.right)

    def _rotate_left(self, z: SummaryIntervalNode | None) -> SummaryIntervalNode | None:
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

    def _rotate_right(
        self, z: SummaryIntervalNode | None
    ) -> SummaryIntervalNode | None:
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
