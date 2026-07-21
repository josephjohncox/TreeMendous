from typing import Generic, Optional, TypeVar, cast, overload

from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase
from treemendous.domain import IntervalResult, Span


class IntervalNode(IntervalNodeBase["IntervalNode"]):
    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)
        self.total_length: int = self.length
        self.height: int = 1

    def update_stats(self) -> None:
        self.update_length()
        self.total_length = self.length
        if self.left:
            self.total_length += self.left.total_length
        if self.right:
            self.total_length += self.right.total_length
        self.height = 1 + max(self.get_height(self.left), self.get_height(self.right))

    @staticmethod
    def get_height(node: Optional["IntervalNode"]) -> int:
        return node.height if node else 0


R = TypeVar("R", bound=IntervalNode)


class IntervalTree(Generic[R], IntervalTreeBase[R]):
    def __init__(self, node_class: type[R]) -> None:
        super().__init__()
        self.node_class = node_class
        self.root: R | None = None

    @overload
    def _typed_child(self, node: None) -> None: ...

    @overload
    def _typed_child(self, node: IntervalNode) -> R: ...

    def _typed_child(self, node: IntervalNode | None) -> R | None:
        """Recover the homogeneous node subtype guaranteed by ``node_class``."""
        return cast(R | None, node)

    def _print_node(self, node: R, indent: str, prefix: str) -> None:
        print(
            f"{indent}{prefix}{node.start}-{node.end} (len={node.length}, total_len={node.total_length})"
        )

    def release_interval(self, start: int, end: int) -> None:
        Span(start, end)
        overlapping_nodes: list[R] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)
        self.root = self._insert(self.root, self.node_class(start, end))

    def reserve_interval(self, start: int, end: int) -> None:
        Span(start, end)
        self.root = self._delete_interval(self.root, start, end)

    def _delete_interval(self, node: R | None, start: int, end: int) -> R | None:
        if not node:
            return None

        if node.end <= start:
            # Interval to delete is after the current node
            node.right = self._delete_interval(
                self._typed_child(node.right), start, end
            )
        elif node.start >= end:
            # Interval to delete is before the current node
            node.left = self._delete_interval(self._typed_child(node.left), start, end)
        else:
            # The current node overlaps with the interval to delete
            # We may need to split the node into up to two intervals

            nodes_to_insert = []

            if node.start < start:
                nodes_to_insert.append(self.node_class(node.start, start))

            if node.end > end:
                nodes_to_insert.append(self.node_class(end, node.end))

            # Delete the current node and replace it with left and right parts
            node = self._merge_subtrees(
                self._delete_interval(self._typed_child(node.left), start, end),
                self._delete_interval(self._typed_child(node.right), start, end),
            )

            # Insert any remaining parts
            for n in nodes_to_insert:
                node = self._insert(node, n)

        if node:
            node.update_stats()
            node = self._rebalance(node)
        return node

    def _delete_overlaps(
        self, node: R | None, start: int, end: int, overlapping_nodes: list[R]
    ) -> R | None:
        if not node:
            return None

        if node.end <= start:
            # No overlap, move to the right
            node.right = self._delete_overlaps(
                self._typed_child(node.right), start, end, overlapping_nodes
            )
        elif node.start >= end:
            # No overlap, move to the left
            node.left = self._delete_overlaps(
                self._typed_child(node.left), start, end, overlapping_nodes
            )
        else:
            # Overlap detected
            overlapping_nodes.append(node)
            # Remove this node and continue searching in both subtrees
            node = self._merge_subtrees(
                self._delete_overlaps(
                    self._typed_child(node.left), start, end, overlapping_nodes
                ),
                self._delete_overlaps(
                    self._typed_child(node.right), start, end, overlapping_nodes
                ),
            )
            return node

        if node:
            node.update_stats()
            node = self._rebalance(node)
        return node

    def _merge_subtrees(self, left: R | None, right: R | None) -> R | None:
        """Join two ordered AVL subtrees without assuming similar heights."""
        if not left:
            return right
        if not right:
            return left

        left_height = IntervalNode.get_height(left)
        right_height = IntervalNode.get_height(right)
        if left_height > right_height + 1:
            left.right = self._merge_subtrees(self._typed_child(left.right), right)
            left.update_stats()
            return self._rebalance(left)
        if right_height > left_height + 1:
            right.left = self._merge_subtrees(left, self._typed_child(right.left))
            right.update_stats()
            return self._rebalance(right)

        # Similar-height trees can be joined through the minimum right node.
        min_node = self._get_min(right)
        right = self._delete_min(right)
        min_node.left = left
        min_node.right = right
        min_node.update_stats()
        return self._rebalance(min_node)

    def _delete_min(self, node: R) -> R | None:
        if node.left is None:
            return self._typed_child(node.right)
        node.left = self._delete_min(self._typed_child(node.left))
        node.update_stats()
        return self._rebalance(node)

    def _insert(self, node: R | None, new_node: R) -> R:
        if not node:
            return new_node

        if new_node.start < node.start:
            node.left = self._insert(self._typed_child(node.left), new_node)
        else:
            node.right = self._insert(self._typed_child(node.right), new_node)

        node.update_stats()
        node = self._rebalance(node)
        return node

    def _get_min(self, node: R) -> R:
        current = node
        while current.left:
            current = self._typed_child(current.left)
        return current

    def _rebalance(self, node: R) -> R:
        """Repair balance even when a bulk deletion changes height by more than one."""
        balance = self._get_balance(node)
        if balance > 1:
            if self._get_balance(node.left) < 0:
                node.left = self._rotate_left(self._typed_child(node.left))
            node = self._rotate_right(node)
            right = self._typed_child(node.right)
            if right is not None and abs(self._get_balance(right)) > 1:
                node.right = self._rebalance(right)
            node.update_stats()
        elif balance < -1:
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(self._typed_child(node.right))
            node = self._rotate_left(node)
            left = self._typed_child(node.left)
            if left is not None and abs(self._get_balance(left)) > 1:
                node.left = self._rebalance(left)
            node.update_stats()
        return self._rebalance(node) if abs(self._get_balance(node)) > 1 else node

    def _get_balance(self, node: IntervalNode | None) -> int:
        if not node:
            return 0
        return IntervalNode.get_height(node.left) - IntervalNode.get_height(node.right)

    @overload
    def _rotate_left(self, z: None) -> None: ...

    @overload
    def _rotate_left(self, z: R) -> R: ...

    def _rotate_left(self, z: R | None) -> R | None:
        if not z or not z.right:
            return z
        y = self._typed_child(z.right)
        subtree = self._typed_child(y.left)

        # Perform rotation
        y.left = z
        z.right = subtree

        # Update heights and stats
        z.update_stats()
        y.update_stats()
        return y

    @overload
    def _rotate_right(self, z: None) -> None: ...

    @overload
    def _rotate_right(self, z: R) -> R: ...

    def _rotate_right(self, z: R | None) -> R | None:
        if not z or not z.left:
            return z
        y = self._typed_child(z.left)
        subtree = self._typed_child(y.right)

        # Perform rotation
        y.right = z
        z.left = subtree

        # Update heights and stats
        z.update_stats()
        y.update_stats()
        return y

    def get_intervals(self) -> list[IntervalResult]:
        intervals: list[IntervalResult] = []
        self._get_intervals(self.root, intervals)
        return intervals

    def _get_intervals(self, node: R | None, intervals: list[IntervalResult]) -> None:
        if not node:
            return
        self._get_intervals(self._typed_child(node.left), intervals)
        intervals.append(IntervalResult(node.start, node.end))
        self._get_intervals(self._typed_child(node.right), intervals)
