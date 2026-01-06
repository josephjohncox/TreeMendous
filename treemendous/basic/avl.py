from typing import Any, Callable, Generic, Optional, List, Tuple, TypeVar, cast, overload
from treemendous.basic.base import IntervalNodeBase, IntervalNodeProtocol, IntervalTreeBase



class IntervalNode(IntervalNodeBase['IntervalNode', Any]):
    def __init__(self, start: int, end: int, data: Optional[Any] = None) -> None:
        super().__init__(start, end, data)
        self.total_length: int = self.length
        self.height: int = 1

    def update_stats(self) -> None:
        self.update_length()
        self.total_length = self.length
        if self.left:
            self.total_length += self.left.total_length
        if self.right:
            self.total_length += self.right.total_length
        self.height = 1 + max(
            self.get_height(self.left), 
            self.get_height(self.right)
        )

    @staticmethod
    def get_height(node: Optional['IntervalNode']) -> int:
        return node.height if node else 0

R = TypeVar('R', bound=IntervalNode)
class IntervalTree(Generic[R], IntervalTreeBase[R, Any]):
    def __init__(
        self,
        node_class: type[R],
        merge_fn: Optional[Callable[[Any, Any], Any]] = None,
        split_fn: Optional[Callable[[Any, int, int, int, int], Any]] = None,
        can_merge: Optional[Callable[[Optional[Any], Optional[Any]], bool]] = None,
        merge_idempotent: bool = False,
        split_idempotent: bool = False,
    ) -> None:
        super().__init__(
            merge_fn=merge_fn,
            split_fn=split_fn,
            can_merge=can_merge,
            merge_idempotent=merge_idempotent,
            split_idempotent=split_idempotent,
        )
        self.node_class = node_class
        self.root: Optional[R] = None

    def _print_node(self, node: R, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}-{node.end} (len={node.length}, total_len={node.total_length})")

    def release_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        overlapping_nodes: List[R] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        merged_data = data
        # Merge overlapping intervals with the new interval
        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)
            merged_data = self.merge_data(merged_data, node.data)
        # Insert the merged interval using the constructor
        self.root = self._insert(self.root, self.node_class(start, end, merged_data))

    def reserve_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        self.root = self._delete_interval(self.root, start, end)

    def _delete_interval(
        self, node: Optional[R], start: int, end: int
    ) -> Optional[R]:
        if not node:
            return None

        if node.end <= start:
            # Interval to delete is after the current node
            node.right = self._delete_interval(node.right, start, end)
        elif node.start >= end:
            # Interval to delete is before the current node
            node.left = self._delete_interval(node.left, start, end)
        else:
            # The current node overlaps with the interval to delete
            # We may need to split the node into up to two intervals

            nodes_to_insert = []

            if node.start < start:
                # Left part remains
                left_data = self.split_data(node.data, node.start, node.end, node.start, start)
                left_node = self.node_class(node.start, start, left_data)
                nodes_to_insert.append(left_node)

            if node.end > end:
                # Right part remains
                right_data = self.split_data(node.data, node.start, node.end, end, node.end)
                right_node = self.node_class(end, node.end, right_data)
                nodes_to_insert.append(right_node)

            # Delete the current node and replace it with left and right parts
            node = self._merge_subtrees(
                self._delete_interval(node.left, start, end),
                self._delete_interval(node.right, start, end)
            )

            # Insert any remaining parts
            for n in nodes_to_insert:
                node = self._insert(node, n)

        if node:
            node.update_stats()
            node = self._rebalance(node)
        return node

    def _delete_overlaps(
        self, node: Optional[R], start: int, end: int, overlapping_nodes: List[R]
    ) -> Optional[R]:
        if not node:
            return None

        if node.end <= start:
            # No overlap, move to the right
            node.right = self._delete_overlaps(node.right, start, end, overlapping_nodes)
        elif node.start >= end:
            # No overlap, move to the left
            node.left = self._delete_overlaps(node.left, start, end, overlapping_nodes)
        else:
            # Overlap detected
            overlapping_nodes.append(node)
            # Remove this node and continue searching in both subtrees
            node = self._merge_subtrees(
                self._delete_overlaps(node.left, start, end, overlapping_nodes),
                self._delete_overlaps(node.right, start, end, overlapping_nodes)
            )
            return node

        if node:
            node.update_stats()
            node = self._rebalance(node)
        return node

    def _merge_subtrees(
        self, left: Optional[R], right: Optional[R]
    ) -> Optional[R]:
        if not left:
            return right
        if not right:
            return left

        # Find the node with the minimum start in the right subtree
        min_node = self._get_min(right)
        right = self._delete_min(right)
        min_node.left = left
        min_node.right = right
        min_node.update_stats()
        return self._rebalance(min_node)

    def _delete_min(self, node: R) -> Optional[R]:
        if node.left is None:
            return node.right
        node.left = self._delete_min(node.left)
        node.update_stats()
        return self._rebalance(node)

    def _insert(self, node: Optional[R], new_node: R) -> R:
        if not node:
            return new_node

        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
        else:
            node.right = self._insert(node.right, new_node)

        node.update_stats()
        node = self._rebalance(node)
        return node

    def _get_min(self, node: IntervalNode) -> IntervalNode:
        current = node
        while current.left:
            current = current.left
        return current

    def _rebalance(self, node: R) -> R:
        balance = self._get_balance(node)
        if balance > 1:
            # Left heavy
            if self._get_balance(node.left) < 0:
                # Left-Right case
                node.left = self._rotate_left(node.left)
            # Left-Left case
            node = self._rotate_right(node)
        elif balance < -1:
            # Right heavy
            if self._get_balance(node.right) > 0:
                # Right-Left case
                node.right = self._rotate_right(node.right)
            # Right-Right case
            node = self._rotate_left(node)
        return node

    def _get_balance(self, node: Optional[IntervalNode]) -> int:
        if not node:
            return 0
        return IntervalNode.get_height(node.left) - IntervalNode.get_height(node.right)

    @overload
    def _rotate_left(self, z: None) -> None: ...

    @overload
    def _rotate_left(self, z: R) -> R: ...

    def _rotate_left(self, z: Optional[R]) -> Optional[R]:
        if not z or not z.right:
            return z
        y: R = z.right
        subtree: Optional[R] = y.left

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

    def _rotate_right(self, z: Optional[R]) -> Optional[R]:
        if not z or not z.left:
            return z
        y: R = z.left
        subtree: Optional[R] = y.right

        # Perform rotation
        y.right = z
        z.left = subtree

        # Update heights and stats
        z.update_stats()
        y.update_stats()
        return y
    
    def get_intervals(self) -> List[Tuple[int, int, Optional[Any]]]:
        intervals: List[Tuple[int, int, Optional[Any]]] = []
        self._get_intervals(self.root, intervals)
        return intervals

    def _get_intervals(self, node: Optional[R], intervals: List[Tuple[int, int, Optional[Any]]]) -> None:
        if not node:
            return
        intervals.append((node.start, node.end, node.data))
        self._get_intervals(node.left, intervals)
        self._get_intervals(node.right, intervals)

# Example usage:
if __name__ == "__main__":
    tree = IntervalTree[IntervalNode](IntervalNode)
    # Initially, the whole interval [0, 100] is available
    tree.release_interval(0, 100)
    print("Initial tree:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [0, 1
    tree.reserve_interval(0, 1)
    print("\nAfter scheduling [0, 1]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Unschedule interval [0, 1]
    tree.release_interval(0, 1)
    print("\nAfter unscheduling [0, 1]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [10, 20]
    tree.reserve_interval(10, 20)
    print("\nAfter scheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [30, 40]
    tree.reserve_interval(30, 40)
    print("\nAfter scheduling [30, 40]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Unschedule interval [10, 20]
    tree.release_interval(10, 20)
    print("\nAfter unscheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Split at pivot 50 (delete [50, 50])
    tree.reserve_interval(50, 50)
    print("\nAfter splitting at pivot 50:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")
