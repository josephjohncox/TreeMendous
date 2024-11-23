from typing import Generic, Optional, List, TypeVar, overload
from base import IntervalNodeBase, IntervalTreeBase

R = TypeVar('R', bound='IntervalNode')

class IntervalNode(IntervalNodeBase['IntervalNode']):
    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)
        self.total_length: int = self.length
        self.height: int = 1

    def update_stats(self) -> None:
        self.update_length()
        self.total_length = self.length
        if self.left:
            assert isinstance(self.left, IntervalNode)
            self.total_length += self.left.total_length
        if self.right:
            assert isinstance(self.right, IntervalNode)
            self.total_length += self.right.total_length

        self.height = 1 + max(self.get_height(self.left), self.get_height(self.right))

    @staticmethod
    def get_height(node: Optional['IntervalNode']) -> int:
        if not node:
            return 0
        return node.height

class IntervalTree(Generic[R], IntervalTreeBase[R]):
    def __init__(self, node_class: type[R]) -> None:
        super().__init__()
        self.node_class = node_class
        self.root: Optional[R] = None

    def _print_node(self, node: R, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}-{node.end} (len={node.length}, total_len={node.total_length})")

    def insert_interval(self, start: int, end: int) -> None:
        overlapping_nodes: List[R] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        # Merge overlapping intervals with the new interval
        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)
        # Insert the merged interval using the constructor
        self.root = self._insert(self.root, self.node_class(start, end))

    def delete_interval(self, start: int, end: int) -> None:
        # Find overlapping intervals
        overlapping_nodes: List[R] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        # For each overlapping interval, we may need to split it
        for node in overlapping_nodes:
            if node.start < start:
                # Left part remains available
                self.root = self._insert(self.root, self.node_class(node.start, start))
            if node.end > end:
                # Right part remains available
                self.root = self._insert(self.root, self.node_class(end, node.end))

    def _delete_overlaps(self, node: Optional[R], start: int, end: int, 
                        overlapping_nodes: List[R]) -> Optional[R]:
        if not node:
            return None

        # If current node is completely before or after the interval
        if end <= node.start:
            node.left = self._delete_overlaps(node.left, start, end, overlapping_nodes)
            node.update_stats()  # Important: update stats after modification
            return node
        if start >= node.end:
            node.right = self._delete_overlaps(node.right, start, end, overlapping_nodes)
            node.update_stats()  # Important: update stats after modification
            return node

        # Handle overlap case
        overlapping_nodes.append(node)
        left_tree = self._delete_overlaps(node.left, start, end, overlapping_nodes)
        right_tree = self._delete_overlaps(node.right, start, end, overlapping_nodes)

        # Merge remaining subtrees
        if left_tree and right_tree:
            min_node = self._get_min(right_tree)
            min_node.left = left_tree
            min_node.update_stats()
            return right_tree
        return left_tree or right_tree

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

# Example usage:
if __name__ == "__main__":
    tree = IntervalTree[IntervalNode](IntervalNode)
    # Initially, the whole interval [0, 100] is available
    tree.insert_interval(0, 100)
    print("Initial tree:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [10, 20]
    tree.delete_interval(10, 20)
    print("\nAfter scheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [30, 40]
    tree.delete_interval(30, 40)
    print("\nAfter scheduling [30, 40]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Unschedule interval [10, 20]
    tree.insert_interval(10, 20)
    print("\nAfter unscheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Split at pivot 50 (delete [50, 50])
    tree.delete_interval(50, 50)
    print("\nAfter splitting at pivot 50:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")