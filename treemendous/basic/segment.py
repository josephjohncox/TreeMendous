from typing import Optional
from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase


class SegmentTreeNode(IntervalNodeBase['SegmentTreeNode', None]):
    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)
        self.total_length: int = self.length
        self.is_full: bool = True

    def update_node(self) -> None:
        if self.left is not None and self.right is not None:
            self.total_length = self.left.total_length + self.right.total_length
            self.is_full = self.left.is_full and self.right.is_full
        else:
            self.total_length = self.length if self.is_full else 0

class SegmentTree(IntervalTreeBase[SegmentTreeNode, None]):
    def __init__(self, start: int, end: int) -> None:
        super().__init__(SegmentTreeNode(start, end))

    def _print_node(self, node: SegmentTreeNode, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}-{node.end} (len={node.length}, total_len={node.total_length}, is_full={node.is_full})")

    def _build(self, node: SegmentTreeNode | None) -> None:
        if node is None:
            return
        if node.end - node.start <= 1:
            return
        mid: int = (node.start + node.end) // 2
        node.left = SegmentTreeNode(node.start, mid)
        node.right = SegmentTreeNode(mid, node.end)
        self._build(node.left)
        self._build(node.right)

    def build(self) -> None:
        self._build(self.root)

    def _update(self, node: Optional[SegmentTreeNode], start: int, end: int, is_full: bool) -> None:
        if node is None:
            return
        if node.end <= start or node.start >= end:
            return
        if node.start >= start and node.end <= end:
            node.is_full = is_full
            node.total_length = node.length if is_full else 0
            node.left = None
            node.right = None
        else:
            if node.left is None or node.right is None:
                mid: int = (node.start + node.end) // 2
                node.left = SegmentTreeNode(node.start, mid)
                node.right = SegmentTreeNode(mid, node.end)
                node.left.is_full = node.right.is_full = node.is_full
                node.left.total_length = (node.left.end - node.left.start) if node.left.is_full else 0
                node.right.total_length = (node.right.end - node.right.start) if node.right.is_full else 0
            self._update(node.left, start, end, is_full)
            self._update(node.right, start, end, is_full)
            node.update_node()

    def schedule_interval(self, start: int, end: int) -> None:
        self._update(self.root, start, end, False)

    def unschedule_interval(self, start: int, end: int) -> None:
        self._update(self.root, start, end, True)
    


# Example usage:
if __name__ == "__main__":
    # Initialize segment tree with interval [0, 100)
    tree = SegmentTree(0, 100)
    tree.build()
    print("Initial tree:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [10, 20)
    tree.schedule_interval(10, 20)
    print("\nAfter scheduling [10, 20):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [30, 40)
    tree.schedule_interval(30, 40)
    print("\nAfter scheduling [30, 40):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Unschedule interval [10, 20)
    tree.unschedule_interval(10, 20)
    print("\nAfter unscheduling [10, 20):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Split at pivot 50 (schedule [50, 50))
    tree.schedule_interval(50, 50)
    print("\nAfter splitting at pivot 50:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")