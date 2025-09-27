from typing import Optional

from treemendous.basic.base import IntervalManagerProtocol, IntervalNodeProtocol
from treemendous.basic.avl import IntervalNode, IntervalTree

class EarliestIntervalNode(IntervalNode, IntervalNodeProtocol):
    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)
        self.min_start: int = start
        self.max_end: int = end
        self.max_length: int = end - start

    def update_stats(self) -> None:
        super().update_stats()
        
        self.min_start = self.start
        self.max_end = self.end
        self.max_length = self.end - self.start

        if self.left:
            assert isinstance(self.left, EarliestIntervalNode)
            self.min_start = min(self.min_start, self.left.min_start)
            self.max_end = max(self.max_end, self.left.max_end)
            self.max_length = max(self.max_length, self.left.max_length)
        if self.right:
            assert isinstance(self.right, EarliestIntervalNode)
            self.min_start = min(self.min_start, self.right.min_start)
            self.max_end = max(self.max_end, self.right.max_end)
            self.max_length = max(self.max_length, self.right.max_length)

class EarliestIntervalTree(IntervalTree[EarliestIntervalNode], IntervalManagerProtocol):
    def __init__(self) -> None:
        super().__init__(EarliestIntervalNode)

    def _print_node(self, node: EarliestIntervalNode, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}-{node.end} "
              f"(min_start={node.min_start}, max_end={node.max_end}, max_length={node.max_length})")

    def find_interval(self, point: int, length: int) -> Optional[EarliestIntervalNode]:
        return self._find_interval(self.root, point, length)

    def _find_interval(self, node: Optional[EarliestIntervalNode], point: int, 
                      length: int) -> Optional[EarliestIntervalNode]:
        if not node:
            return None
        if node.start >= point and (node.end - node.start) >= length:
            # Potential candidate
            left_candidate = self._find_interval(node.left, point, length)
            return left_candidate if left_candidate else node
        elif node.start < point:
            # Search right subtree
            return self._find_interval(node.right, point, length)
        else:
            # Node's interval is too short; search right subtree
            return self._find_interval(node.right, point, length)

    def _insert(self, node: Optional[EarliestIntervalNode], 
                new_node: EarliestIntervalNode) -> EarliestIntervalNode:
        node = super()._insert(node, new_node)
        node.update_stats()  # Update the earliest-specific stats
        return node


# Example usage:
if __name__ == "__main__":
    tree = EarliestIntervalTree()
    # Initially, the whole interval [0, 100) is available
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

    # Schedule interval [1, 2]
    tree.reserve_interval(1, 3)
    print("\nAfter scheduling [1, 3]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [2, 3]
    tree.reserve_interval(2, 5)
    print("\nAfter scheduling [2, 5]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")


    # Schedule interval [10, 20)
    tree.reserve_interval(10, 20)
    print("\nAfter scheduling [10, 20):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [15, 25)
    tree.reserve_interval(15, 25)
    print("\nAfter scheduling [15, 25):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Find interval starting at or after 18 with length at least 5
    result = tree.find_interval(18, 5)
    if result:
        print(f"\nFound interval: [{result.start}, {result.end})")
    else:
        print("\nNo suitable interval found.")

    # Unschedule interval [10, 20)
    tree.release_interval(10, 20)
    print("\nAfter unscheduling [10, 20):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Delete interval overlapping multiple intervals
    tree.reserve_interval(5, 15)
    print("\nAfter deleting interval [5, 15):")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")