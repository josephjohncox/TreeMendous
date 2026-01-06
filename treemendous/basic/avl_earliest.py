from typing import Any, Callable, Optional, List

from treemendous.basic.base import IntervalNodeProtocol
from treemendous.basic.avl import IntervalNode, IntervalTree
from treemendous.basic.protocols import CoreIntervalManagerProtocol, IntervalResult

class EarliestIntervalNode(IntervalNode, IntervalNodeProtocol):
    def __init__(self, start: int, end: int, data: Optional[Any] = None) -> None:
        super().__init__(start, end, data)
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

class EarliestIntervalTree(IntervalTree[EarliestIntervalNode], CoreIntervalManagerProtocol[Any]):
    def __init__(
        self,
        merge_fn: Optional[Callable[[Any, Any], Any]] = None,
        split_fn: Optional[Callable[[Any, int, int, int, int], Any]] = None,
        can_merge: Optional[Callable[[Optional[Any], Optional[Any]], bool]] = None,
        merge_idempotent: bool = False,
        split_idempotent: bool = False,
    ) -> None:
        super().__init__(
            EarliestIntervalNode,
            merge_fn=merge_fn,
            split_fn=split_fn,
            can_merge=can_merge,
            merge_idempotent=merge_idempotent,
            split_idempotent=split_idempotent,
        )

    def _print_node(self, node: EarliestIntervalNode, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}-{node.end} "
              f"(min_start={node.min_start}, max_end={node.max_end}, max_length={node.max_length})")

    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        node = self._find_interval(self.root, start, length)
        if node:
            # Allocate from the requested start point if possible, otherwise from interval start  
            alloc_start = max(start, node.start)
            if node.end - alloc_start >= length:
                return IntervalResult(start=alloc_start, end=alloc_start + length, data=node.data)
        return None

    def get_intervals(self) -> List[IntervalResult]:
        """Get all available intervals as IntervalResult objects"""
        intervals = super().get_intervals()  # Get List[Tuple[int, int, data]]
        return [IntervalResult(start=start, end=end, data=data) for start, end, data in intervals]
    
    def _find_interval(self, node: Optional[EarliestIntervalNode], start: int, 
                      length: int) -> Optional[EarliestIntervalNode]:
        if not node:
            return None
        
        # Check if this interval can satisfy the request
        # Case 1: start is within the interval and there's enough space
        if node.start <= start < node.end and (node.end - start) >= length:
            # This interval works, but check if there's an earlier one
            left_candidate = self._find_interval(node.left, start, length)
            return left_candidate if left_candidate else node
        
        # Case 2: start is before this interval and interval is large enough
        elif start <= node.start and (node.end - node.start) >= length:
            # This interval works, but check if there's an earlier one
            left_candidate = self._find_interval(node.left, start, length)
            return left_candidate if left_candidate else node
        
        # Case 3: start is after this interval's end
        elif start >= node.end:
            return self._find_interval(node.right, start, length)
        
        # Case 4: start is before this interval's start but interval is too small
        elif start < node.start:
            # Check both subtrees
            left_candidate = self._find_interval(node.left, start, length)
            if left_candidate:
                return left_candidate
            return self._find_interval(node.right, start, length)
        
        # Case 5: start is within interval but not enough space remaining
        else:
            # Check right subtree for intervals starting after this one
            return self._find_interval(node.right, start, length)

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
