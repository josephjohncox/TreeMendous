from typing import cast

from treemendous.basic.avl import IntervalNode, IntervalTree
from treemendous.domain import IntervalResult, validate_coordinate, validate_length


class EarliestIntervalNode(IntervalNode):
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


class EarliestIntervalTree(IntervalTree[EarliestIntervalNode]):
    def __init__(self) -> None:
        super().__init__(EarliestIntervalNode)

    def _print_node(self, node: EarliestIntervalNode, indent: str, prefix: str) -> None:
        print(
            f"{indent}{prefix}{node.start}-{node.end} "
            f"(min_start={node.min_start}, max_end={node.max_end}, max_length={node.max_length})"
        )

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
        validate_coordinate(start, "start")
        validate_length(length)
        node = self._find_interval(self.root, start, length)
        if node:
            # Allocate from the requested start point if possible, otherwise from interval start
            alloc_start = max(start, node.start)
            if node.end - alloc_start >= length:
                return IntervalResult(start=alloc_start, end=alloc_start + length)
        return None

    def _find_interval(
        self, node: EarliestIntervalNode | None, start: int, length: int
    ) -> EarliestIntervalNode | None:
        if not node:
            return None

        # Check if this interval can satisfy the request
        # Case 1: start is within the interval and there's enough space
        if node.start <= start < node.end and (node.end - start) >= length:
            # This interval works, but check if there's an earlier one
            left_candidate = self._find_interval(
                cast(EarliestIntervalNode | None, node.left), start, length
            )
            return left_candidate if left_candidate else node

        # Case 2: start is before this interval and interval is large enough
        elif start <= node.start and (node.end - node.start) >= length:
            # This interval works, but check if there's an earlier one
            left_candidate = self._find_interval(
                cast(EarliestIntervalNode | None, node.left), start, length
            )
            return left_candidate if left_candidate else node

        # Case 3: start is after this interval's end
        elif start >= node.end:
            return self._find_interval(
                cast(EarliestIntervalNode | None, node.right), start, length
            )

        # Case 4: start is before this interval's start but interval is too small
        elif start < node.start:
            # Check both subtrees
            left_candidate = self._find_interval(
                cast(EarliestIntervalNode | None, node.left), start, length
            )
            if left_candidate:
                return left_candidate
            return self._find_interval(
                cast(EarliestIntervalNode | None, node.right), start, length
            )

        # Case 5: start is within interval but not enough space remaining
        else:
            # Check right subtree for intervals starting after this one
            return self._find_interval(
                cast(EarliestIntervalNode | None, node.right), start, length
            )

    def _insert(
        self, node: EarliestIntervalNode | None, new_node: EarliestIntervalNode
    ) -> EarliestIntervalNode:
        node = super()._insert(node, new_node)
        node.update_stats()  # Update the earliest-specific stats
        return node


# Example usage:
