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
        if node is None or node.max_length < length or node.max_end <= start:
            return None

        left_candidate = self._find_interval(
            cast(EarliestIntervalNode | None, node.left), start, length
        )
        if left_candidate is not None:
            return left_candidate

        allocation_start = max(start, node.start)
        if allocation_start + length <= node.end:
            return node

        return self._find_interval(
            cast(EarliestIntervalNode | None, node.right), start, length
        )

    def _insert(
        self, node: EarliestIntervalNode | None, new_node: EarliestIntervalNode
    ) -> EarliestIntervalNode:
        return super()._insert(node, new_node)


# Example usage:
