"""Shared implementation helpers for legacy Python interval trees."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, cast

from treemendous.domain import (
    IntervalResult,
    Span,
    validate_coordinate,
    validate_length,
)

D = TypeVar("D")


class IntervalNodeProtocol(Protocol[D]):
    start: int
    end: int
    length: int
    height: int
    total_length: int
    data: D | None
    left: IntervalNodeProtocol[D] | None
    right: IntervalNodeProtocol[D] | None

    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...


T = TypeVar("T")
D_contra = TypeVar("D_contra", contravariant=True)


class IntervalManagerProtocol(Protocol[D_contra]):
    def reserve_interval(
        self, start: int, end: int, data: D_contra | None = None
    ) -> None: ...
    def release_interval(
        self, start: int, end: int, data: D_contra | None = None
    ) -> None: ...
    def find_interval(self, start: int, length: int) -> IntervalResult | None: ...
    def get_intervals(self) -> list[IntervalResult]: ...


class IntervalNodeBase(Generic[T, D]):
    def __init__(self, start: int, end: int, data: D | None = None) -> None:
        Span(start, end)
        self.start = start
        self.end = end
        self.length = end - start
        self._height = 1
        self._total_length = self.length
        self.data = data
        self.left: T | None = None
        self.right: T | None = None

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = value

    @property
    def total_length(self) -> int:
        return self._total_length

    @total_length.setter
    def total_length(self, value: int) -> None:
        self._total_length = value

    def update_length(self) -> None:
        self.length = self.end - self.start


class IntervalTreeBase(Generic[T, D], ABC):
    def __init__(
        self,
        root: T | None = None,
        merge_fn: Callable[[D, D], D] | None = None,
        split_fn: Callable[[D, int, int, int, int], D] | None = None,
        can_merge: Callable[[D | None, D | None], bool] | None = None,
        merge_idempotent: bool = False,
        split_idempotent: bool = False,
    ) -> None:
        self.root = root
        self.merge_fn = merge_fn
        self.split_fn = split_fn
        self.can_merge_fn = can_merge
        self.merge_idempotent = merge_idempotent
        # Kept only as a constructor compatibility field. Restriction callbacks
        # are always invoked; idempotence is not a valid reason to skip them.
        self.split_idempotent = split_idempotent

    @staticmethod
    def validate_span(start: int, end: int) -> Span:
        return Span(start, end)

    @staticmethod
    def validate_query(start: int, length: int) -> tuple[int, int]:
        validate_coordinate(start, "start")
        validate_length(length)
        return start, length

    def print_tree(self) -> None:
        self._print_tree(self.root)

    def _print_tree(self, node: T | None, indent: str = "", prefix: str = "") -> None:
        if node is None:
            return
        view = cast(IntervalNodeProtocol[Any], node)
        self._print_tree(cast(T | None, view.right), indent + "    ", "┌── ")
        self._print_node(node, indent, prefix)
        if view.data is not None:
            print(f"{indent}{prefix}data: {view.data}")
        self._print_tree(cast(T | None, view.left), indent + "    ", "└── ")

    def get_total_available_length(self) -> int:
        if self.root is None:
            return 0
        return cast(IntervalNodeProtocol[Any], self.root).total_length

    def merge_data(self, data1: D | None, data2: D | None) -> D | None:
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if self.merge_idempotent and (data1 is data2 or data1 == data2):
            return data1
        if self.merge_fn is not None:
            return self.merge_fn(data1, data2)
        if isinstance(data1, set) and isinstance(data2, set):
            return data1 | data2  # type: ignore[return-value]
        return data1

    def split_data(
        self,
        data: D | None,
        old_start: int,
        old_end: int,
        new_start: int,
        new_end: int,
    ) -> D | None:
        if data is None or self.split_fn is None:
            return data
        return self.split_fn(data, old_start, old_end, new_start, new_end)

    def can_merge_data(self, data1: D | None, data2: D | None) -> bool:
        return self.can_merge_fn(data1, data2) if self.can_merge_fn else True

    @abstractmethod
    def _print_node(self, node: T, indent: str, prefix: str) -> None: ...

    @abstractmethod
    def get_intervals(self) -> list[Any]: ...
