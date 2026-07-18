"""Shared implementation helpers for internal Python backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, cast

from treemendous.domain import (
    Span,
    validate_coordinate,
    validate_length,
)


class IntervalNodeProtocol(Protocol):
    start: int
    end: int
    length: int
    height: int
    total_length: int
    left: IntervalNodeProtocol | None
    right: IntervalNodeProtocol | None

    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...


T = TypeVar("T")


class IntervalNodeBase(Generic[T]):
    def __init__(self, start: int, end: int) -> None:
        Span(start, end)
        self.start = start
        self.end = end
        self.length = end - start
        self._height = 1
        self._total_length = self.length
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


class IntervalTreeBase(Generic[T], ABC):
    def __init__(self, root: T | None = None) -> None:
        self.root = root

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
        view = cast(IntervalNodeProtocol, node)
        self._print_tree(cast(T | None, view.right), indent + "    ", "┌── ")
        self._print_node(node, indent, prefix)
        self._print_tree(cast(T | None, view.left), indent + "    ", "└── ")

    def get_total_available_length(self) -> int:
        if self.root is None:
            return 0
        return cast(IntervalNodeProtocol, self.root).total_length

    @abstractmethod
    def _print_node(self, node: T, indent: str, prefix: str) -> None: ...

    @abstractmethod
    def get_intervals(self) -> list[Any]: ...
