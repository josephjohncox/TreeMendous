"""Independent finite point-set oracle for small multidimensional boxes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class OracleEntry:
    handle: int
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    data: Any


def _points(
    lower: tuple[int, ...], upper: tuple[int, ...]
) -> frozenset[tuple[int, ...]]:
    axes = tuple(
        range(lower_bound, upper_bound)
        for lower_bound, upper_bound in zip(lower, upper, strict=True)
    )
    return frozenset(product(*axes))


class BruteBoxIndex:
    """Insertion-ordered identity model using explicit finite lattice points."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.version = 0
        self.next_handle = 1
        self.entries: dict[int, OracleEntry] = {}

    def insert(
        self,
        lower: tuple[int, ...],
        upper: tuple[int, ...],
        data: Any,
    ) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.version += 1
        self.entries[handle] = OracleEntry(handle, lower, upper, data)
        return handle

    def update(
        self,
        handle: int,
        *,
        lower: tuple[int, ...] | None = None,
        upper: tuple[int, ...] | None = None,
        data: Any,
    ) -> None:
        entry = self.entries[handle]
        self.entries[handle] = OracleEntry(
            handle,
            entry.lower if lower is None else lower,
            entry.upper if upper is None else upper,
            data,
        )
        self.version += 1

    def remove(self, handle: int) -> OracleEntry:
        self.version += 1
        return self.entries.pop(handle)

    def overlaps(
        self, lower: tuple[int, ...], upper: tuple[int, ...]
    ) -> tuple[OracleEntry, ...]:
        query_points = _points(lower, upper)
        return tuple(
            entry
            for entry in self.entries.values()
            if _points(entry.lower, entry.upper) & query_points
        )
