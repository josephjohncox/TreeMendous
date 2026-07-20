"""Guarded sparse-grid candidate index for bounded boxes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import TYPE_CHECKING

from treemendous.multidimensional.domain import Box, BoxHandle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from treemendous.multidimensional.domain import BoxEntry

_Cell = tuple[int, ...]


@dataclass(frozen=True)
class _GridState:
    postings: dict[_Cell, tuple[BoxHandle, ...]]
    handle_cells: dict[BoxHandle, tuple[_Cell, ...]]
    posting_count: int


class SparseGridStrategy:
    """Copy-on-write sparse grid with preflighted resource limits."""

    algorithm = "sparse_grid"

    def __init__(
        self,
        bounds: Box,
        cell_size: tuple[int, ...],
        *,
        max_total_cells: int,
        max_cells_per_entry: int,
        max_cells_per_query: int,
        max_total_postings: int,
    ) -> None:
        self.bounds = bounds
        self.cell_size = cell_size
        self.grid_shape = tuple(
            (upper - lower + size - 1) // size
            for lower, upper, size in zip(
                bounds.lower,
                bounds.upper,
                cell_size,
                strict=True,
            )
        )
        self.total_possible_cells = prod(self.grid_shape)
        self.max_total_cells = max_total_cells
        self.max_cells_per_entry = max_cells_per_entry
        self.max_cells_per_query = max_cells_per_query
        self.max_total_postings = max_total_postings
        if self.total_possible_cells > max_total_cells:
            raise ValueError(
                "grid has "
                f"{self.total_possible_cells} possible cells, exceeding "
                f"max_total_cells={max_total_cells}"
            )
        self._state = _GridState({}, {}, 0)

    def _cell_ranges(self, box: Box) -> tuple[tuple[range, ...], int]:
        ranges = tuple(
            range(
                (lower - bounds_lower) // size,
                (upper - 1 - bounds_lower) // size + 1,
            )
            for lower, upper, bounds_lower, size in zip(
                box.lower,
                box.upper,
                self.bounds.lower,
                self.cell_size,
                strict=True,
            )
        )
        return ranges, prod(axis_range.stop - axis_range.start for axis_range in ranges)

    def _query_cells(self, box: Box) -> tuple[_Cell, ...]:
        ranges, cell_count = self._cell_ranges(box)
        if cell_count > self.max_cells_per_query:
            raise ValueError(
                f"query touches {cell_count} cells, exceeding "
                f"max_cells_per_query={self.max_cells_per_query}"
            )
        return tuple(product(*ranges))

    def _replacement(
        self,
        handle: BoxHandle,
        *,
        old_cells: tuple[_Cell, ...] = (),
        new_cells: tuple[_Cell, ...] = (),
    ) -> _GridState:
        new_posting_count = self._state.posting_count - len(old_cells) + len(new_cells)
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )

        postings = self._state.postings.copy()
        for cell in old_cells:
            remaining = tuple(
                candidate for candidate in postings[cell] if candidate != handle
            )
            if remaining:
                postings[cell] = remaining
            else:
                del postings[cell]
        for cell in new_cells:
            postings[cell] = (*postings.get(cell, ()), handle)

        handle_cells = self._state.handle_cells.copy()
        if new_cells:
            handle_cells[handle] = new_cells
        else:
            handle_cells.pop(handle, None)
        return _GridState(postings, handle_cells, new_posting_count)

    def prepare_insert(self, handle: BoxHandle, box: Box) -> _GridState:
        ranges, cell_count = self._cell_ranges(box)
        if cell_count > self.max_cells_per_entry:
            raise ValueError(
                f"box touches {cell_count} cells, exceeding "
                f"max_cells_per_entry={self.max_cells_per_entry}"
            )
        new_posting_count = self._state.posting_count + cell_count
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )
        cells = tuple(product(*ranges))
        return self._replacement(handle, new_cells=cells)

    def prepare_update(
        self,
        handle: BoxHandle,
        old_box: Box,
        new_box: Box,
    ) -> _GridState:
        if old_box == new_box:
            return self._state
        old_cells = self._state.handle_cells[handle]
        ranges, cell_count = self._cell_ranges(new_box)
        if cell_count > self.max_cells_per_entry:
            raise ValueError(
                f"box touches {cell_count} cells, exceeding "
                f"max_cells_per_entry={self.max_cells_per_entry}"
            )
        new_posting_count = self._state.posting_count - len(old_cells) + cell_count
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )
        new_cells = tuple(product(*ranges))
        return self._replacement(
            handle,
            old_cells=old_cells,
            new_cells=new_cells,
        )

    def prepare_remove(self, handle: BoxHandle, box: Box) -> _GridState:
        del box
        return self._replacement(
            handle,
            old_cells=self._state.handle_cells[handle],
        )

    def commit(self, prepared: _GridState) -> None:
        """Publish an already-built state; this hook performs only assignment."""
        self._state = prepared

    def candidate_handles(
        self,
        query: Box,
        entries: Mapping[BoxHandle, BoxEntry],
    ) -> tuple[BoxHandle, ...]:
        del entries
        candidates: set[BoxHandle] = set()
        for cell in self._query_cells(query):
            candidates.update(self._state.postings.get(cell, ()))
        return tuple(sorted(candidates, key=lambda handle: handle.sequence))

    def diagnostics(self) -> dict[str, object]:
        return {
            "bounds_lower": self.bounds.lower,
            "bounds_upper": self.bounds.upper,
            "cell_size": self.cell_size,
            "grid_shape": self.grid_shape,
            "total_possible_cells": self.total_possible_cells,
            "occupied_cell_count": len(self._state.postings),
            "posting_count": self._state.posting_count,
            "max_total_cells": self.max_total_cells,
            "max_cells_per_entry": self.max_cells_per_entry,
            "max_cells_per_query": self.max_cells_per_query,
            "max_total_postings": self.max_total_postings,
        }
