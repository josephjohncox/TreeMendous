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

# Conservative accounting units for guardrails, not CPython heap measurements.
# They intentionally include container and reference overhead so checks happen
# before potentially combinatorial tuples, dictionaries, or sets are built.
_STATE_BYTES = 512
_CELL_BYTES = 192
_POSTING_BYTES = 64
_HANDLE_BYTES = 128


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
        max_estimated_bytes: int,
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
        self.max_estimated_bytes = max_estimated_bytes
        if self.total_possible_cells > max_total_cells:
            raise ValueError(
                "grid has "
                f"{self.total_possible_cells} possible cells, exceeding "
                f"max_total_cells={max_total_cells}"
            )
        self._check_memory(_STATE_BYTES, "empty grid")
        self.initial_state = _GridState({}, {}, 0)

    def _check_memory(self, estimate: int, operation: str) -> None:
        if estimate > self.max_estimated_bytes:
            raise ValueError(
                f"{operation} requires an estimated {estimate} bytes, exceeding "
                f"max_estimated_bytes={self.max_estimated_bytes}"
            )

    @staticmethod
    def _retained_bytes(
        posting_count: int, occupied_cells: int, handle_count: int
    ) -> int:
        return (
            _STATE_BYTES
            + occupied_cells * _CELL_BYTES
            + posting_count * _POSTING_BYTES
            + handle_count * _HANDLE_BYTES
        )

    def _preflight_replacement_memory(
        self,
        state: _GridState,
        *,
        posting_count: int,
        added_cell_count: int,
        handle_count: int,
    ) -> None:
        current = self._retained_bytes(
            state.posting_count,
            len(state.postings),
            len(state.handle_cells),
        )
        prospective_occupied = min(
            self.total_possible_cells,
            len(state.postings) + added_cell_count,
        )
        prospective = self._retained_bytes(
            posting_count, prospective_occupied, handle_count
        )
        self._check_memory(current + prospective, "copy-on-write mutation")

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

    def _query_cells(self, state: _GridState, box: Box) -> tuple[_Cell, ...]:
        ranges, cell_count = self._cell_ranges(box)
        if cell_count > self.max_cells_per_query:
            raise ValueError(
                f"query touches {cell_count} cells, exceeding "
                f"max_cells_per_query={self.max_cells_per_query}"
            )
        query_estimate = (
            _STATE_BYTES
            + cell_count * _CELL_BYTES
            + len(state.handle_cells) * _HANDLE_BYTES * 2
        )
        self._check_memory(query_estimate, "query")
        return tuple(product(*ranges))

    def _replacement(
        self,
        state: _GridState,
        handle: BoxHandle,
        *,
        old_cells: tuple[_Cell, ...] = (),
        new_cells: tuple[_Cell, ...] = (),
    ) -> _GridState:
        new_posting_count = state.posting_count - len(old_cells) + len(new_cells)
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )

        postings = state.postings.copy()
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

        handle_cells = state.handle_cells.copy()
        if new_cells:
            handle_cells[handle] = new_cells
        else:
            handle_cells.pop(handle, None)
        return _GridState(postings, handle_cells, new_posting_count)

    def prepare_insert(
        self, state: _GridState, handle: BoxHandle, box: Box
    ) -> _GridState:
        ranges, cell_count = self._cell_ranges(box)
        if cell_count > self.max_cells_per_entry:
            raise ValueError(
                f"box touches {cell_count} cells, exceeding "
                f"max_cells_per_entry={self.max_cells_per_entry}"
            )
        new_posting_count = state.posting_count + cell_count
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )
        self._preflight_replacement_memory(
            state,
            posting_count=new_posting_count,
            added_cell_count=cell_count,
            handle_count=len(state.handle_cells) + 1,
        )
        cells = tuple(product(*ranges))
        return self._replacement(state, handle, new_cells=cells)

    def prepare_update(
        self,
        state: _GridState,
        handle: BoxHandle,
        old_box: Box,
        new_box: Box,
    ) -> _GridState:
        if old_box == new_box:
            return state
        old_cells = state.handle_cells[handle]
        ranges, cell_count = self._cell_ranges(new_box)
        if cell_count > self.max_cells_per_entry:
            raise ValueError(
                f"box touches {cell_count} cells, exceeding "
                f"max_cells_per_entry={self.max_cells_per_entry}"
            )
        new_posting_count = state.posting_count - len(old_cells) + cell_count
        if new_posting_count > self.max_total_postings:
            raise ValueError(
                f"operation would create {new_posting_count} total postings, "
                f"exceeding max_total_postings={self.max_total_postings}"
            )
        self._preflight_replacement_memory(
            state,
            posting_count=new_posting_count,
            added_cell_count=cell_count,
            handle_count=len(state.handle_cells),
        )
        new_cells = tuple(product(*ranges))
        return self._replacement(
            state,
            handle,
            old_cells=old_cells,
            new_cells=new_cells,
        )

    def prepare_remove(
        self, state: _GridState, handle: BoxHandle, box: Box
    ) -> _GridState:
        del box
        old_cells = state.handle_cells[handle]
        new_posting_count = state.posting_count - len(old_cells)
        self._preflight_replacement_memory(
            state,
            posting_count=new_posting_count,
            added_cell_count=0,
            handle_count=len(state.handle_cells) - 1,
        )
        return self._replacement(
            state,
            handle,
            old_cells=old_cells,
        )

    def candidate_handles(
        self,
        state: _GridState,
        query: Box,
        entries: Mapping[BoxHandle, BoxEntry],
    ) -> tuple[BoxHandle, ...]:
        del entries
        cells = self._query_cells(state, query)
        candidates: set[BoxHandle] = set()
        for cell in cells:
            candidates.update(state.postings.get(cell, ()))
        return tuple(sorted(candidates, key=lambda handle: handle.sequence))

    def diagnostics(self, state: _GridState) -> dict[str, object]:
        return {
            "bounds_lower": self.bounds.lower,
            "bounds_upper": self.bounds.upper,
            "cell_size": self.cell_size,
            "grid_shape": self.grid_shape,
            "total_possible_cells": self.total_possible_cells,
            "occupied_cell_count": len(state.postings),
            "posting_count": state.posting_count,
            "max_total_cells": self.max_total_cells,
            "max_cells_per_entry": self.max_cells_per_entry,
            "max_cells_per_query": self.max_cells_per_query,
            "max_total_postings": self.max_total_postings,
            "estimated_memory_bytes": self._retained_bytes(
                state.posting_count,
                len(state.postings),
                len(state.handle_cells),
            ),
            "max_estimated_bytes": self.max_estimated_bytes,
        }
