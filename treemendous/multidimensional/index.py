"""Thread-safe experimental identity-preserving multidimensional indexes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock, local
from typing import Any, Literal, cast
from uuid import uuid4

from treemendous.domain import validate_coordinate
from treemendous.multidimensional.algorithms.grid import SparseGridStrategy
from treemendous.multidimensional.algorithms.projection import AxisProjectionStrategy
from treemendous.multidimensional.diagnostics import (
    BoxIndexDiagnostics,
    linear_diagnostics,
    strategy_diagnostics,
)
from treemendous.multidimensional.domain import (
    Box,
    BoxEntry,
    BoxHandle,
    BoxIndexSnapshot,
    _detached_entry,
)

_MISSING = object()


class _LinearStrategy:
    algorithm = "linear"
    initial_state = None

    def prepare_insert(self, state: None, handle: BoxHandle, box: Box) -> None:
        del state, handle, box

    def prepare_update(
        self,
        state: None,
        handle: BoxHandle,
        old_box: Box,
        new_box: Box,
    ) -> None:
        del state, handle, old_box, new_box

    def prepare_remove(self, state: None, handle: BoxHandle, box: Box) -> None:
        del state, handle, box

    def candidate_handles(
        self,
        state: None,
        query: Box,
        entries: Mapping[BoxHandle, BoxEntry],
    ) -> tuple[BoxHandle, ...]:
        del state, query
        return tuple(entries)

    def diagnostics(self, state: None) -> dict[str, object]:
        del state
        return {}


@dataclass(frozen=True)
class _PublishedState:
    """Authoritative entries and acceleration state published atomically."""

    entries: dict[BoxHandle, BoxEntry]
    strategy_state: Any
    next_sequence: int
    version: int


class _BaseBoxIndex:
    """Shared atomic identity store around a prepared-mutation strategy."""

    def __init__(
        self,
        dimensions: int,
        strategy: Any,
        *,
        payload_cloner: Callable[[Any], Any] = deepcopy,
    ) -> None:
        self._dimensions = dimensions
        self._strategy = strategy
        self._owner = uuid4()
        self._state = _PublishedState({}, strategy.initial_state, 1, 0)
        self._lock = RLock()
        self._payload_context = local()
        if not callable(payload_cloner):
            raise TypeError("payload_cloner must be callable")
        self._payload_cloner = payload_cloner

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __len__(self) -> int:
        with self._lock:
            return len(self._state.entries)

    @contextmanager
    def _payload_copying(self) -> Iterator[None]:
        depth = getattr(self._payload_context, "depth", 0)
        self._payload_context.depth = depth + 1
        try:
            yield
        finally:
            self._payload_context.depth = depth

    def _payload_is_active(self) -> bool:
        return getattr(self._payload_context, "depth", 0) > 0

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        while True:
            if self._payload_is_active():
                raise RuntimeError(
                    "BoxIndex mutation is not allowed during payload copying"
                )
            if self._lock.acquire(timeout=0.01):
                break
        try:
            if self._payload_is_active():
                raise RuntimeError(
                    "BoxIndex mutation is not allowed during payload copying"
                )
            yield
        finally:
            self._lock.release()

    def _clone_payload(self, data: Any) -> Any:
        return self._payload_cloner(data)

    def _clone_snapshot_payload(self, data: Any) -> Any:
        """Clone snapshot output under the live index's reentrancy guard."""
        with self._payload_copying():
            return self._clone_payload(data)

    def _detach(self, entry: BoxEntry) -> BoxEntry:
        return _detached_entry(entry, self._clone_payload)

    def _validate_box(self, box: Box) -> None:
        if not isinstance(box, Box):
            raise TypeError("box must be a Box")
        if box.dimensions != self._dimensions:
            raise ValueError("box dimensions must match the index")

    def _entry_for(self, handle: BoxHandle) -> BoxEntry:
        if not isinstance(handle, BoxHandle) or handle._owner != self._owner:
            raise KeyError(handle)
        try:
            return self._state.entries[handle]
        except KeyError:
            raise KeyError(handle) from None

    def insert(self, box: Box, data: Any = None) -> BoxHandle:
        self._validate_box(box)
        with self._mutation():
            state = self._state
            handle = BoxHandle(state.next_sequence, self._owner)
            prepared = self._strategy.prepare_insert(state.strategy_state, handle, box)
            with self._payload_copying():
                owned_data = self._clone_payload(data)
            replacement_entries = state.entries.copy()
            replacement_entries[handle] = BoxEntry(handle, box, owned_data)
            self._state = _PublishedState(
                replacement_entries,
                prepared,
                state.next_sequence + 1,
                state.version + 1,
            )
            return handle

    def get(self, handle: BoxHandle) -> BoxEntry:
        with self._lock:
            entry = self._entry_for(handle)
            with self._payload_copying():
                return self._detach(entry)

    def update(
        self,
        handle: BoxHandle,
        *,
        box: Box | None = None,
        data: Any = _MISSING,
    ) -> BoxEntry:
        with self._mutation():
            state = self._state
            entry = self._entry_for(handle)
            if box is None and data is _MISSING:
                raise ValueError("update requires a box or data replacement")
            if box is not None:
                self._validate_box(box)
            replacement_box = entry.box if box is None else box
            prepared = self._strategy.prepare_update(
                state.strategy_state,
                handle,
                entry.box,
                replacement_box,
            )
            with self._payload_copying():
                replacement_data = (
                    self._clone_payload(entry.data)
                    if data is _MISSING
                    else self._clone_payload(data)
                )
                candidate = BoxEntry(handle, replacement_box, replacement_data)
                detached = self._detach(candidate)
            replacement_entries = state.entries.copy()
            replacement_entries[handle] = candidate
            self._state = _PublishedState(
                replacement_entries,
                prepared,
                state.next_sequence,
                state.version + 1,
            )
            return detached

    def remove(self, handle: BoxHandle) -> BoxEntry:
        with self._mutation():
            state = self._state
            entry = self._entry_for(handle)
            prepared = self._strategy.prepare_remove(
                state.strategy_state, handle, entry.box
            )
            with self._payload_copying():
                detached = self._detach(entry)
            replacement_entries = state.entries.copy()
            del replacement_entries[handle]
            self._state = _PublishedState(
                replacement_entries,
                prepared,
                state.next_sequence,
                state.version + 1,
            )
            return detached

    def entries(self) -> tuple[BoxEntry, ...]:
        with self._lock:
            with self._payload_copying():
                return tuple(
                    self._detach(entry) for entry in self._state.entries.values()
                )

    def overlaps(self, box: Box) -> tuple[BoxEntry, ...]:
        self._validate_box(box)
        with self._lock:
            state = self._state
            handles = self._strategy.candidate_handles(
                state.strategy_state, box, state.entries
            )
            matches = [
                state.entries[handle]
                for handle in handles
                if state.entries[handle].box.overlaps(box)
            ]
            with self._payload_copying():
                return tuple(self._detach(entry) for entry in matches)

    def snapshot(self) -> BoxIndexSnapshot:
        with self._lock:
            state = self._state
            with self._payload_copying():
                entries = tuple(self._detach(entry) for entry in state.entries.values())
            return BoxIndexSnapshot(
                self._dimensions,
                state.version,
                entries,
                self._clone_snapshot_payload,
            )

    def diagnostics(self) -> BoxIndexDiagnostics:
        with self._lock:
            state = self._state
            entries = tuple(state.entries.values())
            if self._strategy.algorithm == "linear":
                return linear_diagnostics(self._dimensions, state.version, entries)
            algorithm = cast(
                Literal["axis_projection", "sparse_grid"],
                self._strategy.algorithm,
            )
            return strategy_diagnostics(
                algorithm,
                self._dimensions,
                state.version,
                entries,
                self._strategy.diagnostics(state.strategy_state),
            )


class BoxIndex(_BaseBoxIndex):
    """Experimental O(n) box index preserving duplicate record identity."""

    def __init__(
        self,
        dimensions: int,
        *,
        payload_cloner: Callable[[Any], Any] = deepcopy,
    ) -> None:
        validate_coordinate(dimensions, "dimensions")
        if dimensions < 2:
            raise ValueError("BoxIndex requires at least two dimensions")
        super().__init__(
            dimensions,
            _LinearStrategy(),
            payload_cloner=payload_cloner,
        )


class _FixedProjectionBoxIndex(_BaseBoxIndex):
    """Shared fixed-dimensional lower-bound projection implementation."""

    _fixed_dimensions: int

    def __init__(
        self,
        *,
        payload_cloner: Callable[[Any], Any] = deepcopy,
    ) -> None:
        super().__init__(
            self._fixed_dimensions,
            AxisProjectionStrategy(self._fixed_dimensions),
            payload_cloner=payload_cloner,
        )


class BoxIndex2D(_FixedProjectionBoxIndex):
    """Experimental two-dimensional dynamic axis-projection box index."""

    _fixed_dimensions = 2


class BoxIndex3D(_FixedProjectionBoxIndex):
    """Experimental three-dimensional dynamic axis-projection box index."""

    _fixed_dimensions = 3


class BoxIndex4D(_FixedProjectionBoxIndex):
    """Experimental four-dimensional dynamic axis-projection box index."""

    _fixed_dimensions = 4


class BoundedBoxIndex(_BaseBoxIndex):
    """Experimental guarded sparse-grid index for boxes inside fixed bounds."""

    def __init__(
        self,
        bounds: Box,
        cell_size: tuple[int, ...],
        *,
        max_total_cells: int = 1_000_000,
        max_cells_per_entry: int = 100_000,
        max_cells_per_query: int = 100_000,
        max_total_postings: int = 1_000_000,
        max_estimated_bytes: int = 256 * 1024 * 1024,
        payload_cloner: Callable[[Any], Any] = deepcopy,
    ) -> None:
        if not isinstance(bounds, Box):
            raise TypeError("bounds must be a Box")
        if not 2 <= bounds.dimensions <= 8:
            raise ValueError("BoundedBoxIndex requires between 2 and 8 dimensions")
        if not isinstance(cell_size, tuple):
            raise TypeError("cell_size must be a tuple")
        if len(cell_size) != bounds.dimensions:
            raise ValueError("cell_size dimensions must match bounds")
        for axis, size in enumerate(cell_size):
            validate_coordinate(size, f"cell_size[{axis}]")
            if size <= 0:
                raise ValueError(f"cell_size[{axis}] must be greater than zero")
        limits = {
            "max_total_cells": max_total_cells,
            "max_cells_per_entry": max_cells_per_entry,
            "max_cells_per_query": max_cells_per_query,
            "max_total_postings": max_total_postings,
            "max_estimated_bytes": max_estimated_bytes,
        }
        for name, value in limits.items():
            validate_coordinate(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        strategy = SparseGridStrategy(
            bounds,
            cell_size,
            max_total_cells=max_total_cells,
            max_cells_per_entry=max_cells_per_entry,
            max_cells_per_query=max_cells_per_query,
            max_total_postings=max_total_postings,
            max_estimated_bytes=max_estimated_bytes,
        )
        self._bounds = bounds
        self._cell_size = cell_size
        super().__init__(
            bounds.dimensions,
            strategy,
            payload_cloner=payload_cloner,
        )

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def cell_size(self) -> tuple[int, ...]:
        return self._cell_size

    def _validate_box(self, box: Box) -> None:
        super()._validate_box(box)
        if not self._bounds.contains(box):
            raise ValueError("box must be contained in the index bounds")
