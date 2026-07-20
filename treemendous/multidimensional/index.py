"""Thread-safe experimental identity-preserving multidimensional index."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from treemendous.domain import validate_coordinate
from treemendous.multidimensional.algorithms.linear import overlapping_entries
from treemendous.multidimensional.diagnostics import (
    BoxIndexDiagnostics,
    linear_diagnostics,
)
from treemendous.multidimensional.domain import (
    Box,
    BoxEntry,
    BoxHandle,
    BoxIndexSnapshot,
    _detached_entry,
)

_MISSING = object()


class BoxIndex:
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
        self._dimensions = dimensions
        self._owner = uuid4()
        self._next_sequence = 1
        self._version = 0
        self._entries: dict[BoxHandle, BoxEntry] = {}
        self._lock = RLock()
        self._payload_activity_lock = Lock()
        self._payload_activity = 0
        self._payload_cloner = payload_cloner

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @contextmanager
    def _payload_copying(self) -> Iterator[None]:
        with self._payload_activity_lock:
            self._payload_activity += 1
        try:
            yield
        finally:
            with self._payload_activity_lock:
                self._payload_activity -= 1

    def _payload_is_active(self) -> bool:
        with self._payload_activity_lock:
            return self._payload_activity > 0

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
            return self._entries[handle]
        except KeyError:
            raise KeyError(handle) from None

    def insert(self, box: Box, data: Any = None) -> BoxHandle:
        self._validate_box(box)
        with self._mutation():
            with self._payload_copying():
                owned_data = self._clone_payload(data)
            handle = BoxHandle(self._next_sequence, self._owner)
            self._entries[handle] = BoxEntry(handle, box, owned_data)
            self._next_sequence += 1
            self._version += 1
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
            entry = self._entry_for(handle)
            if box is None and data is _MISSING:
                raise ValueError("update requires a box or data replacement")
            if box is not None:
                self._validate_box(box)
            replacement_box = entry.box if box is None else box
            with self._payload_copying():
                replacement_data = (
                    self._clone_payload(entry.data)
                    if data is _MISSING
                    else self._clone_payload(data)
                )
                candidate = BoxEntry(handle, replacement_box, replacement_data)
                detached = self._detach(candidate)
            self._entries[handle] = candidate
            self._version += 1
            return detached

    def remove(self, handle: BoxHandle) -> BoxEntry:
        with self._mutation():
            entry = self._entry_for(handle)
            with self._payload_copying():
                detached = self._detach(entry)
            del self._entries[handle]
            self._version += 1
            return detached

    def entries(self) -> tuple[BoxEntry, ...]:
        with self._lock:
            with self._payload_copying():
                return tuple(self._detach(entry) for entry in self._entries.values())

    def overlaps(self, box: Box) -> tuple[BoxEntry, ...]:
        self._validate_box(box)
        with self._lock:
            matches = overlapping_entries(self._entries.values(), box)
            with self._payload_copying():
                return tuple(self._detach(entry) for entry in matches)

    def snapshot(self) -> BoxIndexSnapshot:
        with self._lock:
            with self._payload_copying():
                entries = tuple(self._detach(entry) for entry in self._entries.values())
            return BoxIndexSnapshot(
                self._dimensions,
                self._version,
                entries,
                self._clone_snapshot_payload,
            )

    def diagnostics(self) -> BoxIndexDiagnostics:
        with self._lock:
            return linear_diagnostics(
                self._dimensions,
                self._version,
                tuple(self._entries.values()),
            )
