"""Protocols for the experimental multidimensional index surface."""

from __future__ import annotations

from typing import Any, Protocol

from treemendous.multidimensional.diagnostics import BoxIndexDiagnostics
from treemendous.multidimensional.domain import (
    Box,
    BoxEntry,
    BoxHandle,
    BoxIndexSnapshot,
)


class BoxIndexProtocol(Protocol):
    @property
    def dimensions(self) -> int: ...

    def __len__(self) -> int: ...

    def insert(self, box: Box, data: Any = None) -> BoxHandle: ...

    def get(self, handle: BoxHandle) -> BoxEntry: ...

    def update(
        self,
        handle: BoxHandle,
        *,
        box: Box | None = None,
        data: Any = ...,
    ) -> BoxEntry: ...

    def remove(self, handle: BoxHandle) -> BoxEntry: ...

    def entries(self) -> tuple[BoxEntry, ...]: ...

    def overlaps(self, box: Box) -> tuple[BoxEntry, ...]: ...

    def snapshot(self) -> BoxIndexSnapshot: ...

    def diagnostics(self) -> BoxIndexDiagnostics: ...
