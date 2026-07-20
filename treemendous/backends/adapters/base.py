"""The geometry-only seam between ``RangeSet`` and raw implementations."""

from __future__ import annotations

from typing import Any

from treemendous.backends.normalize import normalize_intervals
from treemendous.domain import IntervalResult


class BackendAdapter:
    """Hide raw result shapes behind three canonical geometry operations."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def release(self, start: int, end: int) -> None:
        self.implementation.release_interval(start, end)

    def reserve(self, start: int, end: int) -> None:
        self.implementation.reserve_interval(start, end)

    def intervals(self) -> list[IntervalResult]:
        return normalize_intervals(self.implementation.get_intervals())
