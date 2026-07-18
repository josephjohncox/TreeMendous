"""Explicit raw-backend adapters."""

from __future__ import annotations

from typing import Any

from treemendous.basic.protocols import (
    standardize_interval_result,
    standardize_intervals_list,
)
from treemendous.domain import IntervalResult


class BackendAdapter:
    supports_payloads = False

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def release(self, start: int, end: int, data: Any = None) -> None:
        self.implementation.release_interval(start, end)

    def reserve(self, start: int, end: int) -> None:
        self.implementation.reserve_interval(start, end)

    def find(self, start: int, length: int) -> IntervalResult | None:
        return standardize_interval_result(
            self.implementation.find_interval(start, length)
        )

    def intervals(self) -> list[IntervalResult]:
        return standardize_intervals_list(self.implementation.get_intervals())

    def total(self) -> int:
        return self.implementation.get_total_available_length()


class PythonBackendAdapter(BackendAdapter):
    supports_payloads = True

    def release(self, start: int, end: int, data: Any = None) -> None:
        self.implementation.release_interval(start, end, data)

    def reserve(self, start: int, end: int) -> None:
        self.implementation.reserve_interval(start, end)


class CppBackendAdapter(BackendAdapter):
    pass
