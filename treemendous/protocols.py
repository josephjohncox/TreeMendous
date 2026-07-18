"""Canonical public protocols for range-set backends."""

from __future__ import annotations

from typing import Any, Protocol

from treemendous.domain import (
    AvailabilityStats,
    IntervalResult,
    MutationResult,
    RangeSnapshot,
    Span,
)


class RangeSetProtocol(Protocol):
    def add(self, span: Span, payload: Any = None) -> MutationResult: ...
    def discard(
        self, span: Span, *, require_covered: bool = False
    ) -> MutationResult: ...
    def first_fit(
        self,
        length: int,
        *,
        not_before: int,
        not_after: int | None = None,
        payload_predicate: Any = None,
    ) -> IntervalResult | None: ...
    def allocate(
        self,
        length: int,
        *,
        not_before: int,
        not_after: int | None = None,
        payload_predicate: Any = None,
    ) -> IntervalResult | None: ...
    def intervals(self) -> tuple[IntervalResult, ...]: ...
    def stats(self) -> AvailabilityStats: ...
    def snapshot(self) -> RangeSnapshot: ...
