"""The geometry-only seam between ``RangeSet`` and raw implementations."""

from __future__ import annotations

from typing import Any

from treemendous.backends.normalize import normalize_interval, normalize_intervals
from treemendous.domain import IntervalResult, MutationResult, Span


class BackendAdapter:
    """Normalize raw geometry operations and optional accelerated capabilities."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation
        explicitly_authoritative = bool(
            type(implementation).__dict__.get(
                "_treemendous_authoritative_geometry", False
            )
        )
        self._supports_authoritative_geometry = explicitly_authoritative and all(
            callable(getattr(implementation, name, None))
            for name in (
                "release_with_delta",
                "reserve_with_delta",
                "find_interval",
                "find_overlapping_intervals",
            )
        )
        self._supports_atomic_allocate = (
            self._supports_authoritative_geometry
            and callable(getattr(implementation, "allocate_interval", None))
        )
        self._supports_geometry_stats = self._supports_authoritative_geometry and all(
            callable(getattr(implementation, name, None))
            for name in (
                "get_interval_count",
                "get_largest_available_length",
            )
        )
        self._supports_managed_domain = (
            self._supports_authoritative_geometry
            and callable(getattr(implementation, "set_managed_domain", None))
        )
        # Fully-native scalar mutators return only the changed length, building
        # no Span/MutationResult.  They are an accelerated addition to the
        # authoritative surface, never a replacement for it.
        self._supports_scalar_delta = self._supports_authoritative_geometry and all(
            callable(getattr(implementation, name, None))
            for name in (
                "release_delta_length",
                "reserve_delta_length",
            )
        )

    @property
    def supports_authoritative_geometry(self) -> bool:
        return self._supports_authoritative_geometry

    @property
    def supports_scalar_delta(self) -> bool:
        return self._supports_scalar_delta

    @property
    def supports_atomic_allocate(self) -> bool:
        return self._supports_atomic_allocate

    @property
    def supports_geometry_stats(self) -> bool:
        return self._supports_geometry_stats

    @property
    def supports_managed_domain(self) -> bool:
        return self._supports_managed_domain

    def configure_domain(self, spans: tuple[Span, ...]) -> None:
        self.implementation.set_managed_domain(
            [(span.start, span.end) for span in spans]
        )

    def release(self, start: int, end: int) -> None:
        self.implementation.release_interval(start, end)

    def reserve(self, start: int, end: int) -> None:
        self.implementation.reserve_interval(start, end)

    def intervals(self) -> list[IntervalResult]:
        return normalize_intervals(self.implementation.get_intervals())

    @staticmethod
    def _mutation_result(raw: Any) -> MutationResult:
        if not isinstance(raw, MutationResult):
            raise TypeError(
                "authoritative geometry backends must return MutationResult"
            )
        return raw

    def release_with_delta(self, start: int, end: int) -> MutationResult:
        raw = self.implementation.release_with_delta(start, end)
        return self._mutation_result(raw)

    def reserve_with_delta(
        self, start: int, end: int, require_covered: bool
    ) -> MutationResult:
        raw = self.implementation.reserve_with_delta(start, end, require_covered)
        return self._mutation_result(raw)

    def release_delta_length(self, start: int, end: int) -> int:
        return self.implementation.release_delta_length(start, end)

    def reserve_delta_length(self, start: int, end: int, require_covered: bool) -> int:
        return self.implementation.reserve_delta_length(start, end, require_covered)

    def first_fit(self, start: int, length: int) -> IntervalResult | None:
        return normalize_interval(self.implementation.find_interval(start, length))

    def allocate(
        self, start: int, length: int, not_after: int | None
    ) -> IntervalResult | None:
        return normalize_interval(
            self.implementation.allocate_interval(start, length, not_after)
        )

    def geometry_stats(self) -> tuple[int, int]:
        return (
            self.implementation.get_interval_count(),
            self.implementation.get_largest_available_length(),
        )

    def overlaps(self, start: int, end: int) -> list[IntervalResult]:
        return normalize_intervals(
            self.implementation.find_overlapping_intervals(start, end)
        )
