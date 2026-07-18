"""Independent geometry oracle for correctness-checked benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.domain import Span, validate_coordinate, validate_length


@dataclass(frozen=True)
class OracleMutation:
    """Observable mutation evidence produced by the reference model."""

    changed_length: int
    touched_intervals: int
    fully_covered: bool


class RangeOracle:
    """Small ordered-list model for half-open free ranges.

    This deliberately does not call Tree-Mendous implementation code. Benchmark
    traces are validated against it before any timing sample is reported.
    """

    def __init__(self, domain: tuple[tuple[int, int], ...]) -> None:
        self._domain = tuple(Span(start, end) for start, end in domain)
        self._intervals: list[tuple[int, int]] = []

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._intervals)

    @property
    def total(self) -> int:
        return sum(end - start for start, end in self._intervals)

    def _validate_domain(self, span: Span) -> None:
        if not any(part.contains(span) for part in self._domain):
            raise ValueError("span must be contained in the managed domain")

    def _covered(self, span: Span) -> bool:
        cursor = span.start
        for start, end in self._intervals:
            if end <= cursor:
                continue
            if start > cursor:
                return False
            cursor = max(cursor, end)
            if cursor >= span.end:
                return True
        return False

    def add(self, start: int, end: int) -> OracleMutation:
        span = Span(start, end)
        self._validate_domain(span)
        before_total = self.total
        fully_covered = self._covered(span)
        cursor = start
        changed_components = 0
        for current_start, current_end in self._intervals:
            if current_end <= cursor:
                continue
            if current_start >= end:
                break
            if current_start > cursor:
                changed_components += 1
            cursor = max(cursor, current_end)
            if cursor >= end:
                break
        if cursor < end:
            changed_components += 1
        result: list[tuple[int, int]] = []
        merged_start, merged_end = start, end
        inserted = False
        for current_start, current_end in self._intervals:
            if current_end < merged_start:
                result.append((current_start, current_end))
            elif merged_end < current_start:
                if not inserted:
                    result.append((merged_start, merged_end))
                    inserted = True
                result.append((current_start, current_end))
            else:
                merged_start = min(merged_start, current_start)
                merged_end = max(merged_end, current_end)
        if not inserted:
            result.append((merged_start, merged_end))
        self._intervals = result
        changed_length = self.total - before_total
        return OracleMutation(changed_length, changed_components, fully_covered)

    def discard(self, start: int, end: int) -> OracleMutation:
        span = Span(start, end)
        self._validate_domain(span)
        before_total = self.total
        fully_covered = self._covered(span)
        result: list[tuple[int, int]] = []
        touched = 0
        for current_start, current_end in self._intervals:
            if current_end <= start or current_start >= end:
                result.append((current_start, current_end))
                continue
            touched += 1
            if current_start < start:
                result.append((current_start, start))
            if current_end > end:
                result.append((end, current_end))
        self._intervals = result
        return OracleMutation(before_total - self.total, touched, fully_covered)

    def first_fit(
        self, length: int, *, not_before: int, not_after: int | None = None
    ) -> tuple[int, int] | None:
        validate_coordinate(not_before, "not_before")
        validate_length(length)
        if not_after is not None:
            validate_coordinate(not_after, "not_after")
            if not_after <= not_before:
                raise ValueError("not_after must be greater than not_before")
        for interval_start, interval_end in self._intervals:
            start = max(interval_start, not_before)
            end = start + length
            if end <= interval_end and (not_after is None or end <= not_after):
                return start, end
        return None

    def allocate(
        self, length: int, *, not_before: int, not_after: int | None = None
    ) -> tuple[tuple[int, int] | None, OracleMutation | None]:
        found = self.first_fit(length, not_before=not_before, not_after=not_after)
        if found is None:
            return None, None
        return found, self.discard(*found)
