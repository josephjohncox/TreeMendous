"""Independent geometry oracle for correctness-checked benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_coordinate(value: int, name: str = "coordinate") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_length(length: int) -> int:
    _validate_coordinate(length, "length")
    if length <= 0:
        raise ValueError("length must be greater than zero")
    return length


@dataclass(frozen=True, order=True)
class _OracleSpan:
    """Benchmark-local half-open span; intentionally independent of production."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _validate_coordinate(self.start, "start")
        _validate_coordinate(self.end, "end")
        if self.start >= self.end:
            raise ValueError("span must satisfy start < end")

    def contains(self, other: _OracleSpan) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True)
class OracleMutation:
    """Observable mutation evidence produced by the reference model."""

    changed_length: int
    touched_intervals: int
    fully_covered: bool


class RangeOracle:
    """Small ordered-list model for half-open free ranges.

    The model owns its value types and validators and does not import or call
    Tree-Mendous production code. Benchmark traces are validated against this
    implementation in a replay separate from measured backend execution.
    """

    def __init__(self, domain: tuple[tuple[int, int], ...]) -> None:
        self._domain = tuple(_OracleSpan(start, end) for start, end in domain)
        self._intervals: list[tuple[int, int]] = []

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._intervals)

    @property
    def total(self) -> int:
        return sum(end - start for start, end in self._intervals)

    def _validate_domain(self, span: _OracleSpan) -> None:
        if not any(part.contains(span) for part in self._domain):
            raise ValueError("span must be contained in the managed domain")

    def _covered(self, span: _OracleSpan) -> bool:
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
        span = _OracleSpan(start, end)
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
        span = _OracleSpan(start, end)
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
        _validate_coordinate(not_before, "not_before")
        _validate_length(length)
        if not_after is not None:
            _validate_coordinate(not_after, "not_after")
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
