"""Independent geometry oracle for correctness-checked benchmarks."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
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
    """Independent sorted-list model for half-open free ranges.

    The model owns its value types, validators, indexing, and mutation logic. It
    does not import or call Tree-Mendous production code. Binary-search entry
    points and a cached total keep the oracle usable for six-figure benchmark
    datasets without changing its deliberately simple reference algorithm.
    """

    def __init__(
        self,
        domain: tuple[tuple[int, int], ...],
        *,
        initially_available: bool = False,
    ) -> None:
        parts = sorted(_OracleSpan(start, end) for start, end in domain)
        normalized: list[_OracleSpan] = []
        for part in parts:
            if normalized and part.start <= normalized[-1].end:
                previous = normalized[-1]
                normalized[-1] = _OracleSpan(
                    previous.start, max(previous.end, part.end)
                )
            else:
                normalized.append(part)
        self._domain = tuple(normalized)
        self._domain_starts = [part.start for part in normalized]
        self._domain_total = sum(part.end - part.start for part in normalized)
        self._intervals = (
            [(part.start, part.end) for part in normalized]
            if initially_available
            else []
        )
        self._starts = [start for start, _ in self._intervals]
        self._total = self._domain_total if initially_available else 0

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._intervals)

    @property
    def total(self) -> int:
        return self._total

    @property
    def domain_total(self) -> int:
        return self._domain_total

    @property
    def domain_bounds(self) -> tuple[int | None, int | None]:
        if not self._domain:
            return None, None
        return self._domain[0].start, self._domain[-1].end

    def _validate_domain(self, span: _OracleSpan) -> None:
        index = bisect_right(self._domain_starts, span.start) - 1
        if index < 0 or not self._domain[index].contains(span):
            raise ValueError("span must be contained in the managed domain")

    def _first_intersection(self, start: int) -> int:
        index = max(0, bisect_right(self._starts, start) - 1)
        while index < len(self._intervals) and self._intervals[index][1] <= start:
            index += 1
        return index

    def _covered(self, span: _OracleSpan) -> bool:
        cursor = span.start
        index = self._first_intersection(span.start)
        for start, end in self._intervals[index:]:
            if start > cursor:
                return False
            cursor = max(cursor, end)
            if cursor >= span.end:
                return True
        return False

    def add(self, start: int, end: int) -> OracleMutation:
        span = _OracleSpan(start, end)
        self._validate_domain(span)
        fully_covered = self._covered(span)

        merge_left = bisect_left(self._starts, start)
        if merge_left and self._intervals[merge_left - 1][1] >= start:
            merge_left -= 1
        merge_end = end
        merge_right = merge_left
        while (
            merge_right < len(self._intervals)
            and self._intervals[merge_right][0] <= merge_end
        ):
            merge_end = max(merge_end, self._intervals[merge_right][1])
            merge_right += 1
        merge_start = (
            min(start, self._intervals[merge_left][0])
            if merge_left < merge_right
            else start
        )

        cursor = start
        changed_components = 0
        for current_start, current_end in self._intervals[merge_left:merge_right]:
            if current_end <= start or current_start >= end:
                continue
            if current_start > cursor:
                changed_components += 1
            cursor = max(cursor, min(current_end, end))
        if cursor < end:
            changed_components += 1

        removed = sum(
            current_end - current_start
            for current_start, current_end in self._intervals[merge_left:merge_right]
        )
        replacement = (merge_start, merge_end)
        self._intervals[merge_left:merge_right] = [replacement]
        self._starts[merge_left:merge_right] = [merge_start]
        changed_length = merge_end - merge_start - removed
        self._total += changed_length
        return OracleMutation(changed_length, changed_components, fully_covered)

    def discard(self, start: int, end: int) -> OracleMutation:
        span = _OracleSpan(start, end)
        self._validate_domain(span)
        fully_covered = self._covered(span)
        left = self._first_intersection(start)
        right = left
        while right < len(self._intervals) and self._intervals[right][0] < end:
            right += 1

        affected = self._intervals[left:right]
        replacement: list[tuple[int, int]] = []
        if affected and affected[0][0] < start:
            replacement.append((affected[0][0], start))
        if affected and affected[-1][1] > end:
            replacement.append((end, affected[-1][1]))
        changed_length = sum(
            max(0, min(current_end, end) - max(current_start, start))
            for current_start, current_end in affected
        )
        self._intervals[left:right] = replacement
        self._starts[left:right] = [part_start for part_start, _ in replacement]
        self._total -= changed_length
        return OracleMutation(changed_length, len(affected), fully_covered)

    def first_fit(
        self, length: int, *, not_before: int, not_after: int | None = None
    ) -> tuple[int, int] | None:
        _validate_coordinate(not_before, "not_before")
        _validate_length(length)
        if not_after is not None:
            _validate_coordinate(not_after, "not_after")
            if not_after <= not_before:
                raise ValueError("not_after must be greater than not_before")
        index = self._first_intersection(not_before)
        for interval_start, interval_end in self._intervals[index:]:
            if not_after is not None and interval_start >= not_after:
                break
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

    def overlaps(self, start: int, end: int) -> bool:
        span = _OracleSpan(start, end)
        self._validate_domain(span)
        index = self._first_intersection(start)
        return index < len(self._intervals) and self._intervals[index][0] < end
