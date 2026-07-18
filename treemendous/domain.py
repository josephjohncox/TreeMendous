"""Canonical domain values and errors for Tree-Mendous.

All ranges are integer, half-open spans.  The values in this module are storage-
backend independent and form the stable public contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast


class TreeMendousError(Exception):
    """Base class for public Tree-Mendous errors."""


class BackendUnavailableError(TreeMendousError):
    """Raised when an explicitly requested backend cannot be loaded."""


class BackendInvalidError(TreeMendousError):
    """Raised when a backend loads but fails semantic validation."""


class ManagedDomainRequiredError(TreeMendousError):
    """Raised when occupancy analytics are requested without a domain."""


def validate_coordinate(value: int, name: str = "coordinate") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def validate_length(length: int) -> int:
    validate_coordinate(length, "length")
    if length <= 0:
        raise ValueError("length must be greater than zero")
    return length


@dataclass(frozen=True, order=True)
class Span:
    """A validated half-open interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        validate_coordinate(self.start, "start")
        validate_coordinate(self.end, "end")
        if self.start >= self.end:
            raise ValueError("span must satisfy start < end")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


SpanInput: TypeAlias = Span | tuple[int, int] | Iterable[Span | tuple[int, int]]


@dataclass(frozen=True)
class IntervalResult:
    """Canonical result returned by interval queries."""

    start: int
    end: int
    length: int | None = None
    data: Any = None

    def __post_init__(self) -> None:
        Span(self.start, self.end)
        actual = self.end - self.start
        if self.length is None:
            object.__setattr__(self, "length", actual)
        elif self.length != actual:
            raise ValueError("length must equal end - start")

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)


@dataclass(frozen=True)
class ManagedDomain:
    """A normalized tuple of disjoint managed spans."""

    spans: tuple[Span, ...]

    def __init__(self, spans: SpanInput):
        if isinstance(spans, Span):
            raw = [spans]
        elif (
            isinstance(spans, tuple)
            and len(spans) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in spans)
        ):
            raw = [Span(*spans)]
        else:
            raw = []
            items = cast(Iterable[Span | tuple[int, int]], spans)
            for item in items:
                if isinstance(item, Span):
                    raw.append(item)
                else:
                    start, end = item
                    raw.append(Span(start, end))
        if not raw:
            raise ValueError("managed domain must contain at least one span")
        ordered = sorted(raw)
        normalized: list[Span] = []
        for span in ordered:
            if normalized and span.start < normalized[-1].end:
                raise ValueError("managed domain spans must not overlap")
            if normalized and span.start == normalized[-1].end:
                normalized[-1] = Span(normalized[-1].start, span.end)
            else:
                normalized.append(span)
        object.__setattr__(self, "spans", tuple(normalized))

    @property
    def measure(self) -> int:
        return sum(span.length for span in self.spans)

    @property
    def bounds(self) -> tuple[int, int]:
        return self.spans[0].start, self.spans[-1].end

    def contains(self, span: Span) -> bool:
        return any(part.contains(span) for part in self.spans)

    def extended(self, span: Span) -> ManagedDomain:
        """Return the normalized union of this domain and one span."""
        merged: list[Span] = []
        for part in sorted((*self.spans, span)):
            if merged and part.start <= merged[-1].end:
                merged[-1] = Span(merged[-1].start, max(merged[-1].end, part.end))
            else:
                merged.append(part)
        return ManagedDomain(merged)


DomainInput: TypeAlias = ManagedDomain | SpanInput


@dataclass(frozen=True)
class AvailabilityStats:
    total_free: int
    total_occupied: int = 0
    total_space: int | None = None
    free_chunks: int = 0
    largest_chunk: int = 0
    avg_chunk_size: float = 0.0
    utilization: float = 0.0
    fragmentation: float = 0.0
    free_density: float = 0.0
    bounds: tuple[int | None, int | None] = (None, None)

    def __post_init__(self) -> None:
        total_space = (
            self.total_free + self.total_occupied
            if self.total_space is None
            else self.total_space
        )
        if min(self.total_free, self.total_occupied, total_space) < 0:
            raise ValueError("availability measures cannot be negative")
        object.__setattr__(self, "total_space", total_space)
        if total_space:
            object.__setattr__(self, "utilization", self.total_occupied / total_space)
            object.__setattr__(self, "free_density", self.total_free / total_space)
        if self.free_chunks:
            object.__setattr__(
                self, "avg_chunk_size", self.total_free / self.free_chunks
            )
        if self.total_free and self.largest_chunk:
            object.__setattr__(
                self, "fragmentation", 1.0 - self.largest_chunk / self.total_free
            )


@dataclass(frozen=True)
class MutationResult:
    """Mutation evidence; ``fully_covered`` describes the pre-mutation span."""

    changed: tuple[Span, ...]
    changed_length: int
    fully_covered: bool


@dataclass(frozen=True)
class RangeSnapshot:
    """Immutable observable state of a range set."""

    intervals: tuple[IntervalResult, ...]
    total_free: int
    domain: ManagedDomain | None

    def __post_init__(self) -> None:
        if self.total_free != sum(
            interval.end - interval.start for interval in self.intervals
        ):
            raise ValueError("snapshot total does not match intervals")
