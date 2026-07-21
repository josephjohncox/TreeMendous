"""Finite point-set model for one-dimensional interval algebra."""

from __future__ import annotations

from dataclasses import dataclass

SpanPair = tuple[int, int]
PointSet = frozenset[int]


@dataclass(frozen=True)
class ModeledMutation:
    """Extensional result and exact evidence for one modeled mutation."""

    after: PointSet
    changed: tuple[SpanPair, ...]
    changed_length: int
    fully_covered: bool


def valid_spans(extent: int) -> tuple[SpanPair, ...]:
    """Return every nonempty half-open span inside ``[0, extent)``."""
    return tuple(
        (start, end) for start in range(extent) for end in range(start + 1, extent + 1)
    )


def all_point_sets(extent: int) -> tuple[PointSet, ...]:
    """Return every subset of the bounded integer universe."""
    return tuple(
        frozenset(point for point in range(extent) if mask & (1 << point))
        for mask in range(1 << extent)
    )


def points(span: SpanPair) -> PointSet:
    """Return the denotation of one half-open integer span."""
    start, end = span
    if start >= end:
        raise ValueError("span must satisfy start < end")
    return frozenset(range(start, end))


def denotation(spans: tuple[SpanPair, ...]) -> PointSet:
    """Return the union denoted by a tuple of spans."""
    result: set[int] = set()
    for span in spans:
        result.update(points(span))
    return frozenset(result)


def runs(free: PointSet) -> tuple[SpanPair, ...]:
    """Return the unique ordered maximal-run normal form of ``free``."""
    if not free:
        return ()
    ordered = sorted(free)
    result: list[SpanPair] = []
    start = previous = ordered[0]
    for point in ordered[1:]:
        if point != previous + 1:
            result.append((start, previous + 1))
            start = point
        previous = point
    result.append((start, previous + 1))
    return tuple(result)


def normalize(spans: tuple[SpanPair, ...]) -> tuple[SpanPair, ...]:
    """Normalize any finite span representation by its denotation."""
    return runs(denotation(spans))


def measure(free: PointSet) -> int:
    """Return finite counting measure."""
    return len(free)


def release(free: PointSet, target: SpanPair) -> ModeledMutation:
    """Model ``F' = F union A`` and ``changed = NF(A minus F)``."""
    target_points = points(target)
    changed = target_points - free
    return ModeledMutation(
        after=free | target_points,
        changed=runs(frozenset(changed)),
        changed_length=len(changed),
        fully_covered=target_points <= free,
    )


def reserve(
    free: PointSet, target: SpanPair, *, require_covered: bool = False
) -> ModeledMutation:
    """Model ``F' = F minus A`` including covered-reserve rejection."""
    target_points = points(target)
    covered = target_points <= free
    if require_covered and not covered:
        return ModeledMutation(free, (), 0, False)
    changed = target_points & free
    return ModeledMutation(
        after=free - target_points,
        changed=runs(frozenset(changed)),
        changed_length=len(changed),
        fully_covered=covered,
    )


def release_absorbed_count(free: PointSet, target: SpanPair) -> int:
    """Count canonical old spans absorbed by an effective release splice."""
    start, end = target
    merged_end = end
    count = 0
    for current_start, current_end in runs(free):
        if current_end < start:
            continue
        if current_start > merged_end:
            break
        count += 1
        merged_end = max(merged_end, current_end)
    return count


def reserve_structure(free: PointSet, target: SpanPair) -> tuple[int, int]:
    """Return overlapping old-span count ``k`` and boundary remainders ``r``."""
    start, end = target
    affected = tuple(
        (current_start, current_end)
        for current_start, current_end in runs(free)
        if current_start < end and start < current_end
    )
    if not affected:
        return 0, 0
    left_remainder = 1 if affected[0][0] < start else 0
    right_remainder = 1 if affected[-1][1] > end else 0
    remainders = left_remainder + right_remainder
    return len(affected), remainders


def safely_reject_fit(
    *, request_start: int, length: int, upper_length: int, upper_end: int
) -> bool:
    """Apply the sound fit-pruning rejection certificate."""
    return upper_length < length or upper_end <= request_start
