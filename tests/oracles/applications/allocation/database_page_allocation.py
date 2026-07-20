"""Independent tablespace bitmap oracle."""

from treemendous.domain import Span


def allocate(total: int, occupied: set[int], count: int) -> Span | None:
    for start in range(total - count + 1):
        candidate = set(range(start, start + count))
        if candidate.isdisjoint(occupied):
            return Span(start, start + count)
    return None
