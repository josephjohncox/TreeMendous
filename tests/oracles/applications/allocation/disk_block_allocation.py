"""Independent block bitmap allocation oracle."""

from treemendous.domain import Span


def first_extent(total: int, unavailable: set[int], count: int) -> Span | None:
    for start in range(total - count + 1):
        if all(block not in unavailable for block in range(start, start + count)):
            return Span(start, start + count)
    return None
