"""Independent fixed-grid multipart completion oracle."""

from treemendous.domain import Span


def state(
    object_size: int, part_size: int, completed: set[int]
) -> tuple[tuple[Span, ...], Span | None]:
    missing_bytes = {
        offset
        for offset in range(object_size)
        if offset // part_size + 1 not in completed
    }
    missing: list[Span] = []
    for offset in sorted(missing_bytes):
        if missing and missing[-1].end == offset:
            missing[-1] = Span(missing[-1].start, offset + 1)
        else:
            missing.append(Span(offset, offset + 1))
    cursor = 0
    while cursor // part_size + 1 in completed and cursor < object_size:
        cursor = min(object_size, cursor + part_size)
    return tuple(missing), Span(0, cursor) if cursor else None
