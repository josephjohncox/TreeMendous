"""Independent per-byte CDN request coverage oracle."""

from treemendous.domain import Span


def coverage(request: Span, resident: tuple[Span, ...]) -> tuple[int, tuple[Span, ...]]:
    cached = {
        point
        for span in resident
        for point in range(span.start, span.end)
        if request.start <= point < request.end
    }
    missing: list[Span] = []
    for point in range(request.start, request.end):
        if point in cached:
            continue
        if missing and missing[-1].end == point:
            missing[-1] = Span(missing[-1].start, point + 1)
        else:
            missing.append(Span(point, point + 1))
    return len(cached), tuple(missing)
