"""Independent brute-force heap placement oracle."""

from treemendous.domain import Span


def place(
    free: tuple[Span, ...], size: int, header: int, redzone: int, alignment: int
) -> tuple[Span, Span] | None:
    total = header + 2 * redzone + size
    for chunk in free:
        for raw_start in range(chunk.start, chunk.end - total + 1):
            payload_start = raw_start + header + redzone
            if payload_start % alignment == 0:
                return Span(raw_start, raw_start + total), Span(
                    payload_start, payload_start + size
                )
    return None
