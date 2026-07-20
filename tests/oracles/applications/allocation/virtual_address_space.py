"""Independent virtual mapping layout arithmetic."""

from treemendous.domain import Span


def fixed_layout(
    address: int, length: int, page_size: int, guards: int
) -> tuple[Span, Span]:
    payload_page = address // page_size
    mapped_pages = (length + page_size - 1) // page_size
    return (
        Span(payload_page - guards, payload_page + mapped_pages + guards),
        Span(address, address + mapped_pages * page_size),
    )
