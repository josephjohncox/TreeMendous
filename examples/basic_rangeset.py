#!/usr/bin/env python3
"""Allocate a two-hour CPU scheduling window with the public API."""

from treemendous import Span, create_range_set


def main() -> None:
    schedule = create_range_set(domain=(0, 24), backend="py_boundary")
    schedule.discard(Span(0, 9))
    schedule.discard(Span(17, 24))

    booking = schedule.allocate(2, not_before=9, not_after=17)
    assert booking is not None
    assert booking.span == Span(9, 11)

    stats = schedule.stats()
    assert stats.total_free == 6
    assert stats.total_occupied == 18
    print(f"allocated [{booking.start}, {booking.end})")


if __name__ == "__main__":
    main()
