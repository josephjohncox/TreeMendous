"""Naive distributed trace overlap oracle."""

from __future__ import annotations


def overlapping(
    rows: list[tuple[str, str, str, int, int]],
    trace_id: str,
    start: int,
    end: int,
) -> tuple[str, ...]:
    """Return overlapping span IDs by linear scan."""
    return tuple(
        span_id
        for row_trace, span_id, _parent, left, right in rows
        if row_trace == trace_id and left < end and start < right
    )
