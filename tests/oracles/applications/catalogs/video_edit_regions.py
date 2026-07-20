"""Naive video edit region oracle."""

from __future__ import annotations


def affected(
    rows: list[tuple[str, str, str, int, int]],
    start: int,
    end: int,
    tracks: frozenset[str],
) -> tuple[str, ...]:
    """Scan retained edit identities for render invalidation."""
    return tuple(
        region_id
        for region_id, track, _effect, left, right in rows
        if track in tracks and left < end and start < right
    )
