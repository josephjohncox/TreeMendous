"""Exact Cartesian scan oracle for approximate Morton candidates."""

from __future__ import annotations


def search(
    rows: list[tuple[str, int, int, int, int]],
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> tuple[str, ...]:
    """Return exact rectangle intersections without Morton production code."""
    return tuple(
        item_id
        for item_id, left, bottom, right, top in rows
        if left < max_x and min_x < right and bottom < max_y and min_y < top
    )
