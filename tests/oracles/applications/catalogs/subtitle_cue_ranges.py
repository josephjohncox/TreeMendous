"""Naive subtitle active-at-time oracle."""

from __future__ import annotations


def active(
    rows: list[tuple[str, str, int, int, int]], time: int, language: str
) -> tuple[str, ...]:
    """Filter and render-order cue IDs by bounded scan."""
    matches = [row for row in rows if row[1] == language and row[3] <= time < row[4]]
    matches.sort(key=lambda row: (row[2], row[3]))
    return tuple(row[0] for row in matches)
