"""Naive versioned diagnostic range oracle."""

from __future__ import annotations


def query(
    rows: list[tuple[str, str, int, int, int, int]],
    file: str,
    version: int,
    start: int,
    end: int,
    minimum_severity: int,
) -> tuple[str, ...]:
    """Scan and severity-order bounded diagnostic rows."""
    matches = [
        row
        for row in rows
        if row[1] == file
        and row[2] == version
        and row[3] >= minimum_severity
        and row[4] < end
        and start < row[5]
    ]
    matches.sort(key=lambda row: -row[3])
    return tuple(row[0] for row in matches)
