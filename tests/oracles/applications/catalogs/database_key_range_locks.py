"""Naive integer-band database lock oracle."""

from __future__ import annotations


def conflicts(
    rows: list[tuple[str, str, str, int, int, str]],
    table: str,
    owner: str,
    start: int,
    end: int,
    mode: str,
) -> tuple[str, ...]:
    """Return incompatible locks by explicit interval comparisons."""
    return tuple(
        lock_id
        for lock_id, row_table, row_owner, left, right, row_mode in rows
        if row_table == table
        and row_owner != owner
        and left < end
        and start < right
        and (mode == "exclusive" or row_mode == "exclusive")
    )
