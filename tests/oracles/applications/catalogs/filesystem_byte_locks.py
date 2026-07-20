"""Naive filesystem lock compatibility oracle."""

from __future__ import annotations


def conflicts(
    rows: list[tuple[str, str, str, int, int, str]],
    file: str,
    owner: str,
    mode: str,
    start: int,
    end: int,
) -> tuple[str, ...]:
    """Return IDs of incompatible other-owner locks by linear scan."""
    return tuple(
        lock_id
        for lock_id, row_file, row_owner, left, right, row_mode in rows
        if row_file == file
        and row_owner != owner
        and left < end
        and start < right
        and (mode == "exclusive" or row_mode == "exclusive")
    )
