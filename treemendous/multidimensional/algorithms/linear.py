"""Correctness-first linear overlap scan for multidimensional entries."""

from __future__ import annotations

from collections.abc import Iterable

from treemendous.multidimensional.domain import Box, BoxEntry


def overlapping_entries(
    entries: Iterable[BoxEntry], query: Box
) -> tuple[BoxEntry, ...]:
    """Return matching entries without changing insertion order."""
    return tuple(entry for entry in entries if entry.box.overlaps(query))
