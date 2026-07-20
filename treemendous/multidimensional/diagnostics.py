"""Structural diagnostics for experimental multidimensional indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from treemendous.multidimensional.domain import BoxEntry


@dataclass(frozen=True)
class BoxIndexDiagnostics:
    """Read-only structural evidence for a BoxIndex implementation."""

    algorithm: Literal["linear"]
    dimensions: int
    version: int
    entry_count: int
    distinct_box_count: int
    duplicate_entry_count: int


def linear_diagnostics(
    dimensions: int,
    version: int,
    entries: tuple[BoxEntry, ...],
) -> BoxIndexDiagnostics:
    """Describe an insertion-ordered linear entry store."""
    distinct_box_count = len({entry.box for entry in entries})
    return BoxIndexDiagnostics(
        algorithm="linear",
        dimensions=dimensions,
        version=version,
        entry_count=len(entries),
        distinct_box_count=distinct_box_count,
        duplicate_entry_count=len(entries) - distinct_box_count,
    )
