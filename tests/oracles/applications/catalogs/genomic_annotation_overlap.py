"""Naive genomic overlap oracle; deliberately independent of production code."""

from __future__ import annotations


def overlapping(
    rows: list[tuple[str, str, str, str, int, int]],
    assembly: str,
    contig: str,
    start: int,
    end: int,
) -> tuple[str, ...]:
    """Scan bounded rows and return matching feature IDs."""
    return tuple(
        feature_id
        for feature_id, row_assembly, row_contig, _strand, left, right in rows
        if row_assembly == assembly
        and row_contig == contig
        and left < end
        and start < right
    )
