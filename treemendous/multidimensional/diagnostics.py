"""Structural diagnostics for experimental multidimensional indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from treemendous.multidimensional.domain import BoxEntry


@dataclass(frozen=True)
class BoxIndexDiagnostics:
    """Immutable structural evidence for a multidimensional index."""

    algorithm: Literal["linear", "axis_projection", "sparse_grid"]
    dimensions: int
    version: int
    entry_count: int
    distinct_box_count: int
    duplicate_entry_count: int
    projection_sizes: tuple[int, ...] = ()
    bounds_lower: tuple[int, ...] = ()
    bounds_upper: tuple[int, ...] = ()
    cell_size: tuple[int, ...] = ()
    grid_shape: tuple[int, ...] = ()
    total_possible_cells: int | None = None
    occupied_cell_count: int | None = None
    posting_count: int | None = None
    max_total_cells: int | None = None
    max_cells_per_entry: int | None = None
    max_cells_per_query: int | None = None
    max_total_postings: int | None = None


def _geometry_counts(entries: tuple[BoxEntry, ...]) -> tuple[int, int]:
    distinct_box_count = len({entry.box for entry in entries})
    return distinct_box_count, len(entries) - distinct_box_count


def linear_diagnostics(
    dimensions: int,
    version: int,
    entries: tuple[BoxEntry, ...],
) -> BoxIndexDiagnostics:
    """Describe an insertion-ordered linear entry store."""
    distinct_box_count, duplicate_entry_count = _geometry_counts(entries)
    return BoxIndexDiagnostics(
        algorithm="linear",
        dimensions=dimensions,
        version=version,
        entry_count=len(entries),
        distinct_box_count=distinct_box_count,
        duplicate_entry_count=duplicate_entry_count,
    )


def strategy_diagnostics(
    algorithm: Literal["axis_projection", "sparse_grid"],
    dimensions: int,
    version: int,
    entries: tuple[BoxEntry, ...],
    values: dict[str, object],
) -> BoxIndexDiagnostics:
    """Describe a strategy without recording mutable query instrumentation."""
    distinct_box_count, duplicate_entry_count = _geometry_counts(entries)
    return BoxIndexDiagnostics(
        algorithm=algorithm,
        dimensions=dimensions,
        version=version,
        entry_count=len(entries),
        distinct_box_count=distinct_box_count,
        duplicate_entry_count=duplicate_entry_count,
        projection_sizes=cast(tuple[int, ...], values.get("projection_sizes", ())),
        bounds_lower=cast(tuple[int, ...], values.get("bounds_lower", ())),
        bounds_upper=cast(tuple[int, ...], values.get("bounds_upper", ())),
        cell_size=cast(tuple[int, ...], values.get("cell_size", ())),
        grid_shape=cast(tuple[int, ...], values.get("grid_shape", ())),
        total_possible_cells=cast(int | None, values.get("total_possible_cells")),
        occupied_cell_count=cast(int | None, values.get("occupied_cell_count")),
        posting_count=cast(int | None, values.get("posting_count")),
        max_total_cells=cast(int | None, values.get("max_total_cells")),
        max_cells_per_entry=cast(int | None, values.get("max_cells_per_entry")),
        max_cells_per_query=cast(int | None, values.get("max_cells_per_query")),
        max_total_postings=cast(int | None, values.get("max_total_postings")),
    )
