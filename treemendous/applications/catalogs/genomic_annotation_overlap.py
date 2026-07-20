"""Assembly-aware, identity-preserving genomic annotation catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from treemendous.applications._shared.coverage import (
    CoverageSnapshot,
    coverage_snapshot,
)
from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)

Strand = Literal["+", "-", "."]


@dataclass(frozen=True)
class Annotation:
    """One named genomic feature; ``parent_id`` preserves nesting."""

    feature_id: str
    assembly: str
    contig: str
    strand: Strand
    feature_type: str
    parent_id: str | None = None


AnnotationRecord = IntervalRecord[str, Annotation]
AnnotationHandle = RecordHandle[str]
AnnotationSnapshot = IntervalRecordSnapshot[str, Annotation]
_MISSING = object()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


class GenomicAnnotationCatalog:
    """Store coincident and nested features without coalescing their identities."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, Annotation](lambda value: value)

    def add(
        self,
        feature_id: str,
        start: int,
        end: int,
        *,
        assembly: str,
        contig: str,
        strand: Strand = ".",
        feature_type: str,
        parent_id: str | None = None,
    ) -> AnnotationHandle:
        """Add a feature using zero-based half-open coordinates."""
        _text(feature_id, "feature_id")
        if strand not in ("+", "-", "."):
            raise ValueError("strand must be '+', '-', or '.'")
        if parent_id is not None:
            _text(parent_id, "parent_id")
            if parent_id == feature_id:
                raise ValueError("a feature cannot be its own parent")
        annotation = Annotation(
            feature_id,
            _text(assembly, "assembly"),
            _text(contig, "contig"),
            strand,
            _text(feature_type, "feature_type"),
            parent_id,
        )
        return self._index.insert(feature_id, start, end, annotation)

    def update(
        self,
        handle: AnnotationHandle,
        *,
        start: int | None = None,
        end: int | None = None,
        assembly: str | None = None,
        contig: str | None = None,
        strand: Strand | None = None,
        feature_type: str | None = None,
        parent_id: str | None | object = _MISSING,
    ) -> AnnotationRecord:
        """Update a feature while retaining its handle and insertion order.

        Passing ``parent_id=None`` explicitly clears a parent; omitting it
        preserves the existing nesting relationship.
        """
        current = self._index.get(handle)
        if strand is not None and strand not in ("+", "-", "."):
            raise ValueError("strand must be '+', '-', or '.'")
        parent = current.payload.parent_id if parent_id is _MISSING else parent_id
        if parent is not None:
            if not isinstance(parent, str):
                raise TypeError("parent_id must be a string or None")
            _text(parent, "parent_id")
            if parent == current.payload.feature_id:
                raise ValueError("a feature cannot be its own parent")
        payload = replace(
            current.payload,
            assembly=current.payload.assembly
            if assembly is None
            else _text(assembly, "assembly"),
            contig=current.payload.contig
            if contig is None
            else _text(contig, "contig"),
            strand=current.payload.strand if strand is None else strand,
            feature_type=(
                current.payload.feature_type
                if feature_type is None
                else _text(feature_type, "feature_type")
            ),
            parent_id=parent,
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=payload
        )

    def remove(self, handle: AnnotationHandle) -> AnnotationRecord:
        """Remove exactly one feature identity."""
        return self._index.remove(handle, owner=handle.owner)

    def overlapping(
        self,
        assembly: str,
        contig: str,
        start: int,
        end: int,
        *,
        strand: Strand | None = None,
        feature_type: str | None = None,
    ) -> tuple[AnnotationRecord, ...]:
        """Return matching features in stable insertion order."""
        return tuple(
            record
            for record in self._index.overlaps(start, end)
            if record.payload.assembly == assembly
            and record.payload.contig == contig
            and (strand is None or record.payload.strand == strand)
            and (feature_type is None or record.payload.feature_type == feature_type)
        )

    def children(self, assembly: str, parent_id: str) -> tuple[AnnotationRecord, ...]:
        """Return direct nested features for ``parent_id`` in insertion order."""
        return tuple(
            record
            for record in self._index.snapshot().records
            if record.payload.assembly == assembly
            and record.payload.parent_id == parent_id
        )

    def coverage(
        self, assembly: str, contig: str, start: int, end: int
    ) -> CoverageSnapshot[str]:
        """Return an identity-bearing derived coverage projection."""
        records = (
            record
            for record in self._index.overlaps(start, end)
            if record.payload.assembly == assembly and record.payload.contig == contig
        )
        return coverage_snapshot(records, start=start, end=end)

    def snapshot(self) -> AnnotationSnapshot:
        """Return a detached immutable catalog snapshot."""
        return self._index.snapshot()


def create_catalog() -> GenomicAnnotationCatalog:
    """Create an empty genomic annotation catalog."""
    return GenomicAnnotationCatalog()
