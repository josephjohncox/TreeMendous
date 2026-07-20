"""Genomic annotation catalog contract."""

import pytest

from tests.oracles.applications.catalogs.genomic_annotation_overlap import (
    overlapping as oracle_overlapping,
)
from treemendous.applications.catalogs.genomic_annotation_overlap import (
    GenomicAnnotationCatalog,
    create_catalog,
)


def test_annotations_preserve_context_nesting_identity_and_coverage() -> None:
    catalog = GenomicAnnotationCatalog()
    gene = catalog.add(
        "gene-1",
        10,
        30,
        assembly="GRCh38",
        contig="chr1",
        strand="+",
        feature_type="gene",
    )
    exon = catalog.add(
        "exon-1",
        12,
        18,
        assembly="GRCh38",
        contig="chr1",
        strand="+",
        feature_type="exon",
        parent_id="gene-1",
    )
    catalog.add(
        "other",
        12,
        18,
        assembly="GRCh38",
        contig="chr2",
        strand="-",
        feature_type="read",
    )
    rows = [
        ("gene-1", "GRCh38", "chr1", "+", 10, 30),
        ("exon-1", "GRCh38", "chr1", "+", 12, 18),
        ("other", "GRCh38", "chr2", "-", 12, 18),
    ]
    actual = catalog.overlapping("GRCh38", "chr1", 11, 20)
    assert tuple(record.payload.feature_id for record in actual) == oracle_overlapping(
        rows, "GRCh38", "chr1", 11, 20
    )
    assert catalog.children("GRCh38", "gene-1")[0].handle == exon
    assert catalog.coverage("GRCh38", "chr1", 10, 20).maximum_count == 2
    assert catalog.update(exon, start=13).handle == exon
    assert catalog.remove(gene).handle == gene
    assert catalog.snapshot().records[0].handle == exon


def test_annotation_input_and_parent_validators_reject_ambiguous_records() -> None:
    catalog = create_catalog()

    with pytest.raises(ValueError, match="feature_id must be a nonempty string"):
        catalog.add("", 1, 2, assembly="GRCh38", contig="chr1", feature_type="gene")
    with pytest.raises(ValueError, match="strand must be"):
        catalog.add(
            "bad-strand",
            1,
            2,
            assembly="GRCh38",
            contig="chr1",
            strand="?",  # type: ignore[arg-type]
            feature_type="gene",
        )
    with pytest.raises(ValueError, match="parent_id must be a nonempty string"):
        catalog.add(
            "child",
            1,
            2,
            assembly="GRCh38",
            contig="chr1",
            feature_type="gene",
            parent_id="",
        )
    with pytest.raises(ValueError, match="cannot be its own parent"):
        catalog.add(
            "self",
            1,
            2,
            assembly="GRCh38",
            contig="chr1",
            feature_type="gene",
            parent_id="self",
        )
    with pytest.raises(ValueError, match="assembly must be a nonempty string"):
        catalog.add(
            "bad-assembly", 1, 2, assembly="", contig="chr1", feature_type="gene"
        )


def test_annotation_update_validates_edits_and_preserves_handle_identity() -> None:
    catalog = create_catalog()
    handle = catalog.add(
        "exon-1",
        10,
        20,
        assembly="GRCh38",
        contig="chr1",
        strand="+",
        feature_type="exon",
        parent_id="gene-1",
    )
    before = catalog.snapshot()

    with pytest.raises(ValueError, match="strand must be"):
        catalog.update(handle, strand="?")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parent_id must be a string or None"):
        catalog.update(handle, parent_id=1)
    with pytest.raises(ValueError, match="parent_id must be a nonempty string"):
        catalog.update(handle, parent_id="")
    with pytest.raises(ValueError, match="cannot be its own parent"):
        catalog.update(handle, parent_id="exon-1")

    updated = catalog.update(
        handle,
        start=11,
        end=22,
        assembly="T2T-CHM13",
        contig="chr2",
        strand="-",
        feature_type="transcript",
        parent_id=None,
    )
    assert updated.handle == handle
    assert updated.start == 11
    assert updated.end == 22
    assert updated.payload.assembly == "T2T-CHM13"
    assert updated.payload.contig == "chr2"
    assert updated.payload.strand == "-"
    assert updated.payload.feature_type == "transcript"
    assert updated.payload.parent_id is None
    assert before.records[0].payload.parent_id == "gene-1"


def test_annotation_filters_children_coverage_and_removal_are_identity_exact() -> None:
    catalog = create_catalog()
    matching = catalog.add(
        "matching",
        1,
        8,
        assembly="A",
        contig="chr1",
        strand="+",
        feature_type="exon",
        parent_id="gene",
    )
    catalog.add(
        "wrong-strand",
        1,
        8,
        assembly="A",
        contig="chr1",
        strand="-",
        feature_type="exon",
        parent_id="gene",
    )
    catalog.add(
        "wrong-type",
        1,
        8,
        assembly="A",
        contig="chr1",
        strand="+",
        feature_type="gene",
    )
    catalog.add(
        "wrong-assembly",
        1,
        8,
        assembly="B",
        contig="chr1",
        strand="+",
        feature_type="exon",
        parent_id="gene",
    )
    catalog.add(
        "wrong-contig",
        1,
        8,
        assembly="A",
        contig="chr2",
        strand="+",
        feature_type="exon",
        parent_id="gene",
    )

    filtered = catalog.overlapping("A", "chr1", 0, 10, strand="+", feature_type="exon")
    assert len(filtered) == 1
    assert filtered[0].handle == matching
    children = catalog.children("A", "gene")
    actual_child_ids = tuple(record.payload.feature_id for record in children)
    expected_child_ids = ("matching", "wrong-strand", "wrong-contig")
    assert actual_child_ids == expected_child_ids
    coverage = catalog.coverage("A", "chr1", 0, 10)
    assert coverage.maximum_count == 3
    assert catalog.remove(matching).handle == matching
    with pytest.raises(KeyError):
        catalog.remove(matching)
