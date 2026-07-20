"""Genomic annotation catalog contract."""

from tests.oracles.applications.catalogs.genomic_annotation_overlap import (
    overlapping as oracle_overlapping,
)
from treemendous.applications.catalogs.genomic_annotation_overlap import (
    GenomicAnnotationCatalog,
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
