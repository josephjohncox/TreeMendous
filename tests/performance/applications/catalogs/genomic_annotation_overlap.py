"""Smoke benchmark for genomic overlap queries."""

from time import perf_counter

from treemendous.applications.catalogs.genomic_annotation_overlap import (
    GenomicAnnotationCatalog,
)


def run_smoke(iterations: int = 500) -> float:
    catalog = GenomicAnnotationCatalog()
    for index in range(200):
        catalog.add(
            f"f{index}",
            index * 5,
            index * 5 + 20,
            assembly="GRCh38",
            contig="chr1",
            feature_type="exon",
        )
    started = perf_counter()
    for index in range(iterations):
        catalog.overlapping("GRCh38", "chr1", index % 900, index % 900 + 30)
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
