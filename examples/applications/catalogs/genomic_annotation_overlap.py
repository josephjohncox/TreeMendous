"""Query nested genomic features without losing identity."""

from treemendous.applications.catalogs.genomic_annotation_overlap import (
    GenomicAnnotationCatalog,
)


def main() -> None:
    catalog = GenomicAnnotationCatalog()
    catalog.add(
        "gene",
        10,
        50,
        assembly="GRCh38",
        contig="chr1",
        strand="+",
        feature_type="gene",
    )
    catalog.add(
        "exon",
        20,
        30,
        assembly="GRCh38",
        contig="chr1",
        strand="+",
        feature_type="exon",
        parent_id="gene",
    )
    print(
        [
            record.payload.feature_id
            for record in catalog.overlapping("GRCh38", "chr1", 25, 26)
        ]
    )


if __name__ == "__main__":
    main()
