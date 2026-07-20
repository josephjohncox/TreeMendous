# Genomic annotation overlap

## Model

`GenomicAnnotationCatalog` stores each gene, exon, variant, or read as a separate half-open interval record. The payload retains assembly, contig, strand, feature type, feature ID, and an optional parent feature ID. Coincident features therefore remain independently addressable; coverage is only a derived view and never replaces records.

Queries require assembly and contig and may filter strand or feature type. `children` follows retained parent IDs for direct nesting. The same numeric coordinates on another assembly or contig cannot leak into results. Results use original insertion order, while `coverage` reports identity-bearing segments and peak overlap.

## Mutation and validation

`add` validates nonempty identifiers and the `+`, `-`, or `.` strand vocabulary. Coordinates are zero-based, half-open, and nonempty. `update` preserves the stable handle and original order; `remove` deletes only that handle. `snapshot` is immutable and retains all remaining identities.

## Example and complexity

```python
catalog.add("exon-7", 100, 140, assembly="GRCh38", contig="chr1",
            strand="+", feature_type="exon", parent_id="gene-2")
hits = catalog.overlapping("GRCh38", "chr1", 120, 121)
```

The current pure-Python engine scans numeric interval candidates and then applies genomic dimensions. It is intended as correct application semantics, not a claim of a specialized genome database index.
