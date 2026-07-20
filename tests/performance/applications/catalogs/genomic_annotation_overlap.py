"""Validated benchmark for genomic annotation overlap queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.genomic_annotation_overlap import overlapping
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.genomic_annotation_overlap import (
    AnnotationRecord,
    GenomicAnnotationCatalog,
)

_MAX_OPERATIONS = 10_000


def _parameters(operations: int, seed: int) -> Random:
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if not 1 <= operations <= _MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {_MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return Random(seed)


def _record(record: AnnotationRecord) -> tuple[Any, ...]:
    annotation = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
        record.start,
        record.end,
        record.insertion_order,
        annotation.feature_id,
        annotation.assembly,
        annotation.contig,
        annotation.strand,
        annotation.feature_type,
        annotation.parent_id,
    )


def _state(catalog: GenomicAnnotationCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded exact-overlap queries against an independent linear scan."""
    random = _parameters(operations, seed)
    catalog = GenomicAnnotationCatalog()
    rows: list[tuple[str, str, str, str, int, int]] = []
    for index in range(200):
        feature_id = f"f{index}"
        start = index * 5
        catalog.add(
            feature_id,
            start,
            start + 20,
            assembly="GRCh38",
            contig="chr1",
            feature_type="exon",
        )
        rows.append((feature_id, "GRCh38", "chr1", ".", start, start + 20))

    commands = tuple(random.randrange(900) for _ in range(operations))
    expected_state = _state(catalog)
    by_id = {row[6]: row for row in expected_state["records"]}

    def execute() -> tuple[tuple[AnnotationRecord, ...], ...]:
        return tuple(
            catalog.overlapping("GRCh38", "chr1", start, start + 30)
            for start in commands
        )

    def observe(raw: tuple[tuple[AnnotationRecord, ...], ...]) -> ApplicationOutcome:
        results = tuple(tuple(_record(record) for record in result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "returned_features": sum(len(result) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        ids = tuple(
            overlapping(rows, "GRCh38", "chr1", start, start + 30) for start in commands
        )
        results = tuple(tuple(by_id[item_id] for item_id in result) for result in ids)
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_features": sum(len(result) for result in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-genomic-annotation-overlap",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke(iterations: int = 500, seed: int = 0) -> ApplicationSample:
    """Delegate the legacy smoke entry point to the validated benchmark."""
    return run_benchmark(operations=iterations, seed=seed)


if __name__ == "__main__":
    print(run_smoke().to_dict())
