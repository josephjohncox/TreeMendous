"""Validated benchmark for versioned source diagnostic queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.source_diagnostic_ranges import query
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.source_diagnostic_ranges import (
    DiagnosticRecord,
    Severity,
    SourceDiagnosticCatalog,
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


def _record(record: DiagnosticRecord) -> tuple[Any, ...]:
    diagnostic = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        str(record.handle.lineage),
        record.start,
        record.end,
        record.insertion_order,
        diagnostic.diagnostic_id,
        diagnostic.file,
        diagnostic.version,
        diagnostic.severity.value,
        diagnostic.message,
    )


def _state(catalog: SourceDiagnosticCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded diagnostic queries and validate severity ordering by scan."""
    random = _parameters(operations, seed)
    catalog = SourceDiagnosticCatalog()
    rows: list[tuple[str, str, int, int, int, int]] = []
    expected_records: list[tuple[Any, ...]] = []
    for index in range(200):
        diagnostic_id = f"d{index}"
        start = index * 3
        handle = catalog.add(
            diagnostic_id,
            start,
            start + 8,
            file="main.py",
            version=1,
            severity=Severity.WARNING,
            message="warning",
        )
        rows.append(
            (diagnostic_id, "main.py", 1, Severity.WARNING.value, start, start + 8)
        )
        expected_records.append(
            (
                diagnostic_id,
                1,
                str(handle.lineage),
                start,
                start + 8,
                index,
                diagnostic_id,
                "main.py",
                1,
                Severity.WARNING.value,
                "warning",
            )
        )

    commands = tuple(random.randrange(500) for _ in range(operations))
    expected_state = {
        "records": tuple(expected_records),
        "next_sequences": tuple((diagnostic_id, 2) for diagnostic_id, *_ in rows),
        "next_insertion_order": len(rows),
    }
    by_id = {row[6]: row for row in expected_records}

    def execute() -> tuple[tuple[DiagnosticRecord, ...], ...]:
        return tuple(
            catalog.diagnostics("main.py", 1, start, start + 20) for start in commands
        )

    def observe(raw: tuple[tuple[DiagnosticRecord, ...], ...]) -> ApplicationOutcome:
        results = tuple(tuple(_record(record) for record in result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "returned_diagnostics": sum(len(result) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        ids = tuple(
            query(rows, "main.py", 1, start, start + 20, Severity.INFO.value)
            for start in commands
        )
        results = tuple(tuple(by_id[item_id] for item_id in result) for result in ids)
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_diagnostics": sum(len(result) for result in results),
            },
        )

    return run_application_case(
        scenario_id="source-diagnostic-ranges",
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
