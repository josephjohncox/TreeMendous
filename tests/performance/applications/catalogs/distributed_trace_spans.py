"""Validated benchmark for trace overlap and critical-path queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.distributed_trace_spans import overlapping
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.distributed_trace_spans import (
    TraceCatalog,
    TraceRecord,
)

_MAX_OPERATIONS = 10_000
TraceRow = tuple[str, str, str | None, int, int]


def _parameters(operations: int, seed: int) -> Random:
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if not 1 <= operations <= _MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {_MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return Random(seed)


def _record(record: TraceRecord) -> tuple[Any, ...]:
    span = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
        record.start,
        record.end,
        record.insertion_order,
        span.trace_id,
        span.span_id,
        span.parent_span_id,
        span.service,
        span.operation,
    )


def _state(catalog: TraceCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def _critical_path(rows: list[TraceRow], trace_id: str) -> tuple[str, ...]:
    records = [row for row in rows if row[0] == trace_id]
    order = {row[1]: index for index, row in enumerate(records)}
    by_id = {row[1]: row for row in records}
    children: dict[str, list[TraceRow]] = {}
    roots: list[TraceRow] = []
    for row in records:
        parent = row[2]
        if parent is None or parent not in by_id:
            roots.append(row)
        else:
            children.setdefault(parent, []).append(row)

    def best(row: TraceRow) -> tuple[int, tuple[str, ...]]:
        choices = [best(child) for child in children.get(row[1], ())]
        duration = row[4] - row[3]
        if not choices:
            return duration, (row[1],)
        child_duration, child_path = max(
            choices,
            key=lambda choice: (
                choice[0],
                tuple(-order[span_id] for span_id in choice[1]),
            ),
        )
        return duration + child_duration, (row[1], *child_path)

    if not roots:
        return ()
    return max(
        (best(root) for root in roots),
        key=lambda choice: (
            choice[0],
            tuple(-order[span_id] for span_id in choice[1]),
        ),
    )[1]


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded trace queries and validate them with independent scans."""
    random = _parameters(operations, seed)
    catalog = TraceCatalog()
    rows: list[TraceRow] = []
    catalog.add(
        "trace",
        "root",
        0,
        1_000,
        parent_span_id=None,
        service="api",
        operation="request",
    )
    rows.append(("trace", "root", None, 0, 1_000))
    for index in range(100):
        span_id = f"s{index}"
        start = index * 8
        catalog.add(
            "trace",
            span_id,
            start,
            start + 20,
            parent_span_id="root",
            service="worker",
            operation="work",
        )
        rows.append(("trace", span_id, "root", start, start + 20))

    commands = tuple(
        ("overlap", random.randrange(800)) if index % 2 == 0 else ("critical", 0)
        for index in range(operations)
    )
    overlap_rows = [
        (trace_id, span_id, parent or "", start, end)
        for trace_id, span_id, parent, start, end in rows
    ]
    expected_state = _state(catalog)
    by_id = {row[7]: row for row in expected_state["records"]}

    def execute() -> tuple[tuple[str, tuple[TraceRecord, ...]], ...]:
        return tuple(
            (
                kind,
                catalog.overlapping("trace", start, start + 25)
                if kind == "overlap"
                else catalog.critical_path("trace"),
            )
            for kind, start in commands
        )

    def observe(
        raw: tuple[tuple[str, tuple[TraceRecord, ...]], ...],
    ) -> ApplicationOutcome:
        results = tuple(
            (kind, tuple(_record(record) for record in records))
            for kind, records in raw
        )
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "overlap_calls": sum(kind == "overlap" for kind, _ in commands),
                "critical_path_calls": sum(kind == "critical" for kind, _ in commands),
                "returned_spans": sum(len(records) for _, records in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        critical = _critical_path(rows, "trace")
        ids = tuple(
            overlapping(overlap_rows, "trace", start, start + 25)
            if kind == "overlap"
            else critical
            for kind, start in commands
        )
        results = tuple(
            (kind, tuple(by_id[span_id] for span_id in result))
            for (kind, _), result in zip(commands, ids, strict=True)
        )
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "overlap_calls": sum(kind == "overlap" for kind, _ in commands),
                "critical_path_calls": sum(kind == "critical" for kind, _ in commands),
                "returned_spans": sum(len(records) for _, records in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-distributed-trace-queries",
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
