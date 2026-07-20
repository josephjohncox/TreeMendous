"""Validated benchmark for priority-aware alert and suppression evaluation."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.timeseries_alert_windows import active
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.timeseries_alert_windows import (
    AlertCatalog,
    AlertEvaluation,
    AlertRecord,
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


def _record(record: AlertRecord) -> tuple[Any, ...]:
    window = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
        record.start,
        record.end,
        record.insertion_order,
        window.window_id,
        window.series,
        window.kind.value,
        window.priority,
        window.label,
    )


def _state(catalog: AlertCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def _result(result: AlertEvaluation) -> dict[str, Any]:
    return {
        "alerts": tuple(_record(record) for record in result.alerts),
        "suppressions": tuple(_record(record) for record in result.suppressions),
        "suppressed": tuple(_record(record) for record in result.suppressed),
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Evaluate bounded timestamps against an independent priority scan."""
    random = _parameters(operations, seed)
    catalog = AlertCatalog()
    rows: list[tuple[str, str, str, int, int, int]] = []
    insertion_order: dict[str, int] = {}
    for index in range(200):
        window_id = f"a{index}"
        start = index * 10
        priority = index % 10
        catalog.add(
            window_id,
            start,
            start + 100,
            series="cpu",
            kind="alert",
            priority=priority,
            label="cpu",
        )
        insertion_order[window_id] = len(rows)
        rows.append((window_id, "cpu", "alert", priority, start, start + 100))
    for index in range(20):
        window_id = f"s{index}"
        start = index * 100
        catalog.add(
            window_id,
            start,
            start + 20,
            series="cpu",
            kind="suppression",
            priority=5,
            label="maintenance",
        )
        insertion_order[window_id] = len(rows)
        rows.append((window_id, "cpu", "suppression", 5, start, start + 20))

    commands = tuple(random.randrange(2_000) for _ in range(operations))
    expected_state = _state(catalog)
    by_id = {row[6]: row for row in expected_state["records"]}
    priority_by_id = {row[0]: row[3] for row in rows}

    def execute() -> tuple[AlertEvaluation, ...]:
        return tuple(catalog.active_at("cpu", timestamp) for timestamp in commands)

    def observe(raw: tuple[AlertEvaluation, ...]) -> ApplicationOutcome:
        results = tuple(_result(result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "evaluation_calls": operations,
                "firing_alerts": sum(len(result["alerts"]) for result in results),
                "suppressed_alerts": sum(
                    len(result["suppressed"]) for result in results
                ),
            },
        )

    def ordered(ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            sorted(
                ids,
                key=lambda item_id: (
                    -priority_by_id[item_id],
                    insertion_order[item_id],
                ),
            )
        )

    def oracle() -> ApplicationOutcome:
        results: list[dict[str, Any]] = []
        for timestamp in commands:
            firing, suppressed = active(rows, "cpu", timestamp)
            suppressions = tuple(
                row[0]
                for row in rows
                if row[1] == "cpu"
                and row[2] == "suppression"
                and row[4] <= timestamp < row[5]
            )
            results.append(
                {
                    "alerts": tuple(by_id[item_id] for item_id in ordered(firing)),
                    "suppressions": tuple(
                        by_id[item_id] for item_id in ordered(suppressions)
                    ),
                    "suppressed": tuple(
                        by_id[item_id] for item_id in ordered(suppressed)
                    ),
                }
            )
        return ApplicationOutcome(
            tuple(results),
            expected_state,
            {
                "evaluation_calls": operations,
                "firing_alerts": sum(len(result["alerts"]) for result in results),
                "suppressed_alerts": sum(
                    len(result["suppressed"]) for result in results
                ),
            },
        )

    return run_application_case(
        scenario_id="catalog-timeseries-alert-evaluation",
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
