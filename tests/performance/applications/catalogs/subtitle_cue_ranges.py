"""Validated benchmark for language-aware active subtitle cue queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.subtitle_cue_ranges import active
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.subtitle_cue_ranges import (
    CueRecord,
    SubtitleCatalog,
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


def _record(record: CueRecord) -> tuple[Any, ...]:
    cue = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
        record.start,
        record.end,
        record.insertion_order,
        cue.cue_id,
        cue.language,
        cue.layer,
        cue.text,
    )


def _state(catalog: SubtitleCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded active-cue queries and attest exact rendering order."""
    random = _parameters(operations, seed)
    catalog = SubtitleCatalog()
    rows: list[tuple[str, str, int, int, int]] = []
    for index in range(200):
        cue_id = f"c{index}"
        start = index * 100
        layer = index % 3
        catalog.add(
            cue_id,
            start,
            start + 500,
            language="en",
            layer=layer,
            text="cue",
        )
        rows.append((cue_id, "en", layer, start, start + 500))

    commands = tuple(random.randrange(20_000) for _ in range(operations))
    expected_state = _state(catalog)
    by_id = {row[6]: row for row in expected_state["records"]}

    def execute() -> tuple[tuple[CueRecord, ...], ...]:
        return tuple(catalog.active_at(time, language="en") for time in commands)

    def observe(raw: tuple[tuple[CueRecord, ...], ...]) -> ApplicationOutcome:
        results = tuple(tuple(_record(record) for record in result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "returned_cues": sum(len(result) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        ids = tuple(active(rows, time, "en") for time in commands)
        results = tuple(tuple(by_id[item_id] for item_id in result) for result in ids)
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_cues": sum(len(result) for result in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-subtitle-active-cues",
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
