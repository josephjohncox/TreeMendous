"""Validated benchmark for video edit invalidation queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.video_edit_regions import affected
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.video_edit_regions import (
    EditRecord,
    Invalidation,
    VideoEditCatalog,
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


def _identity(record: EditRecord) -> tuple[Any, ...]:
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
    )


def _record(record: EditRecord) -> tuple[Any, ...]:
    region = record.payload
    return (
        *_identity(record),
        record.start,
        record.end,
        record.insertion_order,
        region.region_id,
        region.track,
        region.effect,
        region.parameters,
    )


def _state(catalog: VideoEditCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def _actual_result(result: Invalidation) -> dict[str, Any]:
    return {
        "records": tuple(_record(record) for record in result.records),
        "segments": tuple(
            (
                segment.start,
                segment.end,
                tuple(
                    sorted(
                        (handle.owner, handle.sequence, "engine-lineage")
                        for handle in segment.record_ids
                    )
                ),
            )
            for segment in result.coverage.segments
        ),
        "covered_length": result.coverage.covered_length,
        "maximum_count": result.coverage.maximum_count,
    }


def _expected_result(
    records: tuple[tuple[Any, ...], ...], start: int, end: int
) -> dict[str, Any]:
    points = sorted(
        {start, end}
        | {
            max(start, int(record[3]))
            for record in records
            if int(record[3]) < end and start < int(record[4])
        }
        | {
            min(end, int(record[4]))
            for record in records
            if int(record[3]) < end and start < int(record[4])
        }
    )
    segments: list[tuple[int, int, tuple[tuple[Any, ...], ...]]] = []
    for left, right in zip(points, points[1:], strict=False):
        identities = tuple(
            sorted(
                tuple(record[:3])
                for record in records
                if int(record[3]) < right and left < int(record[4])
            )
        )
        if not identities:
            continue
        if segments and segments[-1][1] == left and segments[-1][2] == identities:
            segments[-1] = (segments[-1][0], right, identities)
        else:
            segments.append((left, right, identities))
    return {
        "records": records,
        "segments": tuple(segments),
        "covered_length": sum(right - left for left, right, _ in segments),
        "maximum_count": max((len(ids) for _, _, ids in segments), default=0),
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded invalidations and validate identities and exact coverage."""
    random = _parameters(operations, seed)
    catalog = VideoEditCatalog()
    rows: list[tuple[str, str, str, int, int]] = []
    for index in range(200):
        region_id = f"r{index}"
        start = index * 5
        track = f"v{index % 4}"
        catalog.add(region_id, start, start + 30, track=track, effect="grade")
        rows.append((region_id, track, "grade", start, start + 30))

    commands = tuple(random.randrange(900) for _ in range(operations))
    tracks = frozenset({"v1"})
    expected_state = _state(catalog)
    by_id = {row[6]: row for row in expected_state["records"]}

    def execute() -> tuple[Invalidation, ...]:
        return tuple(
            catalog.invalidation(start, start + 50, tracks=tracks) for start in commands
        )

    def observe(raw: tuple[Invalidation, ...]) -> ApplicationOutcome:
        results = tuple(_actual_result(result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "returned_regions": sum(len(result["records"]) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        ids = tuple(affected(rows, start, start + 50, tracks) for start in commands)
        selected = tuple(tuple(by_id[item_id] for item_id in result) for result in ids)
        results = tuple(
            _expected_result(records, start, start + 50)
            for records, start in zip(selected, commands, strict=True)
        )
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_regions": sum(len(result["records"]) for result in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-video-edit-invalidation",
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
