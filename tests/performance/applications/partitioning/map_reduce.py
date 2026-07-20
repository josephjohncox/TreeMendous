"""Attested benchmark for record-split map/reduce."""

from __future__ import annotations

from collections.abc import Iterable

from tests.oracles.applications.partitioning.map_reduce import (
    expected_record_splits,
    expected_word_counts,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.map_reduce import (
    InputSplit,
    MapReduceEngine,
)

_DEFAULT_OPERATIONS = 300
_MAX_OPERATIONS = 1_500
_DEFAULT_SEED = 43
_SPLIT_SIZE = 7
_SHARD_SIZE = 5


def _mapper(unit: bytes) -> Iterable[tuple[str, int]]:
    return ((word.lower(), 1) for word in unit.decode().split())


def _sum(left: int, right: int) -> int:
    return left + right


def _split_tuple(split: InputSplit) -> tuple[int, int, int, tuple[bytes, ...]]:
    return split.split_id, split.start, split.end, split.units


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Map bounded complete records and attest split plan plus reduction."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    data = b"".join(
        (
            f"term-{(index + seed) % 29} bucket-{(index * 7 + seed) % 13} shared\n"
        ).encode()
        for index in range(operations)
    )
    engine = MapReduceEngine(
        data,
        _mapper,
        _sum,
        split_size=_SPLIT_SIZE,
        mode="records",
    )

    def execute() -> tuple[tuple[str, int], ...]:
        return engine.run(shard_size=_SHARD_SIZE)

    def observe(raw: tuple[tuple[str, int], ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        splits = tuple(_split_tuple(split) for split in snapshot.splits)
        return ApplicationOutcome(
            results=raw,
            final_state={
                "mode": snapshot.mode,
                "splits": splits,
                "results": snapshot.results,
            },
            counters={
                "records_mapped": operations,
                "emissions": operations * 3,
                "splits_mapped": len(snapshot.splits),
                "result_keys": len(snapshot.results),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        results = expected_word_counts(data)
        splits = expected_record_splits(data, _SPLIT_SIZE)
        return ApplicationOutcome(
            results=results,
            final_state={
                "mode": "records",
                "splits": splits,
                "results": results,
            },
            counters={
                "records_mapped": operations,
                "emissions": operations * 3,
                "splits_mapped": len(splits),
                "result_keys": len(results),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.map_reduce",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
