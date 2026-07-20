"""Validated benchmark for encoded database key lock queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.database_key_range_locks import conflicts
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.database_key_range_locks import (
    DatabaseKeyRangeLocks,
    KeyRangeLock,
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


def _lock(lock: KeyRangeLock) -> tuple[Any, ...]:
    handle = lock.handle.lock
    return (
        lock.handle.table,
        handle.owner,
        handle.sequence,
        "engine-lineage",
        lock.start_key,
        lock.end_key,
        lock.encoded_start,
        lock.encoded_end,
        lock.mode.value,
        lock.owner,
        lock.insertion_order,
    )


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded conflict queries and attest results and unchanged lock state."""
    random = _parameters(operations, seed)
    locks = DatabaseKeyRangeLocks()
    rows: list[tuple[str, str, str, int, int, str]] = []
    for index in range(100):
        owner = f"reader-{index}"
        locks.acquire("table", owner, f"k{index:04d}", f"k{index + 1:04d}", "shared")
        rows.append((owner, "table", owner, index, index + 1, "shared"))

    commands = tuple(random.randrange(99) for _ in range(operations))
    expected_state = tuple(_lock(lock) for lock in locks.snapshot().locks)
    by_owner = {row[9]: row for row in expected_state}

    def execute() -> tuple[tuple[KeyRangeLock, ...], ...]:
        return tuple(
            locks.conflicts(
                "table",
                "writer",
                f"k{start:04d}",
                f"k{start + 2:04d}",
                "exclusive",
            )
            for start in commands
        )

    def observe(raw: tuple[tuple[KeyRangeLock, ...], ...]) -> ApplicationOutcome:
        results = tuple(tuple(_lock(lock) for lock in result) for result in raw)
        return ApplicationOutcome(
            results,
            tuple(_lock(lock) for lock in locks.snapshot().locks),
            {
                "query_calls": operations,
                "returned_locks": sum(len(result) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        result_ids = tuple(
            conflicts(rows, "table", "writer", start, start + 2, "exclusive")
            for start in commands
        )
        results = tuple(
            tuple(by_owner[owner] for owner in owners) for owners in result_ids
        )
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_locks": sum(len(result) for result in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-database-key-range-lock-conflicts",
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
