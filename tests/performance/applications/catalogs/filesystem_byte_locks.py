"""Validated benchmark for filesystem byte-lock conflict queries."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.filesystem_byte_locks import conflicts
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.filesystem_byte_locks import (
    FileLock,
    FilesystemByteLocks,
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


def _lock(lock: FileLock) -> tuple[Any, ...]:
    handle = lock.handle.lock
    return (
        lock.handle.file,
        handle.owner,
        handle.sequence,
        "engine-lineage",
        lock.start,
        lock.end,
        lock.mode.value,
        lock.owner,
        lock.insertion_order,
    )


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded conflict queries and attest every active lock identity."""
    random = _parameters(operations, seed)
    locks = FilesystemByteLocks()
    rows: list[tuple[str, str, str, int, int, str]] = []
    for index in range(200):
        owner = f"reader-{index}"
        start = index * 8
        locks.acquire("data", owner, start, start + 4, "shared")
        rows.append((owner, "data", owner, start, start + 4, "shared"))

    commands = tuple(random.randrange(1_500) for _ in range(operations))
    expected_state = tuple(_lock(lock) for lock in locks.snapshot().locks)
    by_owner = {row[7]: row for row in expected_state}

    def execute() -> tuple[tuple[FileLock, ...], ...]:
        return tuple(
            locks.conflicts("data", "writer", start, start + 16, "exclusive")
            for start in commands
        )

    def observe(raw: tuple[tuple[FileLock, ...], ...]) -> ApplicationOutcome:
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
            conflicts(rows, "data", "writer", "exclusive", start, start + 16)
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
        scenario_id="catalog-filesystem-byte-lock-conflicts",
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
