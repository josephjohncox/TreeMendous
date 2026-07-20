"""Attested benchmark for weighted build-DAG sharding."""

from __future__ import annotations

import random

from tests.oracles.applications.partitioning.build_sharding import expected_plan
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.build_sharding import (
    BuildShardingEngine,
    BuildTask,
)

_DEFAULT_OPERATIONS = 96
_MAX_OPERATIONS = 512
_DEFAULT_SEED = 17


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Execute and attest one bounded weighted build plan."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    randomizer = random.Random(seed)
    task_specs = tuple(
        (
            f"task-{index:04}",
            tuple(
                f"task-{dependency:04}"
                for dependency in (index - 1, index - 3)
                if dependency >= 0
            ),
            randomizer.randrange(1, 10),
        )
        for index in range(operations)
    )
    shard_count = min(11, operations)
    tasks = tuple(BuildTask(*spec) for spec in task_specs)
    engine = BuildShardingEngine(tasks, shard_count=shard_count)

    def execute() -> tuple[str, ...]:
        return engine.run()

    def observe(raw: tuple[str, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        shards = tuple(
            (shard.shard_id, shard.tasks, shard.weight) for shard in snapshot.shards
        )
        return ApplicationOutcome(
            results=raw,
            final_state={
                "topological_order": snapshot.topological_order,
                "shards": shards,
                "completed": snapshot.completed,
            },
            counters={
                "tasks_completed": len(snapshot.completed),
                "shards_executed": len(snapshot.shards),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        order, shards = expected_plan(task_specs, shard_count)
        return ApplicationOutcome(
            results=order,
            final_state={
                "topological_order": order,
                "shards": shards,
                "completed": order,
            },
            counters={
                "tasks_completed": len(order),
                "shards_executed": len(shards),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.build_sharding",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
