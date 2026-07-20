"""Correctness-checked smoke workload for build DAG sharding."""

from tests.oracles.applications.partitioning.build_sharding import is_dependency_order
from treemendous.applications.partitioning.build_sharding import (
    BuildShardingEngine,
    BuildTask,
)


def run_smoke() -> int:
    tasks = tuple(BuildTask(f"task-{i:03}", (() if i == 0 else (f"task-{i - 1:03}",)), (i % 7) + 1) for i in range(100))
    engine = BuildShardingEngine(tasks, shard_count=11)
    order = engine.run()
    dependencies = {task.name: task.dependencies for task in tasks}
    if not is_dependency_order(order, dependencies):
        raise AssertionError("build shard order violates dependency oracle")
    if tuple(name for shard in engine.shards for name in shard.tasks) != order:
        raise AssertionError("build shards are not contiguous")
    return len(order)
