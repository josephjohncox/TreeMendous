"""Build-sharding engine contracts."""

import pytest

from tests.oracles.applications.partitioning.build_sharding import is_dependency_order
from treemendous.applications.partitioning.build_sharding import (
    BuildShardingEngine,
    BuildTask,
)


def test_weighted_shards_are_contiguous_and_dependency_safe() -> None:
    tasks = (
        BuildTask("a", (), 5),
        BuildTask("b", (), 1),
        BuildTask("c", ("a",), 2),
        BuildTask("d", ("b", "c"), 3),
    )
    engine = BuildShardingEngine(tasks, shard_count=3)
    order = engine.run()
    dependencies = {task.name: task.dependencies for task in tasks}
    assert is_dependency_order(order, dependencies)
    assert (
        tuple(name for shard in engine.shards for name in shard.tasks)
        == engine.snapshot().topological_order
    )
    assert sum(shard.weight for shard in engine.shards) == 11


def test_build_sharding_rejects_cycles_and_missing_dependencies() -> None:
    with pytest.raises(ValueError, match="cycle"):
        BuildShardingEngine(
            (BuildTask("a", ("b",)), BuildTask("b", ("a",))), shard_count=1
        )
    with pytest.raises(ValueError, match="unknown"):
        BuildShardingEngine((BuildTask("a", ("missing",)),), shard_count=1)
