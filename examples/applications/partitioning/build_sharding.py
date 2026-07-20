#!/usr/bin/env python3
"""Run weighted DAG build sharding from any working directory."""

from treemendous.applications.partitioning.build_sharding import (
    BuildShardingEngine,
    BuildTask,
)


def main() -> None:
    tasks = (BuildTask("compile", (), 4), BuildTask("test", ("compile",), 2))
    engine = BuildShardingEngine(tasks, shard_count=2)
    if engine.run() != ("compile", "test"):
        raise RuntimeError("unexpected build order")
    print("build-sharding: 2 dependency-safe shards")


if __name__ == "__main__":
    main()
