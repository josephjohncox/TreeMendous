"""Independent exact oracle for build sharding."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

TaskSpec = tuple[str, tuple[str, ...], int]
ShardSpec = tuple[int, tuple[str, ...], int]


def is_dependency_order(
    order: Sequence[str], dependencies: Mapping[str, Sequence[str]]
) -> bool:
    """Return whether ``order`` is a complete dependency-safe ordering."""
    position = {name: index for index, name in enumerate(order)}
    return set(order) == set(dependencies) and all(
        position[dependency] < position[name]
        for name, values in dependencies.items()
        for dependency in values
    )


def expected_plan(
    tasks: Sequence[TaskSpec], shard_count: int
) -> tuple[tuple[str, ...], tuple[ShardSpec, ...]]:
    """Compute the lexical topological order and weighted contiguous shards."""
    dependencies = {name: values for name, values, _ in tasks}
    weights = {name: weight for name, _, weight in tasks}
    children: dict[str, list[str]] = {name: [] for name in dependencies}
    indegree = {name: len(values) for name, values in dependencies.items()}
    for name, values in dependencies.items():
        for dependency in values:
            children[dependency].append(name)
    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        ordered.append(name)
        for child in sorted(children[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    order = tuple(ordered)

    count = min(shard_count, len(order))
    shards: list[ShardSpec] = []
    cursor = 0
    remaining_weight = sum(weights[name] for name in order)
    for shard_id in range(count):
        shards_left = count - shard_id
        items_left = len(order) - cursor
        take = 1
        if shards_left > 1:
            target = remaining_weight / shards_left
            current = weights[order[cursor]]
            maximum = items_left - (shards_left - 1)
            while take < maximum:
                next_weight = weights[order[cursor + take]]
                combined = current + next_weight
                if combined > target and abs(current - target) <= abs(
                    combined - target
                ):
                    break
                current = combined
                take += 1
        else:
            take = items_left
        names = order[cursor : cursor + take]
        weight = sum(weights[name] for name in names)
        shards.append((shard_id, names, weight))
        cursor += take
        remaining_weight -= weight
    return order, tuple(shards)
