"""Validated build DAGs split into weighted contiguous topological shards."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    nonempty,
    positive,
)


@dataclass(frozen=True)
class BuildTask:
    """A uniquely named weighted DAG node."""

    name: str
    dependencies: tuple[str, ...] = ()
    weight: int = 1


@dataclass(frozen=True)
class BuildShard:
    """A contiguous slice of deterministic topological order."""

    shard_id: int
    tasks: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class BuildShardingSnapshot:
    """Immutable shard plan and completed tasks."""

    topological_order: tuple[str, ...]
    shards: tuple[BuildShard, ...]
    completed: tuple[str, ...]


class BuildShardingEngine:
    """Validate dependencies and execute weighted contiguous build shards.

    Execution is only simulated by recording dependency-safe task completion.
    An actual distributed build runner must persist completion and reject stale
    fencing tokens; this engine's claims and events are process-local.
    """

    def __init__(
        self,
        tasks: object,
        *,
        shard_count: int,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise TypeError("tasks must be a sequence")
        if not tasks:
            raise ValueError("tasks must not be empty")
        positive(shard_count, "shard_count")
        by_name: dict[str, BuildTask] = {}
        checked_tasks: list[BuildTask] = []
        for raw_task in cast(Sequence[object], tasks):
            if not isinstance(raw_task, BuildTask):
                raise TypeError("tasks must contain BuildTask values")
            task = raw_task
            nonempty(task.name, "task name")
            positive(task.weight, "task weight")
            if task.name in by_name:
                raise ValueError("task names must be unique")
            if len(task.dependencies) != len(set(task.dependencies)):
                raise ValueError("task dependencies must be unique")
            by_name[task.name] = task
            checked_tasks.append(task)
        missing = sorted(
            {dependency for task in checked_tasks for dependency in task.dependencies}
            - set(by_name)
        )
        if missing:
            raise ValueError(f"unknown task dependencies: {missing!r}")
        if any(task.name in task.dependencies for task in checked_tasks):
            raise ValueError("tasks cannot depend on themselves")
        order = self._topological_order(by_name)
        selected_count = min(shard_count, len(order))
        self._tasks = by_name
        self._order = order
        self._shards = self._partition(order, by_name, selected_count)
        self._completed: set[str] = set()
        self._runtime = PartitionRuntime(len(self._shards), clock=clock)

    @staticmethod
    def _topological_order(tasks: dict[str, BuildTask]) -> tuple[str, ...]:
        indegree = {name: len(task.dependencies) for name, task in tasks.items()}
        children: dict[str, list[str]] = {name: [] for name in tasks}
        for task in tasks.values():
            for dependency in task.dependencies:
                children[dependency].append(task.name)
        ready = [name for name, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            name = heapq.heappop(ready)
            order.append(name)
            for child in sorted(children[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(tasks):
            raise ValueError("build dependency graph contains a cycle")
        return tuple(order)

    @staticmethod
    def _partition(
        order: tuple[str, ...], tasks: dict[str, BuildTask], count: int
    ) -> tuple[BuildShard, ...]:
        result: list[BuildShard] = []
        cursor = 0
        remaining_weight = sum(tasks[name].weight for name in order)
        for shard_id in range(count):
            shards_left = count - shard_id
            items_left = len(order) - cursor
            take = 1
            if shards_left > 1:
                target = remaining_weight / shards_left
                current = tasks[order[cursor]].weight
                maximum = items_left - (shards_left - 1)
                while take < maximum:
                    next_weight = tasks[order[cursor + take]].weight
                    if current + next_weight > target and abs(current - target) <= abs(
                        current + next_weight - target
                    ):
                        break
                    current += next_weight
                    take += 1
            else:
                take = items_left
            names = order[cursor : cursor + take]
            weight = sum(tasks[name].weight for name in names)
            result.append(BuildShard(shard_id, names, weight))
            cursor += take
            remaining_weight -= weight
        return tuple(result)

    @property
    def shards(self) -> tuple[BuildShard, ...]:
        """Return the immutable contiguous shard plan."""
        return self._shards

    def claim(self, owner: str, length: int = 1) -> WorkClaim:
        """Claim contiguous shard IDs."""
        return self._runtime.claim(owner, length)

    def execute_claim(self, claim: WorkClaim) -> tuple[str, ...]:
        """Record a shard band only when all external dependencies are done."""

        def prepare() -> tuple[tuple[str, ...], set[str]]:
            names = tuple(
                name
                for shard in self._shards[claim.span.start : claim.span.end]
                for name in shard.tasks
            )
            local = set(names)
            unavailable = sorted(
                {
                    dependency
                    for name in names
                    for dependency in self._tasks[name].dependencies
                    if dependency not in self._completed and dependency not in local
                }
            )
            if unavailable:
                raise RuntimeError(
                    f"shard dependencies are incomplete: {unavailable!r}"
                )
            completed = self._completed.copy()
            for name in names:
                if not set(self._tasks[name].dependencies) <= completed | local:
                    raise RuntimeError("shard is not internally topological")
                completed.add(name)
            return names, completed

        prepared = self._runtime.execute_claim(
            claim,
            kind="built",
            prepare=prepare,
            commit=lambda value: setattr(self, "_completed", value[1]),
            result=lambda value: {"tasks": len(value[0])},
        )
        return prepared[0]

    def run(self) -> tuple[str, ...]:
        """Execute all shards in dependency-safe order."""
        while True:
            try:
                claim = self.claim("local")
            except ClaimUnavailableError:
                break
            self.execute_claim(claim)
        return tuple(name for name in self._order if name in self._completed)

    def _snapshot(self) -> BuildShardingSnapshot:
        return BuildShardingSnapshot(
            self._order,
            self._shards,
            tuple(name for name in self._order if name in self._completed),
        )

    def snapshot(self) -> BuildShardingSnapshot:
        """Return immutable plan and completion state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[BuildShardingSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def create_build_sharding(
    tasks: Sequence[BuildTask] = (
        BuildTask("compile-a", (), 3),
        BuildTask("compile-b", (), 2),
        BuildTask("test", ("compile-a", "compile-b"), 4),
    ),
    *,
    shard_count: int = 2,
    clock: Clock | None = None,
) -> BuildShardingEngine:
    """Create a validated weighted build-sharding plan."""
    return BuildShardingEngine(tasks, shard_count=shard_count, clock=clock)
