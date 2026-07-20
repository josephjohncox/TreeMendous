"""Deterministic breadth-first graph frontier expansion."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from treemendous.applications._shared.claiming import ClaimUnavailableError
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    nonempty,
    positive,
)


@dataclass(frozen=True)
class GraphSearchSnapshot:
    """Immutable BFS order, distances, and pending frontier."""

    order: tuple[str, ...]
    distances: tuple[tuple[str, int], ...]
    frontier: tuple[str, ...]


class GraphSearchEngine:
    """Expand a validated graph in deterministic BFS/frontier order.

    Graph data and the visited/frontier sets live in one process. Distributed
    workers require an external durable frontier and fencing-token enforcement.
    """

    def __init__(
        self,
        graph: Mapping[str, Sequence[str]],
        start: str,
        *,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(graph, Mapping) or not graph:
            raise ValueError("graph must be a nonempty mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for vertex, neighbors in graph.items():
            nonempty(vertex, "vertex")
            if isinstance(neighbors, (str, bytes)) or not isinstance(
                neighbors, Sequence
            ):
                raise TypeError("neighbors must be a sequence of vertex names")
            checked: list[str] = []
            for neighbor in neighbors:
                checked.append(nonempty(neighbor, "neighbor"))
            normalized[vertex] = tuple(sorted(set(checked)))
        missing = sorted(
            {item for values in normalized.values() for item in values}
            - set(normalized)
        )
        if missing:
            raise ValueError(f"graph references missing vertices: {missing!r}")
        nonempty(start, "start")
        if start not in normalized:
            raise ValueError("start vertex is not in graph")
        self._graph = normalized
        self._frontier: deque[str] = deque((start,))
        self._queued = {start}
        self._visited: set[str] = set()
        self._distances = {start: 0}
        self._order: list[str] = []
        self._runtime = PartitionRuntime(len(normalized), clock=clock)

    def expand(self, *, width: int = 1, owner: str = "local") -> tuple[str, ...]:
        """Claim expansion ordinals and expand up to ``width`` frontier nodes."""
        positive(width, "width")
        if not self._frontier:
            return ()
        count = min(width, len(self._frontier))
        claim = self._runtime.claim(owner, count)

        def prepare() -> tuple[
            tuple[str, ...],
            deque[str],
            set[str],
            set[str],
            dict[str, int],
            list[str],
        ]:
            frontier = deque(self._frontier)
            queued = self._queued.copy()
            visited = self._visited.copy()
            distances = self._distances.copy()
            order = self._order.copy()
            expanded: list[str] = []
            for _ in range(claim.span.length):
                if not frontier:
                    break
                vertex = frontier.popleft()
                if vertex in visited:
                    continue
                visited.add(vertex)
                order.append(vertex)
                expanded.append(vertex)
                distance = distances[vertex] + 1
                for neighbor in self._graph[vertex]:
                    if neighbor not in visited and neighbor not in queued:
                        queued.add(neighbor)
                        distances[neighbor] = distance
                        frontier.append(neighbor)
            return tuple(expanded), frontier, queued, visited, distances, order

        def commit(
            value: tuple[
                tuple[str, ...],
                deque[str],
                set[str],
                set[str],
                dict[str, int],
                list[str],
            ],
        ) -> None:
            (
                _,
                self._frontier,
                self._queued,
                self._visited,
                self._distances,
                self._order,
            ) = value

        prepared = self._runtime.execute_claim(
            claim,
            kind="expanded",
            prepare=prepare,
            commit=commit,
            result=lambda value: {"vertices": len(value[0])},
        )
        return prepared[0]

    def run(self, *, frontier_width: int = 32) -> GraphSearchSnapshot:
        """Run BFS to exhaustion."""
        positive(frontier_width, "frontier_width")
        while self._frontier:
            try:
                self.expand(width=frontier_width)
            except ClaimUnavailableError:
                break
        return self.snapshot()

    def _snapshot(self) -> GraphSearchSnapshot:
        return GraphSearchSnapshot(
            tuple(self._order),
            tuple(sorted((key, self._distances[key]) for key in self._visited)),
            tuple(self._frontier),
        )

    def snapshot(self) -> GraphSearchSnapshot:
        """Return detached BFS state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[GraphSearchSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def create_graph_search(
    graph: Mapping[str, Sequence[str]] | None = None,
    start: str = "a",
    *,
    clock: Clock | None = None,
) -> GraphSearchEngine:
    """Create a deterministic BFS job."""
    selected = (
        {"a": ("b", "c"), "b": ("d",), "c": (), "d": ()} if graph is None else graph
    )
    return GraphSearchEngine(selected, start, clock=clock)
