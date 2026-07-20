"""Attested benchmark for deterministic breadth-first graph search."""

from __future__ import annotations

from tests.oracles.applications.partitioning.graph_search import expected_bfs
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.graph_search import (
    GraphSearchEngine,
    GraphSearchSnapshot,
)

_DEFAULT_OPERATIONS = 256
_MAX_OPERATIONS = 1_000
_DEFAULT_SEED = 29
_FRONTIER_WIDTH = 13


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Expand and attest a bounded connected graph from one start vertex."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    jumps = (1, 2 + seed % 5, 7 + seed % 11)
    graph = {
        str(index): tuple(
            str(index + jump) for jump in jumps if index + jump < operations
        )
        for index in range(operations)
    }
    engine = GraphSearchEngine(graph, "0")

    def execute() -> GraphSearchSnapshot:
        return engine.run(frontier_width=_FRONTIER_WIDTH)

    def observe(raw: GraphSearchSnapshot) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        return ApplicationOutcome(
            results=raw.order,
            final_state={
                "order": snapshot.order,
                "distances": snapshot.distances,
                "frontier": snapshot.frontier,
            },
            counters={
                "vertices_expanded": len(snapshot.order),
                "vertices_discovered": len(snapshot.distances),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        order, distances = expected_bfs(graph, "0")
        return ApplicationOutcome(
            results=order,
            final_state={
                "order": order,
                "distances": distances,
                "frontier": (),
            },
            counters={
                "vertices_expanded": len(order),
                "vertices_discovered": len(distances),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.graph_search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
