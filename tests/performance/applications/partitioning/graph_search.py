"""Correctness-checked smoke workload for graph BFS."""

from tests.oracles.applications.partitioning.graph_search import expected_bfs
from treemendous.applications.partitioning.graph_search import GraphSearchEngine


def run_smoke() -> int:
    graph = {str(i): tuple(str(j) for j in (i + 1, i + 2) if j < 300) for i in range(300)}
    snapshot = GraphSearchEngine(graph, "0").run(frontier_width=13)
    if (snapshot.order, snapshot.distances) != expected_bfs(graph, "0"):
        raise AssertionError("graph BFS smoke differs from oracle")
    return len(snapshot.order)
