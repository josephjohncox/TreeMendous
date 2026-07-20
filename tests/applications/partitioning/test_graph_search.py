"""Graph-search engine contracts."""

import pytest

from tests.oracles.applications.partitioning.graph_search import expected_bfs
from treemendous.applications.partitioning.graph_search import GraphSearchEngine


def test_frontier_expansion_is_deterministic_bfs() -> None:
    graph = {"a": ("c", "b"), "b": ("d",), "c": ("d",), "d": ()}
    snapshot = GraphSearchEngine(graph, "a").run(frontier_width=2)
    empty: tuple[str, ...] = ()
    observed = (snapshot.order, snapshot.distances)
    assert observed == expected_bfs(graph, "a")
    assert snapshot.frontier == empty


def test_graph_search_rejects_dangling_edges() -> None:
    with pytest.raises(ValueError, match="missing"):
        GraphSearchEngine({"a": ("missing",)}, "a")
