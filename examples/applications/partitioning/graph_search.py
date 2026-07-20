#!/usr/bin/env python3
"""Run deterministic BFS from any working directory."""

from treemendous.applications.partitioning.graph_search import GraphSearchEngine


def main() -> None:
    graph = {"a": ("c", "b"), "b": ("d",), "c": (), "d": ()}
    order = GraphSearchEngine(graph, "a").run().order
    if order != ("a", "b", "c", "d"):
        raise RuntimeError("unexpected BFS order")
    print("graph-search: a,b,c,d")


if __name__ == "__main__":
    main()
