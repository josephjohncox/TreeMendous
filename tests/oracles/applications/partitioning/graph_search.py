"""Independent list-based breadth-first search oracle."""

from collections.abc import Mapping, Sequence


def expected_bfs(
    graph: Mapping[str, Sequence[str]], start: str
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    queue = [start]
    distance = {start: 0}
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in sorted(set(graph[node])):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return tuple(order), tuple(sorted(distance.items()))
