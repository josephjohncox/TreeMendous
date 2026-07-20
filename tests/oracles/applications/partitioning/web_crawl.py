"""Independent deterministic traversal oracle for finite page fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def expected_order(
    seed: str, links: Mapping[str, Sequence[str]], limit: int
) -> tuple[str, ...]:
    """Traverse a canonical finite link graph in sorted breadth-first order."""
    queue = [seed]
    known = {seed}
    visited: list[str] = []
    while queue and len(visited) < limit:
        url = queue.pop(0)
        visited.append(url)
        for link in sorted(set(links[url])):
            if link not in known:
                known.add(link)
                queue.append(link)
    return tuple(visited)


def expected_frontier(
    seed: str, links: Mapping[str, Sequence[str]], limit: int
) -> tuple[str, ...]:
    """Return the pending queue after the same bounded traversal."""
    queue = [seed]
    known = {seed}
    visited = 0
    while queue and visited < limit:
        url = queue.pop(0)
        visited += 1
        for link in sorted(set(links[url])):
            if link not in known:
                known.add(link)
                queue.append(link)
    return tuple(queue)
