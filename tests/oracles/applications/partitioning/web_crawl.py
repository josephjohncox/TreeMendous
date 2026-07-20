"""Independent deterministic traversal oracle for finite page fixtures."""

from collections.abc import Mapping, Sequence


def expected_order(seed: str, links: Mapping[str, Sequence[str]], limit: int) -> tuple[str, ...]:
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
