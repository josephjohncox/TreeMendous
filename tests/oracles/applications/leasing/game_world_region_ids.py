"""Naive set/time oracle for shard region ownership and handoff."""

from __future__ import annotations


class NaiveRegionOracle:
    """Reference model with token-ordered exact ownership transfers."""

    def __init__(self, shards: dict[str, set[int]]) -> None:
        self.free = shards
        self.active: dict[int, tuple[str, str, set[int], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(self, shard: str, owner: str, count: int, ttl: int) -> tuple[int, set[int]]:
        values = next(
            (
                set(range(start, start + count))
                for start in sorted(self.free[shard])
                if set(range(start, start + count)) <= self.free[shard]
            ),
            None,
        )
        if values is None:
            raise RuntimeError("unavailable")
        self.free[shard].difference_update(values)
        token = self.next_token
        self.next_token += 1
        self.active[token] = (shard, owner, values, self.now + ttl)
        return token, values

    def handoff(self, token: int, owner: str, ttl: int) -> tuple[int, set[int]]:
        shard, _, values, _ = self.active.pop(token)
        new_token = self.next_token
        self.next_token += 1
        self.active[new_token] = (shard, owner, values, self.now + ttl)
        return new_token, values

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token for token, (_, _, _, deadline) in self.active.items() if deadline <= self.now
        )
        for token in expired:
            shard, _, values, _ = self.active.pop(token)
            self.free[shard].update(values)
        return expired
