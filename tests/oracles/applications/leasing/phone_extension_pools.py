"""Naive set/time oracle for a reserved phone numbering plan."""

from __future__ import annotations


class NaiveExtensionOracle:
    """Reference model using a set after emergency/service exclusions."""

    def __init__(self, first: int, last: int, reserved: set[int]) -> None:
        self.free = set(range(first, last + 1)).difference(reserved)
        self.active: dict[int, tuple[set[int], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(self, count: int, ttl: int) -> tuple[int, set[int]]:
        values = next(
            (
                set(range(start, start + count))
                for start in sorted(self.free)
                if set(range(start, start + count)) <= self.free
            ),
            None,
        )
        if values is None:
            raise RuntimeError("unavailable")
        self.free.difference_update(values)
        token = self.next_token
        self.next_token += 1
        self.active[token] = (values, self.now + ttl)
        return token, values

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token for token, (_, deadline) in self.active.items() if deadline <= self.now
        )
        for token in expired:
            values, _ = self.active.pop(token)
            self.free.update(values)
        return expired
