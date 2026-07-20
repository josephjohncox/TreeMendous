"""Naive set/time oracle for compatible warehouse bin zones."""

from __future__ import annotations


class NaiveBinOracle:
    """Reference model using sets and explicit zone compatibility labels."""

    def __init__(
        self,
        zones: dict[str, tuple[set[int], set[str], set[str]]],
    ) -> None:
        self.zones = zones
        self.active: dict[int, tuple[str, set[int], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(
        self, zone: str, count: int, size: str, hazard: str, ttl: int
    ) -> tuple[int, set[int]]:
        free, sizes, hazards = self.zones[zone]
        if size not in sizes or hazard not in hazards:
            raise ValueError("incompatible")
        values = next(
            (
                set(range(start, start + count))
                for start in sorted(free)
                if set(range(start, start + count)) <= free
            ),
            None,
        )
        if values is None:
            raise RuntimeError("unavailable")
        free.difference_update(values)
        token = self.next_token
        self.next_token += 1
        self.active[token] = (zone, values, self.now + ttl)
        return token, values

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token
            for token, (_, _, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            zone, values, _ = self.active.pop(token)
            self.zones[zone][0].update(values)
        return expired
