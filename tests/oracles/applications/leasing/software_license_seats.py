"""Naive set/time oracle for entitled software seat checkout."""

from __future__ import annotations


class NaiveSeatOracle:
    """Reference model for one product and per-owner entitlement limits."""

    def __init__(self, capacity: int, entitlements: dict[str, int]) -> None:
        self.free = set(range(1, capacity + 1))
        self.entitlements = entitlements
        self.active: dict[int, tuple[str, set[int], int]] = {}
        self.now = 0
        self.next_token = 1

    def checkout(self, owner: str, count: int, ttl: int) -> tuple[int, set[int]]:
        owned = sum(
            len(seats)
            for active_owner, seats, _ in self.active.values()
            if active_owner == owner
        )
        if owned + count > self.entitlements.get(owner, 0):
            raise PermissionError("not entitled")
        seats = set(sorted(self.free)[:count])
        if len(seats) != count:
            raise RuntimeError("unavailable")
        self.free.difference_update(seats)
        token = self.next_token
        self.next_token += 1
        self.active[token] = (owner, seats, self.now + ttl)
        return token, seats

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token
            for token, (_, _, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            _, seats, _ = self.active.pop(token)
            self.free.update(seats)
        return expired
