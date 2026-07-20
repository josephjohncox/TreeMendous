"""Naive set/time oracle for network-scoped VLAN tags."""

from __future__ import annotations


class NaiveVlanOracle:
    """Reference model with one explicit free set per network scope."""

    def __init__(self, scopes: tuple[str, ...], reserved: set[int]) -> None:
        allowed = set(range(1, 4095)).difference(reserved)
        self.free = {scope: set(allowed) for scope in scopes}
        self.active: dict[int, tuple[str, set[int], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(self, scope: str, count: int, ttl: int) -> tuple[int, set[int]]:
        values = next(
            (
                set(range(start, start + count))
                for start in sorted(self.free[scope])
                if set(range(start, start + count)) <= self.free[scope]
            ),
            None,
        )
        if values is None:
            raise RuntimeError("unavailable")
        self.free[scope].difference_update(values)
        token = self.next_token
        self.next_token += 1
        self.active[token] = (scope, values, self.now + ttl)
        return token, values

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token
            for token, (_, _, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            scope, values, _ = self.active.pop(token)
            self.free[scope].update(values)
        return expired
