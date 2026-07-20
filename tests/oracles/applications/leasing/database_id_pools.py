"""Naive cursor/set/time oracle for database ID batches."""

from __future__ import annotations


class NaiveDatabaseIdOracle:
    """Reference model separating temporary, reusable, and committed IDs."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.cursor = 1
        self.now = 0
        self.next_token = 1
        self.active: dict[int, tuple[set[int], int]] = {}
        self.reusable: set[int] = set()
        self.committed: set[int] = set()

    def acquire(
        self, count: int, ttl: int, *, reusable: bool = False
    ) -> tuple[int, set[int]]:
        if reusable:
            ordered = sorted(self.reusable)
            values = next(
                (
                    set(range(start, start + count))
                    for start in ordered
                    if set(range(start, start + count)) <= self.reusable
                ),
                None,
            )
            if values is None:
                raise RuntimeError("unavailable")
            self.reusable.difference_update(values)
        else:
            values = set(range(self.cursor, self.cursor + count))
            if not values or max(values) > self.maximum:
                raise RuntimeError("unavailable")
            self.cursor += count
        token = self.next_token
        self.next_token += 1
        self.active[token] = (values, self.now + ttl)
        return token, values

    def commit(self, token: int) -> None:
        values, _ = self.active.pop(token)
        self.committed.update(values)

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token
            for token, (_, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            values, _ = self.active.pop(token)
            self.reusable.update(values)
        return expired
