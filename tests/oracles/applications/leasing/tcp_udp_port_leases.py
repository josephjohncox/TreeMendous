"""Naive set/time oracle for TCP and UDP port leasing."""

from __future__ import annotations


class NaivePortOracle:
    """Deliberately simple per-port reference model."""

    def __init__(self) -> None:
        allowed = set(range(1024, 49152))
        self.free = {"tcp": set(allowed), "udp": set(allowed)}
        self.active: dict[int, tuple[str, tuple[int, ...], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(
        self, protocol: str, count: int, ttl: int
    ) -> tuple[int, tuple[int, ...]]:
        for start in sorted(self.free[protocol]):
            block = tuple(range(start, start + count))
            if all(port in self.free[protocol] for port in block):
                self.free[protocol].difference_update(block)
                token = self.next_token
                self.next_token += 1
                self.active[token] = (protocol, block, self.now + ttl)
                return token, block
        raise RuntimeError("unavailable")

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token
            for token, (_, _, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            protocol, block, _ = self.active.pop(token)
            self.free[protocol].update(block)
        return expired
