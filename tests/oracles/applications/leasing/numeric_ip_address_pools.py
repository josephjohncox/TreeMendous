"""Naive set/time oracle for one CIDR address pool."""

from __future__ import annotations

import ipaddress


def _encode(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> int:
    try:
        return int(address)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid oracle address: {exc}") from None


class NaiveAddressOracle:
    """Reference model using integer-encoded addresses and Python sets."""

    def __init__(self, cidr: str) -> None:
        network = ipaddress.ip_network(cidr)
        self.free = {_encode(address) for address in network.hosts()}
        self.active: dict[int, tuple[tuple[int, ...], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(self, count: int, ttl: int) -> tuple[int, tuple[int, ...]]:
        for start in sorted(self.free):
            block = tuple(range(start, start + count))
            if all(value in self.free for value in block):
                self.free.difference_update(block)
                token = self.next_token
                self.next_token += 1
                self.active[token] = (block, self.now + ttl)
                return token, block
        raise RuntimeError("unavailable")

    def advance(self, delta: int) -> tuple[int, ...]:
        self.now += delta
        expired = tuple(
            token for token, (_, deadline) in self.active.items() if deadline <= self.now
        )
        for token in expired:
            block, _ = self.active.pop(token)
            self.free.update(block)
        return expired
