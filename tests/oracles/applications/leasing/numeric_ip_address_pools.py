"""Naive independent integer/set oracle for one CIDR address pool."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _encode(address: IPAddress) -> int:
    try:
        return int(address)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid oracle address: {exc}") from None


class NaiveAddressOracle:
    """Reference model using the full integer CIDR minus reservation policy."""

    def __init__(
        self,
        cidr: str,
        *,
        reserved: Iterable[str | IPAddress] = (),
        reserve_network: bool = True,
        reserve_broadcast: bool | None = None,
    ) -> None:
        try:
            network = ipaddress.ip_network(cidr, strict=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid oracle CIDR: {exc}") from None
        self.network = network
        first = _encode(network.network_address)
        last = _encode(network.broadcast_address)
        excluded: set[int] = set()
        if reserve_network:
            excluded.add(first)
        effective_broadcast = (
            network.version == 4 if reserve_broadcast is None else reserve_broadcast
        )
        if effective_broadcast:
            excluded.add(last)
        for value in reserved:
            try:
                address = ipaddress.ip_address(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid reserved address: {exc}") from None
            if address.version != network.version or address not in network:
                raise ValueError(f"reserved address must be inside {network}")
            excluded.add(_encode(address))
        self.free = set(range(first, last + 1)).difference(excluded)
        self.active: dict[int, tuple[tuple[int, ...], int]] = {}
        self.now = 0
        self.next_token = 1

    def acquire(
        self,
        count: int,
        ttl: int,
        *,
        start_address: str | IPAddress | None = None,
    ) -> tuple[int, tuple[int, ...]]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        if start_address is None:
            starts = tuple(sorted(self.free))
        else:
            try:
                address = ipaddress.ip_address(start_address)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid start_address: {exc}") from None
            if address.version != self.network.version or address not in self.network:
                raise ValueError(f"start_address must be inside {self.network}")
            start = _encode(address)
            network_end = _encode(self.network.broadcast_address) + 1
            if start + count > network_end:
                raise ValueError("address block extends outside the CIDR")
            starts = (start,)
        for start in starts:
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
            token
            for token, (_, deadline) in self.active.items()
            if deadline <= self.now
        )
        for token in expired:
            block, _ = self.active.pop(token)
            self.free.update(block)
        return expired
