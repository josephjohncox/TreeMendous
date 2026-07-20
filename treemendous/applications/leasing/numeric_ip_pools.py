"""CIDR-bounded IPv4/IPv6 leasing through integer address encoding."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import LeaseUnavailableError
from treemendous.applications.leasing._common import (
    GroupCheckpoint,
    GroupDiagnostics,
    GroupSnapshot,
    NumericLease,
    PoolGroup,
    ProcessClock,
    require_positive,
    spans_without,
)
from treemendous.domain import Span

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
AddressLease = NumericLease


class AddressUnavailableError(LeaseUnavailableError):
    """Raised when no requested contiguous address block is available."""


@dataclass(frozen=True)
class AddressPoolCheckpoint:
    """Restorable address pool state and its canonical CIDR scope."""

    network: str
    group: GroupCheckpoint


def _network(value: str | IPNetwork) -> IPNetwork:
    try:
        return ipaddress.ip_network(value, strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"network must be a canonical CIDR: {exc}") from None


def _address(value: str | IPAddress, network: IPNetwork, name: str) -> IPAddress:
    try:
        result = ipaddress.ip_address(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from None
    if result.version != network.version or result not in network:
        raise ValueError(f"{name} must be inside {network.with_prefixlen}")
    return result


def _encoded(value: IPAddress, name: str = "address") -> int:
    """Encode a validated ipaddress value and normalize conversion failures."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from None


class NumericIPAddressPool:
    """Lease contiguous addresses from exactly one canonical CIDR.

    Addresses are encoded with ``int(ipaddress.ip_address(...))`` and spans are
    half-open internally. The network address is reserved by default; the IPv4
    broadcast address is also reserved by default. Callers may add individual
    reservations. The allocator and fence validator are process-local only.
    """

    def __init__(
        self,
        network: str | IPNetwork = "192.0.2.0/24",
        *,
        clock: Clock | None = None,
        reserved: Iterable[str | IPAddress] = (),
        reserve_network: bool = True,
        reserve_broadcast: bool | None = None,
    ) -> None:
        self.network = _network(network)
        excluded: list[Span] = []
        network_number = _encoded(self.network.network_address, "network address")
        broadcast_number = _encoded(self.network.broadcast_address, "broadcast address")
        if reserve_network:
            excluded.append(Span(network_number, network_number + 1))
        effective_broadcast = (
            self.network.version == 4
            if reserve_broadcast is None
            else reserve_broadcast
        )
        if effective_broadcast:
            excluded.append(Span(broadcast_number, broadcast_number + 1))
        for value in reserved:
            reserved_address = _address(value, self.network, "reserved address")
            encoded = _encoded(reserved_address, "reserved address")
            excluded.append(Span(encoded, encoded + 1))
        full = Span(network_number, broadcast_number + 1)
        scope = self.network.with_prefixlen
        self._group = PoolGroup(
            {scope: spans_without(full, excluded)},
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: AddressPoolCheckpoint, *, clock: Clock
    ) -> NumericIPAddressPool:
        """Restore to a fresh local lineage using a compatible clock epoch."""
        if not isinstance(checkpoint, AddressPoolCheckpoint):
            raise TypeError("checkpoint must be an AddressPoolCheckpoint")
        engine = cls.__new__(cls)
        engine.network = _network(checkpoint.network)
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        scope = engine.network.with_prefixlen
        if set(engine._group.pools) != {scope}:
            raise ValueError("address checkpoint scope does not match its network")
        full = Span(
            _encoded(engine.network.network_address),
            _encoded(engine.network.broadcast_address) + 1,
        )
        if any(
            not full.contains(span) for span in engine._group.pool(scope).allowed_spans
        ):
            raise ValueError("address checkpoint domain extends outside its network")
        return engine

    def acquire(
        self,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        start_address: str | IPAddress | None = None,
        request_id: str | None = None,
    ) -> AddressLease:
        """Acquire the earliest address block or a requested exact block."""
        count = require_positive(count, "count")
        exact = None
        if start_address is not None:
            parsed_start = _address(start_address, self.network, "start_address")
            start = _encoded(parsed_start, "start_address")
            end = start + count
            broadcast = _encoded(self.network.broadcast_address, "broadcast address")
            if end > broadcast + 1:
                raise ValueError("address block extends outside the CIDR")
            exact = Span(start, end)
        try:
            return self._group.acquire(
                self.network.with_prefixlen,
                owner,
                ttl=ttl,
                size=count,
                exact_span=exact,
                request_id=request_id,
            )
        except LeaseUnavailableError as exc:
            raise AddressUnavailableError(str(exc)) from None

    def first_address(self, handle: AddressLease) -> IPAddress:
        """Decode the first integer in a leased block."""
        return ipaddress.ip_address(handle.resource.start)

    def last_address(self, handle: AddressLease) -> IPAddress:
        """Decode the last integer in a leased block."""
        return ipaddress.ip_address(handle.resource.end - 1)

    def renew(self, handle: AddressLease, *, ttl: int) -> AddressLease:
        """Renew a current address lease."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: AddressLease) -> AddressLease:
        """Release a current address lease."""
        return self._group.release(handle)

    def expire(self) -> tuple[AddressLease, ...]:
        """Materialize elapsed address leases."""
        return self._group.expire()

    def validate_fence(self, handle: AddressLease, address: str | IPAddress) -> bool:
        """Fence one address using ``(scenario, CIDR, encoded-address)``."""
        parsed_address = _address(address, self.network, "address")
        encoded = _encoded(parsed_address)
        if encoded < handle.resource.start or encoded >= handle.resource.end:
            raise ValueError("address is outside the leased block")
        key = ("numeric-ip-address-pools", self.network.with_prefixlen, encoded)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return the immutable encoded-address pool snapshot."""
        return self._group.snapshot()

    def checkpoint(self) -> AddressPoolCheckpoint:
        """Return restorable pool state without downstream fence state."""
        return AddressPoolCheckpoint(
            self.network.with_prefixlen,
            self._group.checkpoint(),
        )

    def diagnostics(self) -> GroupDiagnostics:
        """Return encoded address capacity and lifecycle counters."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> NumericIPAddressPool:
    """Create the manifest factory for numeric IP address pools."""
    return NumericIPAddressPool(**kwargs)
