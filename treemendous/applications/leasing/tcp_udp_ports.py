"""TCP/UDP contiguous port leasing with explicit exclusion policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
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
    inclusive_span,
    require_positive,
    spans_without,
)
from treemendous.domain import Span, validate_coordinate


class PortProtocol(Enum):
    """Transport namespace used to scope otherwise identical port numbers."""

    TCP = "tcp"
    UDP = "udp"


class PortUnavailableError(LeaseUnavailableError):
    """Raised when policy or active leases leave no requested port block."""


PortLease = NumericLease


@dataclass(frozen=True)
class PortPoolCheckpoint:
    """Restorable process-local TCP/UDP pool state."""

    group: GroupCheckpoint


def _protocol(value: PortProtocol | str) -> PortProtocol:
    if isinstance(value, PortProtocol):
        return value
    try:
        return PortProtocol(value)
    except (TypeError, ValueError):
        raise ValueError("protocol must be 'tcp' or 'udp'") from None


def _reserved_spans(ranges: Iterable[tuple[int, int]]) -> tuple[Span, ...]:
    return tuple(inclusive_span(start, end, "reserved port range") for start, end in ranges)


class PortLeaseEngine:
    """Lease contiguous transport ports inside one process.

    TCP and UDP are independent namespaces. Ports outside 1..65535, the
    configured privileged/system interval, the ephemeral interval, and
    protocol-specific reservations are never made available. Fencing uses the
    stable key ``(scenario, protocol, port)``; it is only an in-process example
    unless the downstream service durably stores the same high-water marks.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        system_ports: tuple[int, int] | None = (1, 1023),
        ephemeral_ports: tuple[int, int] | None = (49152, 65535),
        protocol_reserved: Mapping[
            PortProtocol | str, Iterable[tuple[int, int]]
        ] | None = None,
    ) -> None:
        common = []
        if system_ports is not None:
            common.append(inclusive_span(*system_ports, "system port range"))
        if ephemeral_ports is not None:
            common.append(inclusive_span(*ephemeral_ports, "ephemeral port range"))
        by_protocol: dict[PortProtocol, list[Span]] = {
            PortProtocol.TCP: list(common),
            PortProtocol.UDP: list(common),
        }
        if protocol_reserved is not None:
            for raw_protocol, ranges in protocol_reserved.items():
                by_protocol[_protocol(raw_protocol)].extend(_reserved_spans(ranges))
        domain = inclusive_span(1, 65535, "port domain")
        self._group = PoolGroup(
            {
                protocol.value: spans_without(domain, excluded)
                for protocol, excluded in by_protocol.items()
            },
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: PortPoolCheckpoint, *, clock: Clock
    ) -> PortLeaseEngine:
        """Restore into fresh pool lineages; exclusive takeover is external."""
        if not isinstance(checkpoint, PortPoolCheckpoint):
            raise TypeError("checkpoint must be a PortPoolCheckpoint")
        engine = cls.__new__(cls)
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine._group.pools) != {"tcp", "udp"}:
            raise ValueError("port checkpoint must contain tcp and udp pools")
        return engine

    def acquire(
        self,
        protocol: PortProtocol | str,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        start_port: int | None = None,
        request_id: str | None = None,
    ) -> PortLease:
        """Acquire the earliest contiguous allowed block or one exact block."""
        selected = _protocol(protocol)
        count = require_positive(count, "count")
        exact = None
        if start_port is not None:
            validate_coordinate(start_port, "start_port")
            exact = Span(start_port, start_port + count)
        try:
            return self._group.acquire(
                selected.value,
                owner,
                ttl=ttl,
                size=count,
                exact_span=exact,
                request_id=request_id,
            )
        except LeaseUnavailableError as exc:
            raise PortUnavailableError(str(exc)) from None

    def renew(self, handle: PortLease, *, ttl: int) -> PortLease:
        """Renew a current port lease."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: PortLease) -> PortLease:
        """Release a current port lease."""
        return self._group.release(handle)

    def expire(self) -> tuple[PortLease, ...]:
        """Materialize all elapsed TCP and UDP leases."""
        return self._group.expire()

    def validate_fence(self, handle: PortLease, port: int) -> bool:
        """Validate one port write using its stable protocol/number key."""
        validate_coordinate(port, "port")
        if port < handle.resource.start or port >= handle.resource.end:
            raise ValueError("port is outside the leased block")
        key = ("tcp-udp-port-leases", handle.scope, port)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return immutable per-protocol pool snapshots."""
        return self._group.snapshot()

    def checkpoint(self) -> PortPoolCheckpoint:
        """Return pool state; downstream fence high-water marks are external."""
        return PortPoolCheckpoint(self._group.checkpoint())

    def diagnostics(self) -> GroupDiagnostics:
        """Return capacity and lifecycle counters by protocol."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> PortLeaseEngine:
    """Create the manifest factory for TCP/UDP port leasing."""
    return PortLeaseEngine(**kwargs)
