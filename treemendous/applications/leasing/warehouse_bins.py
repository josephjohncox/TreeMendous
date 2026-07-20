"""Zone-scoped warehouse bin ranges with compatibility policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import (
    LeaseRequestConflictError,
    LeaseUnavailableError,
)
from treemendous.applications.leasing._common import (
    GroupCheckpoint,
    GroupDiagnostics,
    GroupSnapshot,
    NumericLease,
    PoolGroup,
    ProcessClock,
    inclusive_span,
    require_positive,
    require_string,
)
from treemendous.domain import Span, validate_coordinate


class BinCompatibilityError(ValueError):
    """Raised when a bin request is incompatible with its zone."""


class BinUnavailableError(LeaseUnavailableError):
    """Raised when a compatible contiguous bin range is unavailable."""


class BinRequestConflictError(LeaseRequestConflictError):
    """Raised when a request ID changes size or hazard metadata."""


@dataclass(frozen=True)
class BinZone:
    """One normalized zone's inclusive IDs and compatibility labels."""

    first_bin: int
    last_bin: int
    size_classes: frozenset[str] = frozenset({"standard"})
    hazards: frozenset[str] = frozenset({"general"})

    def __post_init__(self) -> None:
        inclusive_span(self.first_bin, self.last_bin, "bin zone")
        if not self.size_classes or not self.hazards:
            raise ValueError("zone size_classes and hazards must not be empty")
        for value in (*self.size_classes, *self.hazards):
            require_string(value, "compatibility label")


@dataclass(frozen=True)
class BinLease:
    """A numeric lease plus the compatibility decision used to issue it."""

    inner: NumericLease
    size_class: str
    hazard: str

    @property
    def resource(self) -> Span:
        """Return the half-open bin range."""
        return self.inner.resource

    @property
    def owner(self) -> str:
        """Return the inventory job owner."""
        return self.inner.owner

    @property
    def token(self) -> int:
        """Return the fencing token."""
        return self.inner.token

    @property
    def expires_at(self) -> int:
        """Return the lease expiry timestamp."""
        return self.inner.expires_at


@dataclass(frozen=True)
class _BinMetadata:
    scope: str
    token: int
    size_class: str
    hazard: str
    request_id: str | None


@dataclass(frozen=True)
class BinPoolCheckpoint:
    """Restorable zones, compatibility metadata, and local pool state."""

    zones: tuple[tuple[str, BinZone], ...]
    metadata: tuple[_BinMetadata, ...]
    group: GroupCheckpoint


@dataclass(frozen=True)
class BinPoolSnapshot:
    """Domain leases alongside underlying capacity snapshots."""

    leases: tuple[BinLease, ...]
    pools: GroupSnapshot


class WarehouseBinPool:
    """Lease contiguous bin IDs only from compatible warehouse zones."""

    def __init__(
        self,
        zones: Mapping[str, BinZone] | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        configured = {"A": BinZone(1, 100)} if zones is None else dict(zones)
        if not configured:
            raise ValueError("at least one bin zone is required")
        self.zones: dict[str, BinZone] = {}
        domains: dict[str, tuple[Span, ...]] = {}
        for name, zone in configured.items():
            name = require_string(name, "zone")
            if not isinstance(zone, BinZone):
                raise TypeError("zones must map names to BinZone values")
            self.zones[name] = zone
            domains[name] = (inclusive_span(zone.first_bin, zone.last_bin, "bin zone"),)
        self._metadata: dict[tuple[str, int], _BinMetadata] = {}
        self._request_metadata: dict[str, _BinMetadata] = {}
        self._group = PoolGroup(
            domains,
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: BinPoolCheckpoint, *, clock: Clock
    ) -> WarehouseBinPool:
        """Restore into new process-local zone lineages."""
        if not isinstance(checkpoint, BinPoolCheckpoint):
            raise TypeError("checkpoint must be a BinPoolCheckpoint")
        engine = cls.__new__(cls)
        engine.zones = dict(checkpoint.zones)
        engine._metadata = {
            (item.scope, item.token): item for item in checkpoint.metadata
        }
        if len(engine._metadata) != len(checkpoint.metadata):
            raise ValueError("bin metadata scope/token pairs must be unique")
        engine._request_metadata = {
            item.request_id: item
            for item in checkpoint.metadata
            if item.request_id is not None
        }
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine.zones) != set(engine._group.pools):
            raise ValueError("bin checkpoint zones do not match its pools")
        return engine

    def acquire(
        self,
        zone: str,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        size_class: str = "standard",
        hazard: str = "general",
        start_bin: int | None = None,
        request_id: str | None = None,
    ) -> BinLease:
        """Lease a compatible contiguous range in one zone."""
        try:
            policy = self.zones[zone]
        except KeyError:
            raise BinCompatibilityError(f"unknown zone: {zone!r}") from None
        size_class = require_string(size_class, "size_class")
        hazard = require_string(hazard, "hazard")
        if size_class not in policy.size_classes:
            raise BinCompatibilityError("size class is incompatible with the zone")
        if hazard not in policy.hazards:
            raise BinCompatibilityError("hazard is incompatible with the zone")
        count = require_positive(count, "count")
        exact = None
        if start_bin is not None:
            validate_coordinate(start_bin, "start_bin")
            exact = Span(start_bin, start_bin + count)
        with self._group.lock:
            if request_id is not None:
                prior = self._request_metadata.get(request_id)
                if prior is not None and (
                    prior.scope != zone
                    or prior.size_class != size_class
                    or prior.hazard != hazard
                ):
                    raise BinRequestConflictError(
                        "request ID was reused with different scope or metadata"
                    )
            try:
                inner = self._group.acquire(
                    zone,
                    owner,
                    ttl=ttl,
                    size=count,
                    exact_span=exact,
                    request_id=request_id,
                )
            except LeaseUnavailableError as exc:
                raise BinUnavailableError(str(exc)) from None
            metadata_key = (inner.scope, inner.token)
            metadata = self._metadata.get(metadata_key)
            if metadata is None:
                metadata = _BinMetadata(
                    inner.scope,
                    inner.token,
                    size_class,
                    hazard,
                    request_id,
                )
                self._metadata[metadata_key] = metadata
                if request_id is not None:
                    self._request_metadata[request_id] = metadata
            return BinLease(inner, metadata.size_class, metadata.hazard)

    def renew(self, handle: BinLease, *, ttl: int) -> BinLease:
        """Renew a current compatible bin lease."""
        renewed = self._group.renew(handle.inner, ttl=ttl)
        return BinLease(renewed, handle.size_class, handle.hazard)

    def release(self, handle: BinLease) -> BinLease:
        """Release a current bin range."""
        released = self._group.release(handle.inner)
        return BinLease(released, handle.size_class, handle.hazard)

    def expire(self) -> tuple[BinLease, ...]:
        """Materialize elapsed bin ranges with compatibility metadata."""
        return tuple(self._wrap(inner) for inner in self._group.expire())

    def _wrap(self, inner: NumericLease) -> BinLease:
        metadata = self._metadata[(inner.scope, inner.token)]
        return BinLease(inner, metadata.size_class, metadata.hazard)

    def validate_fence(self, handle: BinLease, bin_id: int) -> bool:
        """Fence one bin with a stable zone/bin key."""
        validate_coordinate(bin_id, "bin_id")
        if bin_id < handle.resource.start or bin_id >= handle.resource.end:
            raise ValueError("bin_id is outside the leased range")
        key = ("warehouse-bin-ranges", handle.inner.scope, bin_id)
        return self._group.validate_fence(key, handle.inner)

    def snapshot(self) -> BinPoolSnapshot:
        """Return domain leases and immutable capacity state."""
        with self._group.lock:
            group = self._group.snapshot()
            inners = (
                NumericLease(scope, lease)
                for scope, pool in group.pools
                for lease in pool.leases
            )
            return BinPoolSnapshot(tuple(self._wrap(inner) for inner in inners), group)

    def checkpoint(self) -> BinPoolCheckpoint:
        """Return complete local bin state, excluding downstream fence marks."""
        with self._group.lock:
            self._group.expire()
            return BinPoolCheckpoint(
                tuple(sorted(self.zones.items())),
                tuple(self._metadata[key] for key in sorted(self._metadata)),
                self._group.checkpoint(),
            )

    def diagnostics(self) -> GroupDiagnostics:
        """Return lifecycle and capacity counters by zone."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> WarehouseBinPool:
    """Create the manifest factory for warehouse bin ranges."""
    return WarehouseBinPool(**kwargs)
