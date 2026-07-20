"""Product-scoped software seat checkout and renewal leases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import LeaseState, LeaseUnavailableError
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

SeatLease = NumericLease


class UnknownProductError(ValueError):
    """Raised when a checkout names an unconfigured product."""


class EntitlementError(PermissionError):
    """Raised when an owner is not entitled to the requested seat count."""


class SeatUnavailableError(LeaseUnavailableError):
    """Raised when entitled product capacity is currently exhausted."""


@dataclass(frozen=True)
class SeatPoolCheckpoint:
    """Restorable product pools and entitlement policy."""

    products: tuple[tuple[str, int], ...]
    entitlement_restricted: bool
    entitlements: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    group: GroupCheckpoint


class SoftwareSeatPool:
    """Lease numbered seats independently for each software product.

    An entitlement value is the maximum concurrently active seats for an owner
    and product. Omitted entitlement policy allows any owner up to product
    capacity. The state and sample fence validator are process-local; a real
    license server must persist checkout state and fencing high-water marks.
    """

    def __init__(
        self,
        products: Mapping[str, int] | None = None,
        *,
        entitlements: Mapping[str, Mapping[str, int]] | None = None,
        clock: Clock | None = None,
    ) -> None:
        configured = {"default": 10} if products is None else dict(products)
        if not configured:
            raise ValueError("at least one product is required")
        self.products: dict[str, int] = {}
        domains: dict[str, tuple[Span, ...]] = {}
        for product, capacity in configured.items():
            product = require_string(product, "product")
            capacity = require_positive(capacity, "product capacity")
            self.products[product] = capacity
            domains[product] = (inclusive_span(1, capacity, "seat domain"),)
        self.entitlements: dict[str, dict[str, int]] | None = None
        if entitlements is not None:
            self.entitlements = {}
            for owner, grants in entitlements.items():
                owner = require_string(owner, "entitlement owner")
                normalized: dict[str, int] = {}
                for product, limit in grants.items():
                    if product not in self.products:
                        raise UnknownProductError(product)
                    normalized[product] = require_positive(limit, "entitlement limit")
                self.entitlements[owner] = normalized
        self._group = PoolGroup(
            domains,
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: SeatPoolCheckpoint, *, clock: Clock
    ) -> SoftwareSeatPool:
        """Restore into fresh local product-pool lineages."""
        if not isinstance(checkpoint, SeatPoolCheckpoint):
            raise TypeError("checkpoint must be a SeatPoolCheckpoint")
        engine = cls.__new__(cls)
        engine.products = dict(checkpoint.products)
        engine.entitlements = (
            {
                owner: dict(grants)
                for owner, grants in checkpoint.entitlements
            }
            if checkpoint.entitlement_restricted
            else None
        )
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine.products) != set(engine._group.pools):
            raise ValueError("seat checkpoint products do not match its pools")
        return engine

    def _limit(self, owner: str, product: str) -> int:
        if self.entitlements is None:
            return self.products[product]
        try:
            return self.entitlements[owner][product]
        except KeyError:
            raise EntitlementError(
                f"owner {owner!r} is not entitled to product {product!r}"
            ) from None

    def checkout(
        self,
        product: str,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        request_id: str | None = None,
    ) -> SeatLease:
        """Checkout the earliest contiguous seat IDs for one product."""
        if product not in self.products:
            raise UnknownProductError(product)
        count = require_positive(count, "count")
        with self._group.lock:
            snapshot = self._group.pool(product).snapshot()
            is_retry = request_id is not None and any(
                lease.request_id == request_id for lease in snapshot.leases
            )
            if not is_retry:
                active = sum(
                    lease.resource.length
                    for lease in snapshot.leases
                    if lease.owner == owner and lease.state is LeaseState.ACTIVE
                )
                if active + count > self._limit(owner, product):
                    raise EntitlementError(
                        "checkout exceeds the concurrent entitlement"
                    )
            try:
                return self._group.acquire(
                    product,
                    owner,
                    ttl=ttl,
                    size=count,
                    request_id=request_id,
                )
            except LeaseUnavailableError as exc:
                raise SeatUnavailableError(str(exc)) from None

    acquire = checkout

    def renew(self, handle: SeatLease, *, ttl: int) -> SeatLease:
        """Renew a current checkout session."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: SeatLease) -> SeatLease:
        """Return checked-out seats immediately."""
        return self._group.release(handle)

    def expire(self) -> tuple[SeatLease, ...]:
        """Materialize elapsed checkout sessions."""
        return self._group.expire()

    def validate_fence(self, handle: SeatLease, seat_id: int) -> bool:
        """Fence one seat with a stable product/seat key."""
        validate_coordinate(seat_id, "seat_id")
        if seat_id < handle.resource.start or seat_id >= handle.resource.end:
            raise ValueError("seat_id is outside the checkout")
        key = ("software-license-seats", handle.scope, seat_id)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return immutable product pool snapshots."""
        return self._group.snapshot()

    def checkpoint(self) -> SeatPoolCheckpoint:
        """Return product policy and lease state, excluding fence state."""
        grants: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = ()
        if self.entitlements is not None:
            grants = tuple(
                (owner, tuple(sorted(products.items())))
                for owner, products in sorted(self.entitlements.items())
            )
        return SeatPoolCheckpoint(
            tuple(sorted(self.products.items())),
            self.entitlements is not None,
            grants,
            self._group.checkpoint(),
        )

    def diagnostics(self) -> GroupDiagnostics:
        """Return capacity and lifecycle counters per product."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> SoftwareSeatPool:
    """Create the manifest factory for software seat pools."""
    return SoftwareSeatPool(**kwargs)
