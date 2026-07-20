"""Phone extension leases constrained by an explicit numbering plan."""

from __future__ import annotations

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
    inclusive_span,
    require_positive,
    require_string,
    spans_without,
)
from treemendous.domain import Span, validate_coordinate

ExtensionLease = NumericLease


class ExtensionUnavailableError(LeaseUnavailableError):
    """Raised when no allowed contiguous extension block is available."""


@dataclass(frozen=True)
class ExtensionPoolCheckpoint:
    """Restorable numbering-plan pool lineage."""

    plan_id: str
    group: GroupCheckpoint


class PhoneExtensionPool:
    """Lease extensions after removing emergency and service numbers.

    The configured bounds are inclusive. Emergency numbers are individual
    reservations and service ranges are inclusive intervals. Fencing uses the
    stable plan/extension key rather than a request-specific block key.
    """

    def __init__(
        self,
        plan_id: str = "default",
        *,
        first_extension: int = 1000,
        last_extension: int = 9999,
        emergency_numbers: Iterable[int] = (911,),
        service_ranges: Iterable[tuple[int, int]] = (),
        clock: Clock | None = None,
    ) -> None:
        self.plan_id = require_string(plan_id, "plan_id")
        domain = inclusive_span(
            first_extension,
            last_extension,
            "numbering plan",
        )
        excluded: list[Span] = []
        for number in emergency_numbers:
            validate_coordinate(number, "emergency number")
            excluded.append(Span(number, number + 1))
        excluded.extend(
            inclusive_span(start, end, "service range")
            for start, end in service_ranges
        )
        self._group = PoolGroup(
            {self.plan_id: spans_without(domain, excluded)},
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: ExtensionPoolCheckpoint, *, clock: Clock
    ) -> PhoneExtensionPool:
        """Restore into a fresh local numbering-plan lineage."""
        if not isinstance(checkpoint, ExtensionPoolCheckpoint):
            raise TypeError("checkpoint must be an ExtensionPoolCheckpoint")
        engine = cls.__new__(cls)
        engine.plan_id = require_string(checkpoint.plan_id, "plan_id")
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine._group.pools) != {engine.plan_id}:
            raise ValueError("extension checkpoint plan does not match its pool")
        return engine

    def acquire(
        self,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        start_extension: int | None = None,
        request_id: str | None = None,
    ) -> ExtensionLease:
        """Acquire the earliest extension block or one exact allowed block."""
        count = require_positive(count, "count")
        exact = None
        if start_extension is not None:
            validate_coordinate(start_extension, "start_extension")
            exact = Span(start_extension, start_extension + count)
        try:
            return self._group.acquire(
                self.plan_id,
                owner,
                ttl=ttl,
                size=count,
                exact_span=exact,
                request_id=request_id,
            )
        except LeaseUnavailableError as exc:
            raise ExtensionUnavailableError(str(exc)) from None

    def renew(self, handle: ExtensionLease, *, ttl: int) -> ExtensionLease:
        """Renew a current extension assignment."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: ExtensionLease) -> ExtensionLease:
        """Release a current extension assignment."""
        return self._group.release(handle)

    def expire(self) -> tuple[ExtensionLease, ...]:
        """Materialize elapsed extension assignments."""
        return self._group.expire()

    def validate_fence(self, handle: ExtensionLease, extension: int) -> bool:
        """Fence one extension using a stable plan/extension key."""
        validate_coordinate(extension, "extension")
        if extension < handle.resource.start or extension >= handle.resource.end:
            raise ValueError("extension is outside the leased block")
        key = ("phone-extension-pools", self.plan_id, extension)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return the immutable numbering-plan pool snapshot."""
        return self._group.snapshot()

    def checkpoint(self) -> ExtensionPoolCheckpoint:
        """Return pool state without process-local downstream fence marks."""
        return ExtensionPoolCheckpoint(self.plan_id, self._group.checkpoint())

    def diagnostics(self) -> GroupDiagnostics:
        """Return numbering-plan capacity and lifecycle counters."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> PhoneExtensionPool:
    """Create the manifest factory for phone extension pools."""
    return PhoneExtensionPool(**kwargs)
