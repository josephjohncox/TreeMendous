"""Thread-safe canonical range-set facade."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from threading import RLock
from typing import Any

from treemendous.backends.adapters import BackendAdapter
from treemendous.backends.types import Capability
from treemendous.domain import (
    AvailabilityStats,
    IntervalResult,
    ManagedDomain,
    ManagedDomainRequiredError,
    MutationResult,
    RangeSnapshot,
    Span,
    UnsupportedCapabilityError,
    validate_coordinate,
    validate_length,
)
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    PayloadPolicy,
    UniformPayloadPolicy,
)

_MISSING = object()


@dataclass(frozen=True)
class _OrderedEvent:
    span: Span
    data: Any
    order_key: tuple[Any, ...]


def _subtract_geometry(
    sources: Iterable[IntervalResult], blockers: Iterable[IntervalResult]
) -> tuple[Span, ...]:
    """Return normalized geometry in *sources* that is absent from *blockers*."""
    blocked = tuple(blockers)
    changed: list[Span] = []
    for source in sources:
        cursor = source.start
        for blocker in blocked:
            if blocker.end <= cursor:
                continue
            if blocker.start >= source.end:
                break
            if blocker.start > cursor:
                changed.append(Span(cursor, min(blocker.start, source.end)))
            cursor = max(cursor, blocker.end)
            if cursor >= source.end:
                break
        if cursor < source.end:
            changed.append(Span(cursor, source.end))
    return tuple(changed)


class RangeSet:
    """A normalized free-range set backed by an explicit adapter.

    Payload-capable instances keep pointwise payload segmentation at the facade
    boundary while the backend stores geometry only. This prevents legacy tree
    merge hooks from smearing a value across an overlap or silently dropping it.
    """

    def __init__(
        self,
        adapter: BackendAdapter,
        *,
        domain: ManagedDomain | Span | tuple[int, int] | None = None,
        capabilities: frozenset[Capability] = frozenset({Capability.CORE}),
        initially_available: bool = True,
        payload_policy: PayloadPolicy[Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._domain = (
            domain
            if isinstance(domain, ManagedDomain)
            else (ManagedDomain(domain) if domain is not None else None)
        )
        self._capabilities = capabilities
        self._lock = RLock()
        payloads = Capability.PAYLOADS in capabilities
        if payloads and not adapter.supports_payloads:
            raise UnsupportedCapabilityError(
                "adapter cannot preserve the catalogued payload capability"
            )
        if payload_policy is not None and not payloads:
            raise UnsupportedCapabilityError(
                "backend does not support payload policies"
            )
        self._payload_policy = payload_policy or (
            UniformPayloadPolicy() if payloads else None
        )
        self._payload_segments: list[IntervalResult] | None = [] if payloads else None
        self._ordered_events: list[_OrderedEvent] | None = (
            [] if isinstance(self._payload_policy, OrderedPayloadPolicy) else None
        )

        if initially_available and self._domain is not None:
            initial_data = self._payload_identity()
            for span in self._domain.spans:
                self._adapter.release(span.start, span.end)
                if self._ordered_events is not None:
                    self._ordered_events.append(
                        self._ordered_event(span, deepcopy(initial_data))
                    )
                elif self._payload_segments is not None:
                    self._payload_segments.append(
                        IntervalResult(span.start, span.end, data=initial_data)
                    )
            if self._ordered_events is not None:
                self._payload_segments = self._fold_ordered_events(self._ordered_events)

    @property
    def domain(self) -> ManagedDomain | None:
        return self._domain

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    @property
    def payload_policy(self) -> PayloadPolicy[Any] | None:
        return self._payload_policy

    def _payload_identity(self) -> Any:
        if isinstance(self._payload_policy, JoinPayloadPolicy):
            return self._payload_policy.bottom
        if isinstance(self._payload_policy, OrderedPayloadPolicy):
            return self._payload_policy.identity
        return None

    def _validate_domain(self, span: Span) -> None:
        if self._domain is not None and not self._domain.contains(span):
            raise ValueError("span must be contained in the managed domain")

    def _geometry_intervals(self) -> tuple[IntervalResult, ...]:
        normalized: list[IntervalResult] = []
        for item in sorted(
            self._adapter.intervals(),
            key=lambda interval: (interval.start, interval.end),
        ):
            if normalized and item.start <= normalized[-1].end:
                previous = normalized[-1]
                normalized[-1] = IntervalResult(
                    previous.start, max(previous.end, item.end)
                )
            else:
                normalized.append(IntervalResult(item.start, item.end))
        return tuple(normalized)

    def intervals(self) -> tuple[IntervalResult, ...]:
        with self._lock:
            if self._payload_segments is not None:
                return tuple(self._payload_segments)
            return self._geometry_intervals()

    def _covered(
        self, span: Span, intervals: tuple[IntervalResult, ...] | None = None
    ) -> bool:
        cursor = span.start
        for interval in (
            intervals if intervals is not None else self._geometry_intervals()
        ):
            if interval.end <= cursor:
                continue
            if interval.start > cursor:
                return False
            cursor = max(cursor, interval.end)
            if cursor >= span.end:
                return True
        return False

    def _merge_payload_segments(
        self, segments: list[IntervalResult]
    ) -> list[IntervalResult]:
        policy = self._payload_policy
        assert policy is not None
        merged: list[IntervalResult] = []
        for segment in segments:
            if (
                merged
                and merged[-1].end == segment.start
                and policy.can_merge(merged[-1].data, segment.data)
                and merged[-1].data == segment.data
            ):
                previous = merged[-1]
                merged[-1] = IntervalResult(
                    previous.start, segment.end, data=previous.data
                )
            else:
                merged.append(segment)
        return merged

    def _ordered_event(self, span: Span, payload: Any) -> _OrderedEvent:
        policy = self._payload_policy
        assert isinstance(policy, OrderedPayloadPolicy)
        return _OrderedEvent(
            span,
            payload,
            (span.start, span.end, policy.event_key(payload)),
        )

    def _fold_ordered_events(self, events: list[_OrderedEvent]) -> list[IntervalResult]:
        policy = self._payload_policy
        assert isinstance(policy, OrderedPayloadPolicy)
        endpoints = sorted(
            {
                endpoint
                for event in events
                for endpoint in (event.span.start, event.span.end)
            }
        )
        result: list[IntervalResult] = []
        for start, end in pairwise(endpoints):
            target = Span(start, end)
            active = sorted(
                (event for event in events if event.span.contains(target)),
                key=lambda event: event.order_key,
            )
            if not active:
                continue
            data = deepcopy(policy.identity)
            for event in active:
                restricted = policy.restrict(deepcopy(event.data), event.span, target)
                data = policy.combine(deepcopy(data), restricted)
            result.append(IntervalResult(start, end, data=data))
        return self._merge_payload_segments(result)

    def _payload_after_add(
        self, segments: list[IntervalResult], span: Span, payload: Any
    ) -> list[IntervalResult]:
        policy = self._payload_policy
        assert policy is not None
        endpoints = {span.start, span.end}
        for segment in segments:
            endpoints.add(segment.start)
            endpoints.add(segment.end)
        ordered = sorted(endpoints)
        result: list[IntervalResult] = []
        for start, end in pairwise(ordered):
            old = next(
                (
                    segment
                    for segment in segments
                    if segment.start <= start and end <= segment.end
                ),
                None,
            )
            in_new = span.start <= start and end <= span.end
            if old is None and not in_new:
                continue
            target = Span(start, end)
            if old is not None and in_new:
                old_data = policy.restrict(old.data, old.span, target)
                new_data = policy.restrict(payload, span, target)
                data = policy.combine(old_data, new_data)
            elif old is not None:
                data = policy.restrict(old.data, old.span, target)
            else:
                data = policy.restrict(payload, span, target)
            result.append(IntervalResult(start, end, data=data))
        return self._merge_payload_segments(result)

    def _payload_after_discard(
        self, segments: list[IntervalResult], span: Span
    ) -> list[IntervalResult]:
        policy = self._payload_policy
        assert policy is not None
        result: list[IntervalResult] = []
        for segment in segments:
            if segment.end <= span.start or segment.start >= span.end:
                result.append(segment)
                continue
            if segment.start < span.start:
                target = Span(segment.start, span.start)
                result.append(
                    IntervalResult(
                        target.start,
                        target.end,
                        data=policy.restrict(segment.data, segment.span, target),
                    )
                )
            if segment.end > span.end:
                target = Span(span.end, segment.end)
                result.append(
                    IntervalResult(
                        target.start,
                        target.end,
                        data=policy.restrict(segment.data, segment.span, target),
                    )
                )
        return self._merge_payload_segments(result)

    def add(self, span: Span, payload: Any = _MISSING) -> MutationResult:
        self._validate_domain(span)
        with self._lock:
            if (
                payload is not _MISSING
                and Capability.PAYLOADS not in self._capabilities
            ):
                raise UnsupportedCapabilityError("backend does not support payloads")
            before = self._geometry_intervals()
            covered = self._covered(span, before)
            prospective_payloads = None
            prospective_events = None
            if self._payload_segments is not None:
                actual_payload = deepcopy(
                    self._payload_identity() if payload is _MISSING else payload
                )
                # Deep copies isolate committed mutable values from arbitrary
                # user callbacks until all policy work has succeeded.
                if self._ordered_events is not None:
                    prospective_events = deepcopy(self._ordered_events)
                    prospective_events.append(self._ordered_event(span, actual_payload))
                    prospective_payloads = self._fold_ordered_events(prospective_events)
                else:
                    prospective_payloads = self._payload_after_add(
                        deepcopy(self._payload_segments), span, actual_payload
                    )
            self._adapter.release(span.start, span.end)
            after = self._geometry_intervals()
            if prospective_payloads is not None:
                self._payload_segments = prospective_payloads
            if prospective_events is not None:
                self._ordered_events = prospective_events
            changed = _subtract_geometry(after, before)
            changed_length = sum(part.length for part in changed)
            return MutationResult(changed, changed_length, covered)

    def discard(self, span: Span, *, require_covered: bool = False) -> MutationResult:
        self._validate_domain(span)
        with self._lock:
            before = self._geometry_intervals()
            covered = self._covered(span, before)
            if require_covered and not covered:
                return MutationResult((), 0, False)
            prospective_events = None
            if self._ordered_events is not None:
                prospective_events = []
                for event in deepcopy(self._ordered_events):
                    if not event.span.overlaps(span):
                        prospective_events.append(event)
                        continue
                    for start, end in (
                        (event.span.start, min(event.span.end, span.start)),
                        (max(event.span.start, span.end), event.span.end),
                    ):
                        if start < end:
                            target = Span(start, end)
                            policy = self._payload_policy
                            assert isinstance(policy, OrderedPayloadPolicy)
                            prospective_events.append(
                                _OrderedEvent(
                                    target,
                                    policy.restrict(
                                        deepcopy(event.data), event.span, target
                                    ),
                                    event.order_key,
                                )
                            )
                prospective_payloads = self._fold_ordered_events(prospective_events)
            elif self._payload_segments is not None:
                prospective_payloads = self._payload_after_discard(
                    deepcopy(self._payload_segments), span
                )
            else:
                prospective_payloads = None
            self._adapter.reserve(span.start, span.end)
            after = self._geometry_intervals()
            if prospective_payloads is not None:
                self._payload_segments = prospective_payloads
            if prospective_events is not None:
                self._ordered_events = prospective_events
            changed = _subtract_geometry(before, after)
            changed_length = sum(part.length for part in changed)
            return MutationResult(changed, changed_length, covered)

    def first_fit(
        self,
        length: int,
        *,
        not_before: int,
        not_after: int | None = None,
        payload_predicate: Callable[[Any], bool] | None = None,
    ) -> IntervalResult | None:
        """Return the earliest fit, preserving payload information.

        A fit crossing heterogeneous payload segments returns their values as a
        coordinate-ordered tuple; a homogeneous fit returns the single value.
        The predicate, when supplied, must accept every crossed segment.
        """
        validate_coordinate(not_before, "not_before")
        validate_length(length)
        if not_after is not None:
            validate_coordinate(not_after, "not_after")
            if not_after <= not_before:
                raise ValueError("not_after must be greater than not_before")
        if (
            payload_predicate is not None
            and Capability.PAYLOADS not in self._capabilities
        ):
            raise UnsupportedCapabilityError(
                "backend does not support payload predicates"
            )
        with self._lock:
            if self._payload_segments is None:
                for interval in self._geometry_intervals():
                    start = max(not_before, interval.start)
                    end = start + length
                    if end <= interval.end and (not_after is None or end <= not_after):
                        return IntervalResult(start, end)
                return None

            # Payload queries scan runs of adjacent accepted segments.  A result
            # crossing equal payloads returns that value; a heterogeneous result
            # returns the tuple of crossed segment values in coordinate order.
            run_start: int | None = None
            run_end: int | None = None
            run_parts: list[tuple[int, int, Any]] = []
            for interval in self._payload_segments:
                start = max(not_before, interval.start)
                end = (
                    interval.end if not_after is None else min(interval.end, not_after)
                )
                accepted = end > start and (
                    payload_predicate is None or payload_predicate(interval.data)
                )
                if not accepted:
                    run_start = run_end = None
                    run_parts = []
                    continue
                if run_end != start:
                    run_start = start
                    run_parts = []
                run_end = end
                run_parts.append((start, end, interval.data))
                assert run_start is not None
                result_end = run_start + length
                if result_end <= run_end:
                    values = [
                        data
                        for part_start, part_end, data in run_parts
                        if part_start < result_end and run_start < part_end
                    ]
                    data = (
                        values[0]
                        if all(value == values[0] for value in values[1:])
                        else tuple(values)
                    )
                    return IntervalResult(run_start, result_end, data=data)
            return None

    def allocate(
        self,
        length: int,
        *,
        not_before: int,
        not_after: int | None = None,
        payload_predicate: Callable[[Any], bool] | None = None,
    ) -> IntervalResult | None:
        with self._lock:
            result = self.first_fit(
                length,
                not_before=not_before,
                not_after=not_after,
                payload_predicate=payload_predicate,
            )
            if result is None:
                return None
            mutation = self.discard(result.span, require_covered=True)
            if not mutation.fully_covered:
                raise RuntimeError(
                    "allocation lost coverage while holding the manager lock"
                )
            return result

    def overlaps(self, span: Span) -> tuple[IntervalResult, ...]:
        self._validate_domain(span)
        return tuple(
            interval
            for interval in self.intervals()
            if interval.start < span.end and span.start < interval.end
        )

    def snapshot(self) -> RangeSnapshot:
        intervals = self.intervals()
        return RangeSnapshot(
            intervals,
            sum(item.end - item.start for item in intervals),
            self._domain,
        )

    def stats(self) -> AvailabilityStats:
        with self._lock:
            if self._domain is None:
                raise ManagedDomainRequiredError(
                    "availability statistics require an explicit managed domain"
                )
            intervals = self._geometry_intervals()
            total_free = sum(item.end - item.start for item in intervals)
            largest = max((item.end - item.start for item in intervals), default=0)
            return AvailabilityStats(
                total_free=total_free,
                total_occupied=self._domain.measure - total_free,
                total_space=self._domain.measure,
                free_chunks=len(intervals),
                largest_chunk=largest,
                bounds=self._domain.bounds,
            )

    # Legacy compatibility methods ---------------------------------------------
    def release_interval(self, start: int, end: int, data: Any = None) -> None:
        span = Span(start, end)
        self.add(span) if data is None else self.add(span, data)

    def reserve_interval(self, start: int, end: int, data: Any = None) -> None:
        del data
        self.discard(Span(start, end))

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
        return self.first_fit(length, not_before=start)

    def get_intervals(self) -> list[IntervalResult]:
        return list(self.intervals())

    def get_total_available_length(self) -> int:
        with self._lock:
            return self._adapter.total()

    def get_availability_stats(self) -> AvailabilityStats:
        return self.stats()

    def get_raw_implementation(self) -> Any:
        with self._lock:
            return self._adapter.implementation

    def require(self, capability: Capability) -> None:
        if capability not in self._capabilities:
            raise UnsupportedCapabilityError(
                f"backend does not support {capability.name}"
            )
