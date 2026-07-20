"""Distributed trace spans with identity, ancestry, and overlap queries."""

from __future__ import annotations

from dataclasses import dataclass, replace

from treemendous.applications._shared.coverage import (
    CoverageSnapshot,
    coverage_snapshot,
)
from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)

TraceOwner = tuple[str, str]


@dataclass(frozen=True)
class TraceSpan:
    """Trace/span/parent/service metadata retained for one timed operation."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str


TraceRecord = IntervalRecord[TraceOwner, TraceSpan]
TraceHandle = RecordHandle[TraceOwner]
TraceSnapshot = IntervalRecordSnapshot[TraceOwner, TraceSpan]
_MISSING = object()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


class TraceCatalog:
    """Retain every span and derive ancestry-aware critical paths."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[TraceOwner, TraceSpan](lambda value: value)
        self._handles: dict[TraceOwner, TraceHandle] = {}

    def add(
        self,
        trace_id: str,
        span_id: str,
        start: int,
        end: int,
        *,
        parent_span_id: str | None,
        service: str,
        operation: str,
    ) -> TraceHandle:
        """Add a span whose ID is unique within its trace."""
        owner = (_text(trace_id, "trace_id"), _text(span_id, "span_id"))
        if owner in self._handles:
            raise ValueError("span_id must be unique within a trace")
        if parent_span_id is not None:
            _text(parent_span_id, "parent_span_id")
            if parent_span_id == span_id:
                raise ValueError("a span cannot parent itself")
        span = TraceSpan(
            trace_id,
            span_id,
            parent_span_id,
            _text(service, "service"),
            _text(operation, "operation"),
        )
        handle = self._index.insert(owner, start, end, span)
        self._handles[owner] = handle
        return handle

    def update(
        self,
        handle: TraceHandle,
        *,
        start: int | None = None,
        end: int | None = None,
        parent_span_id: str | None | object = _MISSING,
        service: str | None = None,
        operation: str | None = None,
    ) -> TraceRecord:
        """Update timing or metadata without changing trace/span identity."""
        current = self._index.get(handle)
        parent = (
            current.payload.parent_span_id
            if parent_span_id is _MISSING
            else parent_span_id
        )
        if parent is not None:
            if not isinstance(parent, str):
                raise TypeError("parent_span_id must be a string or None")
            _text(parent, "parent_span_id")
            if parent == current.payload.span_id:
                raise ValueError("a span cannot parent itself")
        span = replace(
            current.payload,
            parent_span_id=parent,
            service=(
                current.payload.service
                if service is None
                else _text(service, "service")
            ),
            operation=(
                current.payload.operation
                if operation is None
                else _text(operation, "operation")
            ),
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=span
        )

    def remove(self, handle: TraceHandle) -> TraceRecord:
        """Remove exactly one span identity."""
        removed = self._index.remove(handle, owner=handle.owner)
        del self._handles[handle.owner]
        return removed

    def overlapping(
        self,
        trace_id: str,
        start: int,
        end: int,
        *,
        service: str | None = None,
    ) -> tuple[TraceRecord, ...]:
        """Return concurrent trace activity in insertion order."""
        return tuple(
            record
            for record in self._index.overlaps(start, end)
            if record.payload.trace_id == trace_id
            and (service is None or record.payload.service == service)
        )

    def concurrency(
        self, trace_id: str, start: int, end: int
    ) -> CoverageSnapshot[TraceOwner]:
        """Return identity-bearing overlap coverage for one trace."""
        return coverage_snapshot(
            self.overlapping(trace_id, start, end), start=start, end=end
        )

    def critical_path(self, trace_id: str) -> tuple[TraceRecord, ...]:
        """Return the maximum-duration parent/child chain.

        Duration is summed along each explicit ancestry chain. Ties select the
        chain whose first differing span was inserted earlier. Missing parents
        make a span a root; ancestry cycles are rejected.
        """
        records = tuple(
            record
            for record in self._index.snapshot().records
            if record.payload.trace_id == trace_id
        )
        by_id = {record.payload.span_id: record for record in records}
        children: dict[str, list[TraceRecord]] = {}
        roots: list[TraceRecord] = []
        for record in records:
            parent = record.payload.parent_span_id
            if parent is None or parent not in by_id:
                roots.append(record)
            else:
                children.setdefault(parent, []).append(record)
        visiting: set[str] = set()
        visited: set[str] = set()

        def best_from(record: TraceRecord) -> tuple[int, tuple[TraceRecord, ...]]:
            span_id = record.payload.span_id
            if span_id in visiting:
                raise ValueError("trace ancestry contains a cycle")
            visiting.add(span_id)
            choices = [best_from(child) for child in children.get(span_id, ())]
            visiting.remove(span_id)
            visited.add(span_id)
            duration = record.end - record.start
            if not choices:
                return duration, (record,)
            child_duration, child_path = max(
                choices,
                key=lambda choice: (
                    choice[0],
                    tuple(-item.insertion_order for item in choice[1]),
                ),
            )
            return duration + child_duration, (record, *child_path)

        choices = [best_from(root) for root in roots]
        if len(visited) != len(records):
            # Components without roots can only be cyclic.
            remaining = next(
                record for record in records if record.payload.span_id not in visited
            )
            best_from(remaining)
        if not choices:
            return ()
        return max(
            choices,
            key=lambda choice: (
                choice[0],
                tuple(-item.insertion_order for item in choice[1]),
            ),
        )[1]

    def snapshot(self) -> TraceSnapshot:
        """Return an immutable insertion-ordered span snapshot."""
        return self._index.snapshot()


def create_catalog() -> TraceCatalog:
    """Create an empty distributed trace catalog."""
    return TraceCatalog()
