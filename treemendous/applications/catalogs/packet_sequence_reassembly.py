"""Identity-preserving packet fragment and payload reassembly catalog."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.coverage import CoverageSegment, coverage_segments
from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)
from treemendous.domain import Span


@dataclass(frozen=True)
class PacketFragment:
    """One received fragment, including duplicate receive identity."""

    flow_id: str
    payload: bytes


PacketRecord = IntervalRecord[str, PacketFragment]
PacketHandle = RecordHandle[str]
PacketSnapshot = IntervalRecordSnapshot[str, PacketFragment]


@dataclass(frozen=True)
class ReassemblyResult:
    """Bounded reassembly with fragments, gaps, duplicate coverage, and data."""

    flow_id: str
    start: int
    end: int
    fragments: tuple[PacketRecord, ...]
    gaps: tuple[Span, ...]
    duplicate_coverage: tuple[CoverageSegment[str], ...]
    payload: bytes | None

    @property
    def complete(self) -> bool:
        """Whether every requested sequence byte was available."""
        return not self.gaps


def _flow(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("flow_id must be a nonempty string")
    return value


class PacketReassemblyCatalog:
    """Store every packet arrival and derive deterministic first-arrival data."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, PacketFragment](lambda value: value)

    def add(self, flow_id: str, sequence: int, payload: bytes) -> PacketHandle:
        """Record a nonempty fragment; coincident duplicates remain distinct."""
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be nonempty bytes")
        return self._index.insert(
            _flow(flow_id),
            sequence,
            sequence + len(payload),
            PacketFragment(flow_id, payload),
        )

    def update(
        self,
        handle: PacketHandle,
        *,
        sequence: int | None = None,
        payload: bytes | None = None,
    ) -> PacketRecord:
        """Correct one received fragment without replacing its identity."""
        current = self._index.get(handle)
        data = current.payload.payload if payload is None else payload
        if not isinstance(data, bytes) or not data:
            raise ValueError("payload must be nonempty bytes")
        start = current.start if sequence is None else sequence
        return self._index.update(
            handle,
            owner=handle.owner,
            start=start,
            end=start + len(data),
            payload=PacketFragment(current.payload.flow_id, data),
        )

    def remove(self, handle: PacketHandle) -> PacketRecord:
        """Remove exactly one arrival identity."""
        return self._index.remove(handle, owner=handle.owner)

    def fragments(self, flow_id: str, start: int, end: int) -> tuple[PacketRecord, ...]:
        """Return overlapping arrivals in receive order."""
        return tuple(
            record
            for record in self._index.overlaps(start, end)
            if record.payload.flow_id == flow_id
        )

    def assemble(self, flow_id: str, start: int, end: int) -> ReassemblyResult:
        """Assemble a bounded sequence using the earliest byte received.

        Later duplicate/conflicting bytes are preserved as records but do not
        overwrite first-arrival payload. ``payload`` is ``None`` when gaps
        remain, avoiding an ambiguous concatenation across missing bytes.
        """
        requested = Span(start, end)
        fragments = self.fragments(_flow(flow_id), start, end)
        byte_values: list[int | None] = [None] * (end - start)
        for record in fragments:
            for position in range(max(start, record.start), min(end, record.end)):
                offset = position - start
                if byte_values[offset] is None:
                    byte_values[offset] = record.payload.payload[
                        position - record.start
                    ]
        gaps: list[Span] = []
        cursor = 0
        while cursor < len(byte_values):
            if byte_values[cursor] is not None:
                cursor += 1
                continue
            gap_start = cursor
            while cursor < len(byte_values) and byte_values[cursor] is None:
                cursor += 1
            gaps.append(Span(start + gap_start, start + cursor))
        duplicates = tuple(
            segment
            for segment in coverage_segments(fragments, start=start, end=end)
            if segment.count > 1
        )
        assembled = (
            None if gaps else bytes(value for value in byte_values if value is not None)
        )
        return ReassemblyResult(
            flow_id,
            requested.start,
            requested.end,
            fragments,
            tuple(gaps),
            duplicates,
            assembled,
        )

    def snapshot(self) -> PacketSnapshot:
        """Return every retained arrival in receive order."""
        return self._index.snapshot()


def create_catalog() -> PacketReassemblyCatalog:
    """Create an empty packet reassembly catalog."""
    return PacketReassemblyCatalog()
