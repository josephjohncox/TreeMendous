"""Versioned source diagnostics with deterministic edit remapping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)


class Severity(IntEnum):
    """Diagnostic severity ordered from informational to fatal."""

    INFO = 10
    WARNING = 20
    ERROR = 30
    FATAL = 40


@dataclass(frozen=True)
class Diagnostic:
    """Identity and source-version metadata for one diagnostic."""

    diagnostic_id: str
    file: str
    version: int
    severity: Severity
    message: str


DiagnosticRecord = IntervalRecord[str, Diagnostic]
DiagnosticHandle = RecordHandle[str]
DiagnosticSnapshot = IntervalRecordSnapshot[str, Diagnostic]


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _version(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("version must be a nonnegative integer")
    return value


class SourceDiagnosticCatalog:
    """Retain diagnostic identity across queries and source edits."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, Diagnostic](lambda value: value)

    def add(
        self,
        diagnostic_id: str,
        start: int,
        end: int,
        *,
        file: str,
        version: int,
        severity: Severity | int,
        message: str,
    ) -> DiagnosticHandle:
        """Add one diagnostic at a byte range in a specific source version."""
        try:
            parsed_severity = Severity(severity)
        except (TypeError, ValueError) as exc:
            raise ValueError("severity must be a Severity") from exc
        diagnostic = Diagnostic(
            _nonempty(diagnostic_id, "diagnostic_id"),
            _nonempty(file, "file"),
            _version(version),
            parsed_severity,
            _nonempty(message, "message"),
        )
        return self._index.insert(diagnostic_id, start, end, diagnostic)

    def update(
        self,
        handle: DiagnosticHandle,
        *,
        start: int | None = None,
        end: int | None = None,
        version: int | None = None,
        severity: Severity | int | None = None,
        message: str | None = None,
    ) -> DiagnosticRecord:
        """Update a diagnostic without changing identity or order."""
        current = self._index.get(handle)
        parsed = current.payload.severity if severity is None else Severity(severity)
        payload = replace(
            current.payload,
            version=current.payload.version if version is None else _version(version),
            severity=parsed,
            message=current.payload.message
            if message is None
            else _nonempty(message, "message"),
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=payload
        )

    def remove(self, handle: DiagnosticHandle) -> DiagnosticRecord:
        """Remove exactly one diagnostic."""
        return self._index.remove(handle, owner=handle.owner)

    def diagnostics(
        self,
        file: str,
        version: int,
        start: int,
        end: int,
        *,
        minimum_severity: Severity | int = Severity.INFO,
    ) -> tuple[DiagnosticRecord, ...]:
        """Find intersecting diagnostics ordered by severity then insertion."""
        threshold = Severity(minimum_severity)
        matches = [
            record
            for record in self._index.overlaps(start, end)
            if record.payload.file == file
            and record.payload.version == version
            and record.payload.severity >= threshold
        ]
        return tuple(
            sorted(
                matches,
                key=lambda record: (-record.payload.severity, record.insertion_order),
            )
        )

    def remap_edit(
        self,
        file: str,
        version: int,
        edit_start: int,
        edit_end: int,
        replacement_length: int,
        *,
        new_version: int,
    ) -> tuple[DiagnosticHandle, ...]:
        """Map diagnostics through ``[edit_start, edit_end)`` replacement.

        Anchors inside replaced text map proportionally only up to the new
        replacement length. Diagnostics collapsed by deletion are removed.
        The returned identities are all diagnostics updated or removed.
        """
        if edit_start < 0 or edit_end <= edit_start:
            raise ValueError("edit range must be nonempty and nonnegative")
        if replacement_length < 0:
            raise ValueError("replacement_length cannot be negative")
        old_version = _version(version)
        target_version = _version(new_version)
        if target_version <= old_version:
            raise ValueError("new_version must be greater than version")
        delta = replacement_length - (edit_end - edit_start)

        def map_offset(offset: int) -> int:
            if offset <= edit_start:
                return offset
            if offset >= edit_end:
                return offset + delta
            return edit_start + min(offset - edit_start, replacement_length)

        changed: list[DiagnosticHandle] = []
        for record in self._index.snapshot().records:
            diagnostic = record.payload
            if diagnostic.file != file or diagnostic.version != old_version:
                continue
            mapped_start = map_offset(record.start)
            mapped_end = map_offset(record.end)
            changed.append(record.handle)
            if mapped_start >= mapped_end:
                self._index.remove(record.handle, owner=record.handle.owner)
                continue
            self._index.update(
                record.handle,
                owner=record.handle.owner,
                start=mapped_start,
                end=mapped_end,
                payload=replace(diagnostic, version=target_version),
            )
        return tuple(changed)

    def snapshot(self) -> DiagnosticSnapshot:
        """Return a detached immutable diagnostic snapshot."""
        return self._index.snapshot()


def create_catalog() -> SourceDiagnosticCatalog:
    """Create an empty source diagnostic catalog."""
    return SourceDiagnosticCatalog()
