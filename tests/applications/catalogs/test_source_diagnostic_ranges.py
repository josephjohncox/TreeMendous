"""Source diagnostic catalog contract."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from tests.oracles.applications.catalogs.source_diagnostic_ranges import (
    DiagnosticRow,
    query,
    remap_edit,
)
from treemendous.applications.catalogs.source_diagnostic_ranges import (
    Diagnostic,
    DiagnosticRecord,
    DiagnosticSnapshot,
    Severity,
    SourceDiagnosticCatalog,
)


def _row(record: DiagnosticRecord) -> DiagnosticRow:
    payload = record.payload
    return (
        payload.diagnostic_id,
        payload.file,
        payload.version,
        int(payload.severity),
        record.start,
        record.end,
    )


def _catalog_for(rows: list[DiagnosticRow]) -> SourceDiagnosticCatalog:
    catalog = SourceDiagnosticCatalog()
    for diagnostic_id, file, version, severity, start, end in rows:
        catalog.add(
            diagnostic_id,
            start,
            end,
            file=file,
            version=version,
            severity=severity,
            message=diagnostic_id,
        )
    return catalog


def test_diagnostics_preserve_versions_priority_and_remap_edits() -> None:
    catalog = SourceDiagnosticCatalog()
    warning = catalog.add(
        "w", 5, 8, file="main.py", version=1, severity=Severity.WARNING, message="warn"
    )
    error = catalog.add(
        "e", 9, 12, file="main.py", version=1, severity=Severity.ERROR, message="error"
    )
    rows: list[DiagnosticRow] = [
        ("w", "main.py", 1, 20, 5, 8),
        ("e", "main.py", 1, 30, 9, 12),
    ]
    actual = catalog.diagnostics("main.py", 1, 0, 20, minimum_severity=Severity.INFO)
    assert tuple(record.payload.diagnostic_id for record in actual) == query(
        rows, "main.py", 1, 0, 20, 10
    )
    assert list(catalog.remap_edit("main.py", 1, 6, 10, 2, new_version=2)) == [
        warning,
        error,
    ]
    assert tuple(_row(record) for record in catalog.snapshot().records) == remap_edit(
        rows, "main.py", 1, 6, 10, 2, new_version=2
    )
    assert catalog.update(error, severity=Severity.FATAL).handle == error
    assert catalog.remove(warning).handle == warning


@pytest.mark.parametrize(
    ("edit_start", "edit_end", "replacement_length", "expected"),
    [
        (
            5,
            5,
            3,
            (
                ("before", "main.py", 2, 20, 1, 4),
                ("left", "main.py", 2, 20, 2, 9),
                ("right", "main.py", 2, 20, 8, 12),
                ("enclosing", "main.py", 2, 20, 2, 13),
                ("inside", "main.py", 2, 20, 8, 9),
                ("other", "other.py", 1, 20, 5, 8),
            ),
        ),
        (
            4,
            7,
            0,
            (
                ("before", "main.py", 2, 20, 1, 4),
                ("left", "main.py", 2, 20, 2, 4),
                ("right", "main.py", 2, 20, 4, 6),
                ("enclosing", "main.py", 2, 20, 2, 7),
                ("other", "other.py", 1, 20, 5, 8),
            ),
        ),
        (
            4,
            7,
            5,
            (
                ("before", "main.py", 2, 20, 1, 4),
                ("left", "main.py", 2, 20, 2, 6),
                ("right", "main.py", 2, 20, 5, 11),
                ("enclosing", "main.py", 2, 20, 2, 12),
                ("inside", "main.py", 2, 20, 5, 6),
                ("other", "other.py", 1, 20, 5, 8),
            ),
        ),
        (
            4,
            7,
            1,
            (
                ("before", "main.py", 2, 20, 1, 4),
                ("left", "main.py", 2, 20, 2, 5),
                ("right", "main.py", 2, 20, 5, 7),
                ("enclosing", "main.py", 2, 20, 2, 8),
                ("other", "other.py", 1, 20, 5, 8),
            ),
        ),
    ],
)
def test_remap_edit_matches_independent_boundary_oracle(
    edit_start: int,
    edit_end: int,
    replacement_length: int,
    expected: tuple[DiagnosticRow, ...],
) -> None:
    rows: list[DiagnosticRow] = [
        ("before", "main.py", 1, 20, 1, 4),
        ("left", "main.py", 1, 20, 2, 6),
        ("right", "main.py", 1, 20, 5, 9),
        ("enclosing", "main.py", 1, 20, 2, 10),
        ("inside", "main.py", 1, 20, 5, 6),
        ("other", "other.py", 1, 20, 5, 8),
    ]
    modeled = remap_edit(
        rows,
        "main.py",
        1,
        edit_start,
        edit_end,
        replacement_length,
        new_version=2,
    )
    assert modeled == expected
    catalog = _catalog_for(rows)

    changed = catalog.remap_edit(
        "main.py",
        1,
        edit_start,
        edit_end,
        replacement_length,
        new_version=2,
    )

    assert len(changed) == 5
    assert tuple(_row(record) for record in catalog.snapshot().records) == modeled


def test_insertion_affinity_at_start_end_and_inside_is_explicit() -> None:
    rows: list[DiagnosticRow] = [
        ("ending", "main.py", 1, 20, 2, 5),
        ("starting", "main.py", 1, 20, 5, 8),
        ("enclosing", "main.py", 1, 20, 2, 8),
        ("before", "main.py", 1, 20, 1, 4),
        ("after", "main.py", 1, 20, 6, 9),
    ]
    expected: tuple[DiagnosticRow, ...] = (
        ("ending", "main.py", 2, 20, 2, 5),
        ("starting", "main.py", 2, 20, 8, 11),
        ("enclosing", "main.py", 2, 20, 2, 11),
        ("before", "main.py", 2, 20, 1, 4),
        ("after", "main.py", 2, 20, 9, 12),
    )
    catalog = _catalog_for(rows)

    catalog.remap_edit("main.py", 1, 5, 5, 3, new_version=2)

    modeled = remap_edit(rows, "main.py", 1, 5, 5, 3, new_version=2)
    assert modeled == expected
    assert tuple(_row(record) for record in catalog.snapshot().records) == modeled


def test_insertion_remaps_coincident_diagnostics_independently() -> None:
    rows: list[DiagnosticRow] = [
        ("first", "main.py", 1, 20, 5, 8),
        ("second", "main.py", 1, 30, 5, 8),
    ]
    catalog = _catalog_for(rows)

    changed = catalog.remap_edit("main.py", 1, 5, 5, 2, new_version=2)

    assert len(set(changed)) == 2
    assert tuple(_row(record) for record in catalog.snapshot().records) == remap_edit(
        rows, "main.py", 1, 5, 5, 2, new_version=2
    )


def test_remap_edit_is_atomic_when_staging_fails_mid_batch() -> None:
    rows: list[DiagnosticRow] = [
        ("first", "main.py", 1, 20, 1, 4),
        ("second", "main.py", 1, 20, 5, 8),
        ("third", "main.py", 1, 20, 9, 12),
    ]
    catalog = _catalog_for(rows)
    before = catalog.snapshot()
    original_cloner = catalog._index._cloner

    def fail_on_second_new_version(diagnostic: Diagnostic) -> Diagnostic:
        if diagnostic.version == 2 and diagnostic.diagnostic_id == "second":
            raise RuntimeError("injected staging failure")
        return diagnostic

    catalog._index._cloner = fail_on_second_new_version
    with pytest.raises(RuntimeError, match="injected staging failure"):
        catalog.remap_edit("main.py", 1, 3, 7, 2, new_version=2)
    catalog._index._cloner = original_cloner

    assert catalog.snapshot() == before


def test_concurrent_reads_and_updates_cannot_interleave_with_remap() -> None:
    rows: list[DiagnosticRow] = [
        ("first", "main.py", 1, 20, 1, 4),
        ("second", "main.py", 1, 20, 5, 8),
    ]
    catalog = _catalog_for(rows)
    second = catalog.snapshot().records[1].handle
    clone_started = Event()
    clone_may_finish = Event()
    snapshot_started = Event()
    snapshot_finished = Event()
    update_started = Event()
    update_finished = Event()
    snapshots: list[DiagnosticSnapshot] = []
    errors: list[BaseException] = []
    original_cloner = catalog._index._cloner

    def blocking_cloner(diagnostic: Diagnostic) -> Diagnostic:
        if (
            diagnostic.version == 2
            and diagnostic.diagnostic_id == "first"
            and not clone_started.is_set()
        ):
            clone_started.set()
            if not clone_may_finish.wait(timeout=2):
                raise RuntimeError("test did not release staged remap")
        return diagnostic

    def run_remap() -> None:
        try:
            catalog.remap_edit("main.py", 1, 3, 6, 1, new_version=2)
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    def take_snapshot() -> None:
        snapshot_started.set()
        try:
            snapshots.append(catalog.snapshot())
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)
        finally:
            snapshot_finished.set()

    def update_second() -> None:
        update_started.set()
        try:
            catalog.update(second, message="updated after remap")
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)
        finally:
            update_finished.set()

    catalog._index._cloner = blocking_cloner
    remapper = Thread(target=run_remap)
    remapper.start()
    assert clone_started.wait(timeout=1)
    observer = Thread(target=take_snapshot)
    writer = Thread(target=update_second)
    observer.start()
    writer.start()
    assert snapshot_started.wait(timeout=1)
    assert update_started.wait(timeout=1)
    assert not snapshot_finished.wait(timeout=0.05)
    assert not update_finished.wait(timeout=0.05)
    clone_may_finish.set()
    remapper.join(timeout=1)
    observer.join(timeout=1)
    writer.join(timeout=1)
    catalog._index._cloner = original_cloner

    assert not remapper.is_alive()
    assert not observer.is_alive()
    assert not writer.is_alive()
    assert not errors
    snapshot = snapshots[0]
    assert {record.payload.version for record in snapshot.records} == {2}
    assert catalog.snapshot().records[1].payload.message == "updated after remap"
