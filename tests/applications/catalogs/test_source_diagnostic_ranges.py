"""Source diagnostic catalog contract."""

from tests.oracles.applications.catalogs.source_diagnostic_ranges import query
from treemendous.applications.catalogs.source_diagnostic_ranges import (
    Severity,
    SourceDiagnosticCatalog,
)


def test_diagnostics_preserve_versions_priority_and_remap_edits() -> None:
    catalog = SourceDiagnosticCatalog()
    warning = catalog.add(
        "w", 5, 8, file="main.py", version=1, severity=Severity.WARNING, message="warn"
    )
    error = catalog.add(
        "e", 9, 12, file="main.py", version=1, severity=Severity.ERROR, message="error"
    )
    rows = [("w", "main.py", 1, 20, 5, 8), ("e", "main.py", 1, 30, 9, 12)]
    actual = catalog.diagnostics("main.py", 1, 0, 20, minimum_severity=Severity.INFO)
    assert tuple(record.payload.diagnostic_id for record in actual) == query(
        rows, "main.py", 1, 0, 20, 10
    )
    assert list(catalog.remap_edit("main.py", 1, 6, 10, 2, new_version=2)) == [
        warning,
        error,
    ]
    assert catalog.snapshot().records[0].span.start == 5
    assert catalog.snapshot().records[1].span.start == 8
    assert catalog.update(error, severity=Severity.FATAL).handle == error
    assert catalog.remove(warning).handle == warning
