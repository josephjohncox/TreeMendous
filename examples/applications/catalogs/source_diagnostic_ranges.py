"""Remap a versioned source diagnostic through an edit."""

from treemendous.applications.catalogs.source_diagnostic_ranges import (
    Severity,
    SourceDiagnosticCatalog,
)


def main() -> None:
    catalog = SourceDiagnosticCatalog()
    catalog.add(
        "unused",
        10,
        15,
        file="app.py",
        version=1,
        severity=Severity.WARNING,
        message="unused name",
    )
    catalog.remap_edit("app.py", 1, 0, 2, 8, new_version=2)
    print(catalog.diagnostics("app.py", 2, 0, 30)[0])


if __name__ == "__main__":
    main()
