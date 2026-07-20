"""Smoke benchmark for source diagnostic queries."""

from time import perf_counter

from treemendous.applications.catalogs.source_diagnostic_ranges import (
    Severity,
    SourceDiagnosticCatalog,
)


def run_smoke(iterations: int = 500) -> float:
    catalog = SourceDiagnosticCatalog()
    for index in range(200):
        catalog.add(
            f"d{index}",
            index * 3,
            index * 3 + 8,
            file="main.py",
            version=1,
            severity=Severity.WARNING,
            message="warning",
        )
    started = perf_counter()
    for index in range(iterations):
        catalog.diagnostics("main.py", 1, index % 500, index % 500 + 20)
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
