"""Smoke benchmark for trace overlap and critical-path queries."""

from time import perf_counter

from treemendous.applications.catalogs.distributed_trace_spans import TraceCatalog


def run_smoke(iterations: int = 250) -> float:
    catalog = TraceCatalog()
    catalog.add(
        "trace",
        "root",
        0,
        1000,
        parent_span_id=None,
        service="api",
        operation="request",
    )
    for index in range(100):
        catalog.add(
            "trace",
            f"s{index}",
            index * 8,
            index * 8 + 20,
            parent_span_id="root",
            service="worker",
            operation="work",
        )
    started = perf_counter()
    for index in range(iterations):
        catalog.overlapping("trace", index % 800, index % 800 + 25)
        catalog.critical_path("trace")
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
