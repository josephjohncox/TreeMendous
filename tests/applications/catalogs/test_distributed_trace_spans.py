"""Distributed trace span catalog contract."""

from tests.oracles.applications.catalogs.distributed_trace_spans import overlapping
from treemendous.applications.catalogs.distributed_trace_spans import TraceCatalog


def test_trace_spans_keep_ancestry_service_overlap_and_critical_path() -> None:
    catalog = TraceCatalog()
    root = catalog.add(
        "trace",
        "root",
        0,
        20,
        parent_span_id=None,
        service="gateway",
        operation="request",
    )
    child = catalog.add(
        "trace",
        "db",
        2,
        12,
        parent_span_id="root",
        service="database",
        operation="select",
    )
    sibling = catalog.add(
        "trace", "cache", 3, 8, parent_span_id="root", service="cache", operation="get"
    )
    rows = [
        ("trace", "root", "", 0, 20),
        ("trace", "db", "root", 2, 12),
        ("trace", "cache", "root", 3, 8),
    ]
    assert tuple(
        record.payload.span_id for record in catalog.overlapping("trace", 4, 5)
    ) == overlapping(rows, "trace", 4, 5)
    assert [record.handle for record in catalog.critical_path("trace")] == [root, child]
    assert catalog.concurrency("trace", 4, 5).maximum_count == 3
    assert catalog.update(sibling, end=15).handle == sibling
    assert [record.handle for record in catalog.critical_path("trace")] == [
        root,
        sibling,
    ]
    assert catalog.remove(child).handle == child
    assert len(catalog.snapshot().records) == 2
