"""Distributed trace span catalog contract."""

import pytest

from tests.oracles.applications.catalogs.distributed_trace_spans import (
    TraceRow,
    critical_path,
    overlapping,
)
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
    rows: list[TraceRow] = [
        ("trace", "root", None, 0, 20),
        ("trace", "db", "root", 2, 12),
        ("trace", "cache", "root", 3, 8),
    ]
    assert tuple(
        record.payload.span_id for record in catalog.overlapping("trace", 4, 5)
    ) == overlapping(rows, "trace", 4, 5)
    path = catalog.critical_path("trace")
    assert tuple(record.payload.span_id for record in path) == critical_path(
        rows, "trace"
    )
    assert [record.handle for record in path] == [root, child]
    assert catalog.concurrency("trace", 4, 5).maximum_count == 3
    assert catalog.update(sibling, end=15).handle == sibling
    assert [record.handle for record in catalog.critical_path("trace")] == [
        root,
        sibling,
    ]
    assert catalog.remove(child).handle == child
    assert len(catalog.snapshot().records) == 2


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                ("trace", "root", None, 0, 5),
                ("trace", "first", "root", 5, 9),
                ("trace", "second", "root", 10, 14),
            ],
            ("root", "first"),
        ),
        (
            [
                ("trace", "root", None, 0, 5),
                ("trace", "first", "root", 5, 9),
                ("trace", "second", "root", 10, 14),
                ("trace", "deep", "second", 14, 15),
            ],
            ("root", "second", "deep"),
        ),
        (
            [
                ("trace", "root", None, 0, 5),
                ("trace", "child", "root", 5, 9),
                ("trace", "orphan", "missing", 20, 30),
            ],
            ("orphan",),
        ),
    ],
)
def test_critical_path_matches_independent_fixed_vectors(
    rows: list[TraceRow], expected: tuple[str, ...]
) -> None:
    catalog = TraceCatalog()
    for trace_id, span_id, parent_id, start, end in rows:
        catalog.add(
            trace_id,
            span_id,
            start,
            end,
            parent_span_id=parent_id,
            service="service",
            operation="operation",
        )

    modeled = critical_path(rows, "trace")
    actual = tuple(record.payload.span_id for record in catalog.critical_path("trace"))

    assert modeled == expected
    assert actual == modeled


def test_critical_path_cycle_failure_does_not_mutate_catalog() -> None:
    rows: list[TraceRow] = [
        ("trace", "first", "second", 0, 2),
        ("trace", "second", "first", 2, 4),
    ]
    catalog = TraceCatalog()
    for trace_id, span_id, parent_id, start, end in rows:
        catalog.add(
            trace_id,
            span_id,
            start,
            end,
            parent_span_id=parent_id,
            service="service",
            operation="operation",
        )
    before = catalog.snapshot()

    with pytest.raises(ValueError, match="cycle"):
        critical_path(rows, "trace")
    with pytest.raises(ValueError, match="cycle"):
        catalog.critical_path("trace")

    assert catalog.snapshot() == before
