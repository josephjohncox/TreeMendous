"""Find a distributed trace critical path."""

from treemendous.applications.catalogs.distributed_trace_spans import TraceCatalog


def main() -> None:
    catalog = TraceCatalog()
    catalog.add(
        "trace", "root", 0, 20, parent_span_id=None, service="api", operation="request"
    )
    catalog.add(
        "trace", "sql", 3, 15, parent_span_id="root", service="db", operation="select"
    )
    print([record.payload.span_id for record in catalog.critical_path("trace")])


if __name__ == "__main__":
    main()
