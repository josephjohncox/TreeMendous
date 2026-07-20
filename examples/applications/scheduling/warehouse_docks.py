"""Cargo-compatible dock appointment example."""

from treemendous.applications.scheduling.warehouse_docks import (
    create_warehouse_dock_scheduler,
)


def main() -> None:
    scheduler = create_warehouse_dock_scheduler()
    placement = scheduler.book(
        "carrier", 2, cargo_type="dry", earliest_start=5, latest_end=10,
        handling_before=1, handling_after=1, request_id="load-1",
    )
    print(placement.resource, placement.start, placement.end)


if __name__ == "__main__":
    main()
