"""Calibrated laboratory instrument example."""

from treemendous.applications.scheduling.lab_instruments import (
    create_lab_instrument_scheduler,
)


def main() -> None:
    scheduler = create_lab_instrument_scheduler()
    placement = scheduler.book(
        "sample", 4, capabilities=frozenset({"imaging"}),
        earliest_start=5, latest_end=20, request_id="sample-1",
    )
    print(placement.resource, placement.start, placement.reservation.occupied_span.end)


if __name__ == "__main__":
    main()
