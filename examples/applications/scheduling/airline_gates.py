"""Aircraft-compatible gate assignment example."""

from treemendous.applications.scheduling.airline_gates import (
    create_airline_gate_scheduler,
)


def main() -> None:
    scheduler = create_airline_gate_scheduler()
    placement = scheduler.assign(
        "TM123",
        10,
        15,
        aircraft_type="A320",
        turnaround_before=1,
        turnaround_after=2,
        request_id="arrival",
    )
    print(placement.resource, placement.reservation.occupied_span)


if __name__ == "__main__":
    main()
