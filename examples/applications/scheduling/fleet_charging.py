"""Bounded fleet charging example."""

from treemendous.applications.scheduling.fleet_charging import (
    create_fleet_charging_scheduler,
)


def main() -> None:
    scheduler = create_fleet_charging_scheduler()
    session = scheduler.schedule(
        "van", 35, connector="ccs", arrival=2, departure=8, request_id="route-1"
    )
    print(session.charger, session.duration, session.reservation.start)


if __name__ == "__main__":
    main()
