"""Dependency-aware maintenance planning example."""

from treemendous.applications.scheduling.maintenance import create_maintenance_scheduler


def main() -> None:
    scheduler = create_maintenance_scheduler()
    database = scheduler.schedule("database", "api", 3, latest_end=20)
    application = scheduler.schedule(
        "application", "api", 2, dependencies=("database",), latest_end=20
    )
    print(database.reservation.start, application.reservation.start)


if __name__ == "__main__":
    main()
