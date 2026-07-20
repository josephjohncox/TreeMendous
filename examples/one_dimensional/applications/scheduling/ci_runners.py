"""Deterministic CI runner example."""

from treemendous.applications.scheduling.ci_runners import create_ci_runner_scheduler


def main() -> None:
    scheduler = create_ci_runner_scheduler()
    placement = scheduler.schedule(
        "tests", 3, labels=frozenset({"linux"}), release_time=2, deadline=10
    )
    print(placement.resource, placement.start, placement.end)


if __name__ == "__main__":
    main()
