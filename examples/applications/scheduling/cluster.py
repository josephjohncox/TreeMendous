"""Deterministic cluster placement example."""

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.cluster import create_cluster_scheduler


def main() -> None:
    scheduler = create_cluster_scheduler()
    placement = scheduler.schedule(
        "build", 2, CapacityVector(cpu=2, memory=4),
        required_labels=frozenset({"linux"}), latest_end=8, request_id="demo",
    )
    print(placement.resource, placement.start, placement.end)


if __name__ == "__main__":
    main()
