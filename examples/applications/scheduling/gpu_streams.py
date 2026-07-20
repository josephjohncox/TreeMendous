"""Deterministic GPU stream placement example."""

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.gpu_streams import create_gpu_stream_scheduler


def main() -> None:
    scheduler = create_gpu_stream_scheduler()
    placement = scheduler.schedule(
        "kernel",
        2,
        CapacityVector(memory=4, slots=1),
        compatibility="compute",
        dependency_ready_times={"upload": 3},
        latest_end=8,
    )
    print(placement.device, placement.stream, placement.start)


if __name__ == "__main__":
    main()
