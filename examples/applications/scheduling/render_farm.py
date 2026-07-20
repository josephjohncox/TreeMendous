"""Deterministic render-farm chunk and retry example."""

from treemendous.applications.scheduling.render_farm import create_render_farm_scheduler


def main() -> None:
    scheduler = create_render_farm_scheduler()
    first = scheduler.assign_chunk(
        "shot", 1, 20, 2, earliest_start=0, latest_end=4, request_id="attempt-1"
    )
    retry = scheduler.retry(
        "shot", first.id, duration=2, earliest_start=2, latest_end=6,
        request_id="attempt-2",
    )
    print(retry.frames.start, retry.frames.end, retry.attempt)


if __name__ == "__main__":
    main()
