"""Exact two-dimensional channel/time reservation example."""

from treemendous.applications.scheduling.radio_spectrum import (
    create_radio_spectrum_scheduler,
)


def main() -> None:
    scheduler = create_radio_spectrum_scheduler(channel_count=32)
    reservation = scheduler.reserve(
        "transmitter", 8, 4, 100, 110, guard_channels=1, request_id="burst-1"
    )
    print(
        reservation.channel_start,
        reservation.channel_end,
        reservation.start,
        reservation.end,
    )


if __name__ == "__main__":
    main()
