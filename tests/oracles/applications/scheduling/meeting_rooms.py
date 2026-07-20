"""Independent meeting normalization/compatibility reference."""


def normalize(local_start: int, local_end: int, offset: int) -> tuple[int, int]:
    return local_start - offset, local_end - offset


def expected_room(
    rooms: tuple[tuple[str, int, frozenset[str]], ...],
    attendees: int,
    features: frozenset[str],
) -> str | None:
    return min(
        (name for name, seats, available in rooms if attendees <= seats and features <= available),
        default=None,
    )
