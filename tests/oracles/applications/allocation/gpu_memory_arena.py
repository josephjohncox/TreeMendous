"""Independent stream completion oracle."""


def eligible(
    pending: tuple[tuple[str, int, int], ...], stream: str, epoch: int
) -> tuple[int, ...]:
    return tuple(
        allocation_id
        for candidate_stream, candidate_epoch, allocation_id in sorted(
            pending, key=lambda item: (item[1], item[2])
        )
        if candidate_stream == stream and candidate_epoch <= epoch
    )
