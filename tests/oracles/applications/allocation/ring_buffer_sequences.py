"""Independent ring cursor transition oracle."""


def produce(
    capacity: int, producer: int, consumer: int, count: int, overwrite: bool
) -> tuple[int, int, int]:
    overflow = max(0, producer - consumer + count - capacity)
    if overflow and not overwrite:
        raise OverflowError
    return producer + count, consumer + overflow, overflow
