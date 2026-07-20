"""Independent dock compatibility/buffer reference."""


def occupied(start: int, end: int, before: int, after: int) -> tuple[int, int]:
    return start - before, end + after


def compatible(cargo_types: frozenset[str], cargo_type: str) -> bool:
    return cargo_type in cargo_types
