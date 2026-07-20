"""Independent gate buffer/compatibility reference."""


def occupied(arrival: int, departure: int, before: int, after: int) -> tuple[int, int]:
    return arrival - before, departure + after


def compatible(types: frozenset[str], aircraft_type: str) -> bool:
    return aircraft_type in types
