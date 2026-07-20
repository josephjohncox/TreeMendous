"""Bounded point-scan packet reassembly oracle."""

from __future__ import annotations


def assemble(
    fragments: list[tuple[int, bytes]], start: int, end: int
) -> tuple[bytes | None, tuple[int, ...]]:
    """Use first-arrival bytes and report missing sequence points."""
    values: list[int | None] = [None] * (end - start)
    for sequence, fragment_payload in fragments:
        for point in range(
            max(start, sequence), min(end, sequence + len(fragment_payload))
        ):
            if values[point - start] is None:
                values[point - start] = fragment_payload[point - sequence]
    missing = tuple(
        start + index for index, value in enumerate(values) if value is None
    )
    assembled: bytes | None = (
        None if missing else bytes(value for value in values if value is not None)
    )
    return assembled, missing
