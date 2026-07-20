"""Independent whole-input word-count and record-split oracle."""

from __future__ import annotations

SplitSpec = tuple[int, int, int, tuple[bytes, ...]]


def expected_word_counts(data: bytes) -> tuple[tuple[str, int], ...]:
    """Count words directly over the complete input."""
    counts: dict[str, int] = {}
    for word in data.decode().split():
        word = word.lower()
        counts[word] = counts.get(word, 0) + 1
    return tuple(sorted(counts.items()))


def expected_record_splits(data: bytes, split_size: int) -> tuple[SplitSpec, ...]:
    """Partition complete newline records without calling the engine splitter."""
    records = data.splitlines(keepends=True) or [data]
    result: list[SplitSpec] = []
    cursor = 0
    for split_id, first in enumerate(range(0, len(records), split_size)):
        units = tuple(records[first : first + split_size])
        length = sum(len(unit) for unit in units)
        result.append((split_id, cursor, cursor + length, units))
        cursor += length
    return tuple(result)
