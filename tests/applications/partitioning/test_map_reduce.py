"""Map/reduce input-split engine contracts."""

import pytest

from tests.oracles.applications.partitioning.map_reduce import expected_word_counts
from treemendous.applications.partitioning.map_reduce import MapReduceEngine


def _mapper(unit: bytes):
    return ((word.lower(), 1) for word in unit.decode().split())


def _sum(left: int, right: int) -> int:
    return left + right


def _invalid_mapper(_: bytes):
    return (("", 1),)


def test_record_and_byte_splits_run_real_map_reduce_work() -> None:
    data = b"One two\ntwo three\n"
    for mode, size in (("records", 1), ("bytes", len(data))):
        engine = MapReduceEngine(data, _mapper, _sum, split_size=size, mode=mode)
        assert engine.run(shard_size=1) == expected_word_counts(data)
        assert engine.snapshot().results == expected_word_counts(data)


def test_mapper_failure_abandons_without_partial_output() -> None:
    engine = MapReduceEngine(b"x", _invalid_mapper, _sum, split_size=1)
    with pytest.raises(RuntimeError, match="mapper"):
        engine.run()
    expected: tuple[tuple[str, int], ...] = ()
    assert engine.reduce() == expected
