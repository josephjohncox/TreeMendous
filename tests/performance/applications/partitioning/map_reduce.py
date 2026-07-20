"""Correctness-checked smoke workload for map/reduce input splits."""

from tests.oracles.applications.partitioning.map_reduce import expected_word_counts
from treemendous.applications.partitioning.map_reduce import MapReduceEngine


def _mapper(unit: bytes):
    return ((word.lower(), 1) for word in unit.decode().split())


def _sum(left: int, right: int) -> int:
    return left + right


def run_smoke() -> int:
    data = b"".join(f"term-{i % 17} shared\n".encode() for i in range(500))
    engine = MapReduceEngine(data, _mapper, _sum, split_size=19, mode="records")
    observed = engine.run(shard_size=5)
    if observed != expected_word_counts(data):
        raise AssertionError("map/reduce smoke differs from whole-input oracle")
    return len(engine.splits)
