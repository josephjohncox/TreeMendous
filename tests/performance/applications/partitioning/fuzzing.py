"""Correctness-checked smoke workload for deterministic fuzzing."""

from tests.oracles.applications.partitioning.fuzzing import generated, signature
from treemendous.applications.partitioning.fuzzing import FuzzingEngine


def _target(data: bytes) -> None:
    if len(data) == 5:
        raise RuntimeError("five")


def run_smoke() -> int:
    engine = FuzzingEngine(_target, cases=400, seed=11, max_input_size=16)
    crashes = engine.run(shard_size=29, fail_first_claim=True)
    if any(engine.input_for(i) != generated(11, i, 16) for i in range(400)):
        raise AssertionError("fuzz input mapping differs from oracle")
    expected = {signature(RuntimeError("five"))}
    if {item.signature for item in crashes} != expected:
        raise AssertionError("fuzz crash signatures differ from oracle")
    return len(engine.snapshot().executed_ordinals)
