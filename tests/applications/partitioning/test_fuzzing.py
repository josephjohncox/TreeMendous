"""Fuzzing engine contracts."""

import pytest

from tests.oracles.applications.partitioning.fuzzing import generated, signature
from treemendous.applications.partitioning.fuzzing import FuzzingEngine


def test_inputs_crashes_dedup_and_abandoned_retry_are_deterministic() -> None:
    def target(data: bytes) -> None:
        if len(data) % 3 == 0:
            raise ValueError("multiple-of-three")

    engine = FuzzingEngine(target, cases=20, seed=4, max_input_size=8)
    crashes = engine.run(shard_size=5, fail_first_claim=True)
    assert engine.input_for(7) == generated(4, 7, 8)
    assert len({item.signature for item in crashes}) == len(crashes) == 1
    assert crashes[0].signature == signature(ValueError("multiple-of-three"))
    assert engine.snapshot().retries == 1
    assert engine.snapshot().executed_ordinals == tuple(range(20))


def test_fuzzing_rejects_out_of_domain_ordinals() -> None:
    engine = FuzzingEngine(lambda _: None, cases=1)
    with pytest.raises(ValueError, match="outside"):
        engine.input_for(1)
