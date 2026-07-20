#!/usr/bin/env python3
"""Run deterministic injected-target fuzzing from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from treemendous.applications.partitioning.fuzzing import FuzzingEngine


def target(data: bytes) -> None:
    if len(data) == 2:
        raise ValueError("length two")


def main() -> None:
    engine = FuzzingEngine(target, cases=40, seed=3, max_input_size=4)
    crashes = engine.run(shard_size=7, fail_first_claim=True)
    if len(crashes) != 1 or engine.snapshot().retries != 1:
        raise RuntimeError("unexpected fuzzing result")
    print("fuzzing: 1 signature, 1 retried claim")


if __name__ == "__main__":
    main()
