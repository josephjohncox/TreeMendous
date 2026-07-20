"""Attested ring produce/consume application benchmark."""

from __future__ import annotations

import random

from tests.performance.applications.allocation._shared import (
    span,
    stable_evidence,
    validate_inputs,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.allocation.ring_buffer import RingBuffer

DEFAULT_OPERATIONS = 100
CAPACITY = 64


def _commands(operations: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.randint(1, 16) for _ in range(operations))


def _oracle(commands: tuple[int, ...], modulus: int) -> ApplicationOutcome:
    results = []
    cursor = 0
    for count in commands:
        end = cursor + count
        produced = {
            "sequences": span(cursor, end),
            "modular_start": cursor % modulus,
            "start_epoch": cursor // modulus,
            "overwritten": 0,
        }
        consumed = {
            "sequences": span(cursor, end),
            "modular_start": cursor % modulus,
            "start_epoch": cursor // modulus,
        }
        results.append((produced, consumed))
        cursor = end
    complete = span(0, cursor)
    final_state = {
        "capacity": CAPACITY,
        "modulus": modulus,
        "policy": "backpressure",
        "producer_cursor": cursor,
        "consumer_cursor": cursor,
        "occupancy": 0,
        "free_slots": CAPACITY,
        "overwritten": 0,
        "sequences": {
            "modulus": modulus,
            "max_ranges": 2,
            "origin": 0,
            "reference": cursor - 1,
            "received_ranges": (complete,),
            "contiguous_range": complete,
            "missing_ranges": (),
        },
    }
    return ApplicationOutcome(
        tuple(results),
        final_state,
        {"commands": len(commands), "application_calls": 2 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic ring churn and attest the timed ring state."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    modulus = max(256, sum(commands) + 1)
    ring = RingBuffer(CAPACITY, sequence_modulus=modulus)

    def execute() -> tuple[tuple[object, object], ...]:
        outcomes: list[tuple[object, object]] = []
        for count in commands:
            produced = ring.produce(count)
            consumed = ring.consume(count)
            outcomes.append((produced, consumed))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, object], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(ring.snapshot()),
            {"commands": operations, "application_calls": 2 * operations},
        )

    return run_application_case(
        scenario_id="ring-buffer-sequences",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=lambda: _oracle(commands, modulus),
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    """Delegate the legacy smoke entry point to the attested benchmark."""
    return run_benchmark(operations=operations)


if __name__ == "__main__":
    print(run_benchmark().to_dict())
