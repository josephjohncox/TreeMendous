"""Attested allocate/free heap application benchmark."""

from __future__ import annotations

import random
from typing import Any

from tests.performance.applications.allocation._shared import (
    empty_fragmentation,
    handle,
    span,
    stable_evidence,
    validate_inputs,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.allocation.heap import HeapAllocator

DEFAULT_OPERATIONS = 100
CAPACITY = 256
HEADER_SIZE = 8
REDZONE_SIZE = 4


def _commands(operations: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    rng = random.Random(seed)
    return tuple(
        (index, rng.randint(8, 64), rng.choice((8, 16, 32)))
        for index in range(operations)
    )


def _expected_block(
    allocation_id: int, owner: int, size: int, alignment: int
) -> dict[str, Any]:
    prefix = HEADER_SIZE + REDZONE_SIZE
    payload_start = -((-prefix) // alignment) * alignment
    raw_start = payload_start - prefix
    raw_end = raw_start + prefix + size + REDZONE_SIZE
    return {
        "raw_handle": handle(allocation_id, owner, raw_start, raw_end),
        "payload": span(payload_start, payload_start + size),
        "requested_size": size,
        "header_size": HEADER_SIZE,
        "redzone_size": REDZONE_SIZE,
        "alignment": alignment,
    }


def _oracle(commands: tuple[tuple[int, int, int], ...]) -> ApplicationOutcome:
    results = tuple(
        (_expected_block(index + 1, owner, size, alignment), None)
        for index, (owner, size, alignment) in enumerate(commands)
    )
    final_state = {
        "capacity": CAPACITY,
        "blocks": (),
        "free_ranges": (span(0, CAPACITY),),
        "diagnostics": empty_fragmentation(CAPACITY),
    }
    return ApplicationOutcome(
        results,
        final_state,
        {"commands": len(commands), "application_calls": 2 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic heap churn and attest the timed engine's final state."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    heap = HeapAllocator(CAPACITY, header_size=HEADER_SIZE, redzone_size=REDZONE_SIZE)

    def execute() -> tuple[tuple[object, None], ...]:
        outcomes: list[tuple[object, None]] = []
        for owner, size, alignment in commands:
            block = heap.allocate(size, owner=owner, alignment=alignment)
            heap.free(block, owner=owner)
            outcomes.append((block, None))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, None], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(heap.snapshot()),
            {"commands": operations, "application_calls": 2 * operations},
        )

    return run_application_case(
        scenario_id="heap-free-space",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=lambda: _oracle(commands),
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    """Delegate the legacy smoke entry point to the attested benchmark."""
    return run_benchmark(operations=operations)


if __name__ == "__main__":
    print(run_benchmark().to_dict())
