"""Attested stream-deferred GPU arena application benchmark."""

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
from treemendous.applications.allocation.gpu_arena import GPUMemoryArena

DEFAULT_OPERATIONS = 100
CAPACITY = 4096
STREAMS = ("stream-0", "stream-1", "stream-2")
DEVICE_ALIGNMENT = 256


def _commands(operations: int, seed: int) -> tuple[tuple[str, int, int, int], ...]:
    rng = random.Random(seed)
    return tuple(
        (
            rng.choice(STREAMS),
            rng.randint(64, 384),
            rng.choice((256, 512, 1024)),
            index,
        )
        for index in range(operations)
    )


def _expected_buffer(
    allocation_id: int, stream: str, size: int, alignment: int
) -> dict[str, Any]:
    return {
        "handle": handle(allocation_id, stream, 0, size),
        "stream": stream,
        "alignment": alignment,
    }


def _oracle(commands: tuple[tuple[str, int, int, int], ...]) -> ApplicationOutcome:
    results = []
    completed: dict[str, int] = {}
    for index, (stream, size, alignment, epoch) in enumerate(commands):
        buffer = _expected_buffer(index + 1, stream, size, alignment)
        deferred = {"buffer": buffer, "completion_epoch": epoch}
        results.append((buffer, deferred, (buffer,)))
        completed[stream] = epoch
    final_state = {
        "live_buffers": (),
        "deferred_frees": (),
        "completed_epochs": tuple(completed.items()),
        "free_ranges": (span(0, CAPACITY),),
        "diagnostics": empty_fragmentation(CAPACITY),
    }
    return ApplicationOutcome(
        tuple(results),
        final_state,
        {"commands": len(commands), "application_calls": 3 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic deferred frees and attest the timed arena state."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    arena = GPUMemoryArena(CAPACITY, device_alignment=DEVICE_ALIGNMENT)

    def execute() -> tuple[tuple[object, object, tuple[object, ...]], ...]:
        outcomes: list[tuple[object, object, tuple[object, ...]]] = []
        for stream, size, alignment, epoch in commands:
            buffer = arena.allocate(size, stream=stream, alignment=alignment)
            deferred = arena.defer_free(buffer, stream=stream, completion_epoch=epoch)
            reclaimed = arena.advance_completion(stream, epoch)
            outcomes.append((buffer, deferred, reclaimed))
        return tuple(outcomes)

    def observe(
        raw: tuple[tuple[object, object, tuple[object, ...]], ...],
    ) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(arena.snapshot()),
            {"commands": operations, "application_calls": 3 * operations},
        )

    return run_application_case(
        scenario_id="gpu-memory-arena",
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
