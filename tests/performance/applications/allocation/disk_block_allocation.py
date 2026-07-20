"""Attested filesystem extent churn application benchmark."""

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
from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator

DEFAULT_OPERATIONS = 100
TOTAL_BLOCKS = 128
METADATA_BLOCKS = 8
BLOCK_SIZE = 4096


def _commands(operations: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = random.Random(seed)
    return tuple((index, rng.randint(1, 16)) for index in range(operations))


def _expected_extent(
    allocation_id: int, file_id: int, block_count: int
) -> dict[str, Any]:
    return {
        "handle": handle(
            allocation_id,
            file_id,
            METADATA_BLOCKS,
            METADATA_BLOCKS + block_count,
        ),
        "file_id": file_id,
        "block_size": BLOCK_SIZE,
    }


def _oracle(commands: tuple[tuple[int, int], ...]) -> ApplicationOutcome:
    results = tuple(
        (_expected_extent(index + 1, file_id, block_count), None)
        for index, (file_id, block_count) in enumerate(commands)
    )
    final_state = {
        "total_blocks": TOTAL_BLOCKS,
        "block_size": BLOCK_SIZE,
        "metadata_blocks": span(0, METADATA_BLOCKS),
        "extents": (),
        "free_extents": (span(METADATA_BLOCKS, TOTAL_BLOCKS),),
        "diagnostics": empty_fragmentation(
            TOTAL_BLOCKS, reserved_space=METADATA_BLOCKS
        ),
    }
    return ApplicationOutcome(
        results,
        final_state,
        {"commands": len(commands), "application_calls": 2 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic extent churn and attest the timed disk instance."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    disk = DiskBlockAllocator(
        TOTAL_BLOCKS,
        block_size=BLOCK_SIZE,
        metadata_blocks=METADATA_BLOCKS,
    )

    def execute() -> tuple[tuple[object, None], ...]:
        outcomes: list[tuple[object, None]] = []
        for file_id, block_count in commands:
            extent = disk.allocate_extent(file_id, block_count)
            disk.free_extent(extent, file_id=file_id)
            outcomes.append((extent, None))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, None], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(disk.snapshot()),
            {"commands": operations, "application_calls": 2 * operations},
        )

    return run_application_case(
        scenario_id="disk-block-allocation",
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
