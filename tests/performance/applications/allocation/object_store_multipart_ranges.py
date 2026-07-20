"""Attested object-store multipart completion application benchmark."""

from __future__ import annotations

import random
from typing import Any

from tests.performance.applications.allocation._shared import (
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
from treemendous.applications.allocation.multipart_upload import MultipartUploadTracker

DEFAULT_OPERATIONS = 100
PART_SIZE = 8
UPLOAD_ID = "benchmark-upload"


def _commands(operations: int, seed: int) -> tuple[int, ...]:
    part_numbers = list(range(1, operations + 1))
    random.Random(seed).shuffle(part_numbers)
    return tuple(part_numbers)


def _expected_part(part_number: int, allocation_id: int) -> dict[str, Any]:
    start = (part_number - 1) * PART_SIZE
    end = start + PART_SIZE
    return {
        "part_number": part_number,
        "etag": f"etag-{part_number}",
        "byte_range": span(start, end),
        "attempt": 1,
        "handle": handle(allocation_id, UPLOAD_ID, start, end),
    }


def _oracle(commands: tuple[int, ...]) -> ApplicationOutcome:
    allocation_ids = {
        part_number: index + 1 for index, part_number in enumerate(commands)
    }
    object_size = len(commands) * PART_SIZE
    results = tuple(
        _expected_part(part_number, allocation_ids[part_number])
        for part_number in commands
    )
    final_state = {
        "upload_id": UPLOAD_ID,
        "object_size": object_size,
        "part_size": PART_SIZE,
        "completed": tuple(
            _expected_part(part_number, allocation_ids[part_number])
            for part_number in range(1, len(commands) + 1)
        ),
        "missing_ranges": (),
        "contiguous_completion": span(0, object_size),
        "diagnostics": {
            "total_parts": len(commands),
            "completed_parts": len(commands),
            "missing_bytes": 0,
            "retry_count": 0,
        },
    }
    return ApplicationOutcome(
        results,
        final_state,
        {"commands": len(commands), "application_calls": len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic part completion and attest the timed upload state."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    upload = MultipartUploadTracker(
        operations * PART_SIZE, PART_SIZE, upload_id=UPLOAD_ID
    )

    def execute() -> tuple[object, ...]:
        return tuple(
            upload.complete_part(part_number, f"etag-{part_number}")
            for part_number in commands
        )

    def observe(raw: tuple[object, ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(upload.snapshot()),
            {"commands": operations, "application_calls": operations},
        )

    return run_application_case(
        scenario_id="object-store-multipart-ranges",
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
