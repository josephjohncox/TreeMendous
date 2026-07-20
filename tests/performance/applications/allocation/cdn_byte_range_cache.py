"""Attested CDN residency, coverage, and eviction application benchmark."""

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
from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache

DEFAULT_OPERATIONS = 100
OBJECT_SIZE = 512
OBJECT_ID = "benchmark-object"


def _commands(operations: int, seed: int) -> tuple[tuple[int, int, str], ...]:
    rng = random.Random(seed)
    commands: list[tuple[int, int, str]] = []
    for index in range(operations):
        length = rng.randint(1, 64)
        start = rng.randint(0, OBJECT_SIZE - length)
        commands.append((start, length, f"segment-{index}"))
    return tuple(commands)


def _expected_segment(
    allocation_id: int, start: int, length: int, cache_key: str
) -> dict[str, Any]:
    return {
        "handle": handle(allocation_id, OBJECT_ID, start, start + length),
        "cache_key": cache_key,
    }


def _oracle(commands: tuple[tuple[int, int, str], ...]) -> ApplicationOutcome:
    results = []
    for index, (start, length, cache_key) in enumerate(commands):
        segment = _expected_segment(index + 1, start, length, cache_key)
        coverage = {
            "requested": span(start, start + length),
            "resident_ranges": (span(start, start + length),),
            "missing_ranges": (),
            "covered_bytes": length,
        }
        results.append((segment, coverage, None))
    final_state = {
        "object_id": OBJECT_ID,
        "object_size": OBJECT_SIZE,
        "segments": (),
        "missing_ranges": (span(0, OBJECT_SIZE),),
        "diagnostics": {
            "resident_bytes": 0,
            "evictions": len(commands),
            "fragmentation": empty_fragmentation(OBJECT_SIZE),
        },
    }
    return ApplicationOutcome(
        tuple(results),
        final_state,
        {"commands": len(commands), "application_calls": 3 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic cache churn and attest the timed cache instance."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    cache = CDNByteRangeCache(OBJECT_SIZE, object_id=OBJECT_ID)

    def execute() -> tuple[tuple[object, object, None], ...]:
        outcomes: list[tuple[object, object, None]] = []
        for start, length, cache_key in commands:
            segment = cache.cache_segment(start, length, cache_key=cache_key)
            coverage = cache.request_coverage(start, length)
            cache.evict(segment)
            outcomes.append((segment, coverage, None))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, object, None], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(cache.snapshot()),
            {"commands": operations, "application_calls": 3 * operations},
        )

    return run_application_case(
        scenario_id="cdn-byte-range-cache",
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
