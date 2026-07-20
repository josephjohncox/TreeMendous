"""Attested virtual map/unmap application benchmark."""

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
from treemendous.applications.allocation.virtual_address import VirtualAddressSpace

DEFAULT_OPERATIONS = 100
TOTAL_PAGES = 128
PAGE_SIZE = 4096


def _commands(operations: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    rng = random.Random(seed)
    return tuple(
        (index, rng.randint(1, 8 * PAGE_SIZE), rng.randint(0, 2))
        for index in range(operations)
    )


def _expected_mapping(
    allocation_id: int, owner: int, length: int, guard_pages: int
) -> dict[str, Any]:
    mapped_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    reserved_pages = mapped_pages + 2 * guard_pages
    return {
        "handle": handle(allocation_id, owner, 0, reserved_pages),
        "owner": owner,
        "address": guard_pages * PAGE_SIZE,
        "mapped_length": mapped_pages * PAGE_SIZE,
        "requested_length": length,
        "page_size": PAGE_SIZE,
        "guard_pages": guard_pages,
        "movable": True,
    }


def _oracle(commands: tuple[tuple[int, int, int], ...]) -> ApplicationOutcome:
    results = tuple(
        (_expected_mapping(index + 1, owner, length, guard_pages), None)
        for index, (owner, length, guard_pages) in enumerate(commands)
    )
    final_state = {
        "page_size": PAGE_SIZE,
        "mappings": (),
        "free_page_ranges": (span(0, TOTAL_PAGES),),
        "diagnostics": empty_fragmentation(TOTAL_PAGES),
    }
    return ApplicationOutcome(
        results,
        final_state,
        {"commands": len(commands), "application_calls": 2 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic mapping churn and attest the timed address space."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    space = VirtualAddressSpace(TOTAL_PAGES, page_size=PAGE_SIZE)

    def execute() -> tuple[tuple[object, None], ...]:
        outcomes: list[tuple[object, None]] = []
        for owner, length, guard_pages in commands:
            mapping = space.map(length, owner=owner, guard_pages=guard_pages)
            space.unmap(mapping, owner=owner)
            outcomes.append((mapping, None))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, None], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(space.snapshot()),
            {"commands": operations, "application_calls": 2 * operations},
        )

    return run_application_case(
        scenario_id="virtual-address-space",
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
