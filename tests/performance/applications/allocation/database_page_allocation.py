"""Attested database page recycle application benchmark."""

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
from treemendous.applications.allocation.database_pages import DatabasePageAllocator

DEFAULT_OPERATIONS = 100
TOTAL_PAGES = 128
METADATA_PAGES = 8
PAGE_SIZE = 8192
TABLESPACE = "benchmark"


def _commands(operations: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = random.Random(seed)
    return tuple((index, rng.randint(1, 16)) for index in range(operations))


def _expected_extent(
    allocation_id: int, table_id: int, page_count: int, reused: bool
) -> dict[str, Any]:
    return {
        "handle": handle(
            allocation_id,
            table_id,
            METADATA_PAGES,
            METADATA_PAGES + page_count,
        ),
        "table_id": table_id,
        "tablespace": TABLESPACE,
        "reused": reused,
    }


def _oracle(commands: tuple[tuple[int, int], ...]) -> ApplicationOutcome:
    results = tuple(
        (_expected_extent(index + 1, table_id, page_count, index > 0), None)
        for index, (table_id, page_count) in enumerate(commands)
    )
    final_state = {
        "tablespace": TABLESPACE,
        "page_size": PAGE_SIZE,
        "extents": (),
        "free_pages": (span(METADATA_PAGES, TOTAL_PAGES),),
        "diagnostics": {
            "fragmentation": empty_fragmentation(
                TOTAL_PAGES, reserved_space=METADATA_PAGES
            ),
            "reuse_allocations": len(commands) - 1,
            "pages_ever_freed": max(page_count for _, page_count in commands),
        },
    }
    return ApplicationOutcome(
        results,
        final_state,
        {"commands": len(commands), "application_calls": 2 * len(commands)},
    )


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = 42
) -> ApplicationSample:
    """Run deterministic page recycling and attest the timed allocator state."""
    validate_inputs(operations, seed)
    commands = _commands(operations, seed)
    pages = DatabasePageAllocator(
        TOTAL_PAGES,
        tablespace=TABLESPACE,
        page_size=PAGE_SIZE,
        metadata_pages=METADATA_PAGES,
    )

    def execute() -> tuple[tuple[object, None], ...]:
        outcomes: list[tuple[object, None]] = []
        for table_id, page_count in commands:
            extent = pages.allocate_pages(table_id, page_count)
            pages.free_pages(extent, table_id=table_id)
            outcomes.append((extent, None))
        return tuple(outcomes)

    def observe(raw: tuple[tuple[object, None], ...]) -> ApplicationOutcome:
        return ApplicationOutcome(
            stable_evidence(raw),
            stable_evidence(pages.snapshot()),
            {"commands": operations, "application_calls": 2 * operations},
        )

    return run_application_case(
        scenario_id="database-page-allocation",
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
