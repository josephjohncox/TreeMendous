"""Attested benchmark for deterministic search-index merging."""

from __future__ import annotations

from tests.oracles.applications.partitioning.index_merge import expected_index
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.index_merge import (
    IndexMergeEngine,
    TermPostings,
)

_DEFAULT_OPERATIONS = 180
_MAX_OPERATIONS = 750
_DEFAULT_SEED = 37
_BAND_SIZE = 7
_SEGMENTS = 5


def _postings_tuple(item: TermPostings) -> tuple[str, tuple[int, ...]]:
    return item.term, item.postings


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Merge and attest every term from deterministic overlapping segments."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    segments = tuple(
        {
            f"term-{term:04}": tuple(
                range(
                    (segment * 3 + term + seed) % 17,
                    operations * 3 + 17,
                    11 + segment,
                )
            )
            for term in range(operations)
            if (term + segment + seed) % 4 != 0
        }
        for segment in range(_SEGMENTS)
    )
    engine = IndexMergeEngine(segments)

    def execute() -> tuple[TermPostings, ...]:
        return engine.run(band_size=_BAND_SIZE)

    def observe(raw: tuple[TermPostings, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        merged = tuple(_postings_tuple(item) for item in raw)
        state_merged = tuple(_postings_tuple(item) for item in snapshot.merged)
        return ApplicationOutcome(
            results=merged,
            final_state={
                "terms": snapshot.terms,
                "merged": state_merged,
            },
            counters={
                "terms_merged": len(snapshot.merged),
                "source_segments": len(segments),
                "merge_bands": (operations + _BAND_SIZE - 1) // _BAND_SIZE,
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        merged = expected_index(segments)
        return ApplicationOutcome(
            results=merged,
            final_state={
                "terms": tuple(term for term, _ in merged),
                "merged": merged,
            },
            counters={
                "terms_merged": len(merged),
                "source_segments": len(segments),
                "merge_bands": (operations + _BAND_SIZE - 1) // _BAND_SIZE,
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="distributed-index-merge",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
