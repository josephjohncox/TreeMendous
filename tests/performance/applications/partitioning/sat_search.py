"""Attested benchmark for exact partitioned CNF SAT search."""

from __future__ import annotations

from tests.oracles.applications.partitioning.sat_search import expected_ordinals
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.sat_search import (
    SatisfyingAssignment,
    SatSearchEngine,
)

_DEFAULT_OPERATIONS = 32
_MAX_OPERATIONS = 256
_DEFAULT_SEED = 53
_VARIABLES = 8
_PREFIX_BITS = 5
_SHARD_SIZE = 4


def _solution_tuple(
    solution: SatisfyingAssignment,
) -> tuple[int, tuple[bool, ...]]:
    return solution.ordinal, solution.values


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Evaluate a bounded seeded CNF and attest every satisfying assignment."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    clauses = tuple(
        (
            (index + seed) % _VARIABLES + 1,
            (index * 3 + seed + 1) % _VARIABLES + 1,
        )
        for index in range(operations)
    )
    engine = SatSearchEngine(
        _VARIABLES,
        clauses,
        prefix_bits=_PREFIX_BITS,
    )

    def execute() -> tuple[SatisfyingAssignment, ...]:
        return engine.run(shard_size=_SHARD_SIZE)

    def observe(raw: tuple[SatisfyingAssignment, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        solutions = tuple(_solution_tuple(item) for item in raw)
        state_solutions = tuple(_solution_tuple(item) for item in snapshot.solutions)
        return ApplicationOutcome(
            results=solutions,
            final_state={
                "variables": snapshot.variables,
                "prefix_bits": snapshot.prefix_bits,
                "solutions": state_solutions,
            },
            counters={
                "clauses": len(clauses),
                "assignments_enumerated": 1 << _VARIABLES,
                "prefixes_evaluated": 1 << _PREFIX_BITS,
                "solutions": len(snapshot.solutions),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        ordinals = expected_ordinals(_VARIABLES, clauses)
        solutions = tuple(
            (
                ordinal,
                tuple(
                    bool(ordinal & (1 << variable)) for variable in range(_VARIABLES)
                ),
            )
            for ordinal in ordinals
        )
        return ApplicationOutcome(
            results=solutions,
            final_state={
                "variables": _VARIABLES,
                "prefix_bits": _PREFIX_BITS,
                "solutions": solutions,
            },
            counters={
                "clauses": len(clauses),
                "assignments_enumerated": 1 << _VARIABLES,
                "prefixes_evaluated": 1 << _PREFIX_BITS,
                "solutions": len(solutions),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="distributed-sat-search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
