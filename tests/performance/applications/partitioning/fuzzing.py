"""Attested benchmark for deterministic generated-input fuzzing."""

from __future__ import annotations

from tests.oracles.applications.partitioning.fuzzing import (
    benchmark_input,
    benchmark_signature,
    expected_crashes,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.fuzzing import Crash, FuzzingEngine

_DEFAULT_OPERATIONS = 320
_MAX_OPERATIONS = 2_000
_DEFAULT_SEED = 11
_MAX_INPUT_SIZE = 16
_SHARD_SIZE = 29


def _target(data: bytes) -> None:
    if len(data) == 5:
        raise RuntimeError("five")


def _crash_tuple(crash: Crash) -> tuple[str, int, bytes, str, str]:
    return (
        crash.signature,
        crash.ordinal,
        crash.input,
        crash.exception_type,
        crash.message,
    )


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Run deterministic cases, including one real abandoned-claim retry."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    engine = FuzzingEngine(
        _target,
        cases=operations,
        seed=seed,
        max_input_size=_MAX_INPUT_SIZE,
        input_provider=benchmark_input,
        signature_provider=benchmark_signature,
    )

    def execute() -> tuple[Crash, ...]:
        return engine.run(shard_size=_SHARD_SIZE, fail_first_claim=True)

    def observe(raw: tuple[Crash, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        crashes = tuple(_crash_tuple(crash) for crash in raw)
        state_crashes = tuple(_crash_tuple(crash) for crash in snapshot.crashes)
        successful_claims = (operations + _SHARD_SIZE - 1) // _SHARD_SIZE
        return ApplicationOutcome(
            results=crashes,
            final_state={
                "executed_ordinals": snapshot.executed_ordinals,
                "crashes": state_crashes,
                "retries": snapshot.retries,
            },
            counters={
                "target_calls": len(snapshot.executed_ordinals),
                "successful_claims": successful_claims,
                "abandoned_claims": snapshot.retries,
                "unique_crashes": len(snapshot.crashes),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        crashes = expected_crashes(operations)
        successful_claims = (operations + _SHARD_SIZE - 1) // _SHARD_SIZE
        return ApplicationOutcome(
            results=crashes,
            final_state={
                "executed_ordinals": tuple(range(operations)),
                "crashes": crashes,
                "retries": 1,
            },
            counters={
                "target_calls": operations,
                "successful_claims": successful_claims,
                "abandoned_claims": 1,
                "unique_crashes": len(crashes),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="distributed-fuzzing",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
