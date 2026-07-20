"""Attested benchmark for halo-aware byte regex scanning."""

from __future__ import annotations

from tests.oracles.applications.partitioning.regex_scan import expected_matches
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.regex_scan import (
    RegexMatch,
    RegexScanEngine,
)

_DEFAULT_OPERATIONS = 240
_MAX_OPERATIONS = 1_500
_DEFAULT_SEED = 47
_PATTERN = b"needle"
_HALO = len(_PATTERN)
_CHUNK_SIZE = 17


def _match_tuple(match: RegexMatch) -> tuple[int, int, bytes]:
    return match.start, match.end, match.value


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Scan a bounded byte buffer and attest every ordered match and offset."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    data = (
        b"".join(
            f"{(index * 17 + seed) % 10_000:04}|".encode() + _PATTERN + b"|"
            for index in range(operations)
        )
        + b"tail"
    )
    engine = RegexScanEngine(data, _PATTERN, halo=_HALO)

    def execute() -> tuple[RegexMatch, ...]:
        return engine.run(chunk_size=_CHUNK_SIZE)

    def observe(raw: tuple[RegexMatch, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        matches = tuple(_match_tuple(match) for match in raw)
        state_matches = tuple(_match_tuple(match) for match in snapshot.matches)
        return ApplicationOutcome(
            results=matches,
            final_state={
                "halo": snapshot.halo,
                "matches": state_matches,
            },
            counters={
                "core_bytes_scanned": len(data),
                "chunks_scanned": (len(data) + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
                "matches": len(snapshot.matches),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        matches = expected_matches(data, _PATTERN)
        return ApplicationOutcome(
            results=matches,
            final_state={
                "halo": _HALO,
                "matches": matches,
            },
            counters={
                "core_bytes_scanned": len(data),
                "chunks_scanned": (len(data) + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
                "matches": len(matches),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.regex_scan",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
