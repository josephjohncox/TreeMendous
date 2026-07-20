"""Fixed injected-input and crash evidence for the fuzzing benchmark."""

from __future__ import annotations

CrashSpec = tuple[str, int, bytes, str, str]

_CRASH_ORDINAL = 3
_CRASH_INPUT = b"five!"
_CRASH_SIGNATURE = "benchmark-runtime-five"


def benchmark_input(ordinal: int) -> bytes:
    """Return a fixed benchmark vector without reproducing input generation."""
    return _CRASH_INPUT if ordinal == _CRASH_ORDINAL else b"ok"


def benchmark_signature(_: Exception) -> str:
    """Return the injected fixed identity used by the benchmark."""
    return _CRASH_SIGNATURE


def expected_crashes(cases: int) -> tuple[CrashSpec, ...]:
    """Return the one fixed crash when its ordinal belongs to the workload."""
    if cases <= _CRASH_ORDINAL:
        return ()
    return (
        (
            _CRASH_SIGNATURE,
            _CRASH_ORDINAL,
            _CRASH_INPUT,
            "RuntimeError",
            "five",
        ),
    )
