"""Independent deterministic-input and crash-state oracle for fuzzing."""

from __future__ import annotations

import hashlib
import random

CrashSpec = tuple[str, int, bytes, str, str]


def generated(seed: int, ordinal: int, maximum: int) -> bytes:
    """Generate the bytes assigned to one ordinal without using the engine."""
    source = random.Random((seed << 32) ^ ordinal)
    return bytes(source.randrange(256) for _ in range(source.randrange(maximum + 1)))


def signature(exc: Exception) -> str:
    """Return the stable external crash identity used by the scenario."""
    identity = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def expected_crashes(cases: int, seed: int, maximum: int) -> tuple[CrashSpec, ...]:
    """Execute the benchmark target rule independently over generated inputs."""
    crashes: dict[str, CrashSpec] = {}
    for ordinal in range(cases):
        data = generated(seed, ordinal, maximum)
        if len(data) != 5:
            continue
        exc = RuntimeError("five")
        crash_signature = signature(exc)
        crashes.setdefault(
            crash_signature,
            (crash_signature, ordinal, data, "RuntimeError", "five"),
        )
    return tuple(sorted(crashes.values()))
