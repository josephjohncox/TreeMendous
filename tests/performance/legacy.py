"""Failure helper for retired, semantically invalid benchmark entry points."""

from __future__ import annotations


def fail_legacy(script: str, replacement: str = "protocol_benchmark.py") -> int:
    """Fail loudly rather than emit non-oracle-validated measurements."""
    print(
        f"{script} was retired because it did not validate identical ordered traces. "
        f"Use tests/performance/{replacement}; no measurement was recorded."
    )
    return 2
