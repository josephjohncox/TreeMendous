"""Smoke benchmark for real phone extension acquisition and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.phone_extensions import PhoneExtensionPool


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Acquire and expire one real extension per iteration."""
    clock = LogicalClock()
    engine = PhoneExtensionPool(clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.acquire(f"endpoint-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
