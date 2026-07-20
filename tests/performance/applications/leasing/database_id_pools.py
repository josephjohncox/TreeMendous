"""Smoke benchmark for real monotonic ID acquisition and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.database_ids import DatabaseIdPool


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Acquire and expire one real monotonic ID batch per iteration."""
    clock = LogicalClock()
    engine = DatabaseIdPool(maximum_id=max(iterations, 1), clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.acquire(f"writer-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
