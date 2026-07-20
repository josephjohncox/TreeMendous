"""Smoke benchmark for real software seat checkout and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.software_seats import SoftwareSeatPool


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Checkout and expire one real product seat per iteration."""
    clock = LogicalClock()
    engine = SoftwareSeatPool({"editor": 8}, clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.checkout("editor", f"client-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
