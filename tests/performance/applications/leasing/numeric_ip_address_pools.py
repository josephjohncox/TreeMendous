"""Smoke benchmark for real numeric IP acquisitions and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.numeric_ip_pools import NumericIPAddressPool


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Acquire and expire one real IPv4 address per iteration."""
    clock = LogicalClock()
    engine = NumericIPAddressPool("198.51.100.0/24", clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.acquire(f"client-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
