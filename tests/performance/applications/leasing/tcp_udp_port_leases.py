"""Smoke benchmark for real port acquisitions and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.tcp_udp_ports import PortLeaseEngine


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Acquire and expire one real TCP lease per iteration."""
    clock = LogicalClock()
    engine = PortLeaseEngine(clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.acquire("tcp", f"service-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
