"""Smoke benchmark for real VLAN tag acquisition and expiry."""

from __future__ import annotations

from time import perf_counter

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.vlan_tags import VlanTagPool


def run_smoke(iterations: int = 100) -> dict[str, int | float]:
    """Acquire and expire one real scoped VLAN tag per iteration."""
    clock = LogicalClock()
    engine = VlanTagPool(("edge",), clock=clock)
    started = perf_counter()
    for index in range(iterations):
        engine.acquire("edge", f"controller-{index}", ttl=1)
        clock.advance()
        engine.expire()
    return {"iterations": iterations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
