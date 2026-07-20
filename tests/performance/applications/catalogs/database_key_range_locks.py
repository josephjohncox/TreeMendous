"""Smoke benchmark for encoded database key lock queries."""

from time import perf_counter

from treemendous.applications.catalogs.database_key_range_locks import (
    DatabaseKeyRangeLocks,
)


def run_smoke(iterations: int = 500) -> float:
    locks = DatabaseKeyRangeLocks()
    for index in range(100):
        locks.acquire(
            "table", f"reader-{index}", f"k{index:04d}", f"k{index + 1:04d}", "shared"
        )
    started = perf_counter()
    for index in range(iterations):
        locks.conflicts(
            "table",
            "writer",
            f"k{index % 99:04d}",
            f"k{index % 99 + 2:04d}",
            "exclusive",
        )
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
