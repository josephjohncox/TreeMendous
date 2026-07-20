"""Smoke benchmark for filesystem lock conflict queries."""

from time import perf_counter

from treemendous.applications.catalogs.filesystem_byte_locks import FilesystemByteLocks


def run_smoke(iterations: int = 500) -> float:
    locks = FilesystemByteLocks()
    for index in range(200):
        locks.acquire("data", f"reader-{index}", index * 8, index * 8 + 4, "shared")
    started = perf_counter()
    for index in range(iterations):
        locks.conflicts("data", "writer", index % 1500, index % 1500 + 16, "exclusive")
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
