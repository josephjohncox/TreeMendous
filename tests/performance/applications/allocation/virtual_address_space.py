"""Actual virtual map/unmap smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.virtual_address import VirtualAddressSpace


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    space = VirtualAddressSpace(1024)
    started = perf_counter()
    for index in range(operations):
        mapping = space.map(4096 + index % 4096, owner=index, guard_pages=1)
        space.unmap(mapping, owner=index)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
