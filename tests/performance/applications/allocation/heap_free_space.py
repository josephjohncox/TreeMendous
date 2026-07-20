"""Actual allocate/free heap smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.heap import HeapAllocator


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    heap = HeapAllocator(4096, header_size=8, redzone_size=4)
    started = perf_counter()
    for index in range(operations):
        block = heap.allocate(32 + index % 32, owner=index, alignment=16)
        heap.free(block, owner=index)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
