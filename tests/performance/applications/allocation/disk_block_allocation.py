"""Actual filesystem extent churn smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    disk = DiskBlockAllocator(4096, metadata_blocks=8)
    started = perf_counter()
    for index in range(operations):
        extent = disk.allocate_extent(index, 1 + index % 8)
        disk.free_extent(extent, file_id=index)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
