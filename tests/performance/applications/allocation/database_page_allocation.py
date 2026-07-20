"""Actual database page recycle smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.database_pages import DatabasePageAllocator


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    pages = DatabasePageAllocator(4096, metadata_pages=8)
    started = perf_counter()
    for index in range(operations):
        extent = pages.allocate_pages(index, 1 + index % 4)
        pages.free_pages(extent, table_id=index)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
