"""Recycle a freed database page extent."""

from treemendous.applications.allocation.database_pages import DatabasePageAllocator


def main() -> None:
    pages = DatabasePageAllocator(128, tablespace="orders", metadata_pages=2)
    old = pages.allocate_pages("staging", 8)
    pages.free_pages(old, table_id="staging")
    reused = pages.allocate_pages("orders", 4)
    print("pages", reused.pages, "reused", reused.reused)
    print("reuse allocations", pages.snapshot().diagnostics.reuse_allocations)


if __name__ == "__main__":
    main()
