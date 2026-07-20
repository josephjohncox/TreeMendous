"""Database page application contracts."""

from tests.oracles.applications.allocation.database_page_allocation import allocate
from treemendous.applications.allocation.database_pages import DatabasePageAllocator
from treemendous.domain import Span


def test_tablespace_extents_reuse_freed_pages_first() -> None:
    pages = DatabasePageAllocator(20, metadata_pages=2, tablespace="orders")
    expected = allocate(20, {0, 1}, 4)
    old = pages.allocate_pages("table-a", 4)
    assert old.pages == expected == Span(2, 6)
    pages.allocate_pages("table-b", 3)
    pages.free_pages(old, table_id="table-a")
    reused = pages.allocate_pages("table-c", 2)
    assert reused.pages == Span(2, 4)
    assert reused.reused
    assert pages.snapshot().diagnostics.reuse_allocations == 1


def test_checkpoint_restores_reuse_history_and_geometry() -> None:
    pages = DatabasePageAllocator(10, metadata_pages=1)
    extent = pages.allocate_pages("a", 2)
    pages.free_pages(extent, table_id="a")
    checkpoint = pages.checkpoint()
    pages.allocate_pages("b", 2)
    pages.restore(checkpoint)
    snapshot = pages.snapshot()
    assert not snapshot.extents
    assert snapshot.diagnostics.pages_ever_freed == 2
