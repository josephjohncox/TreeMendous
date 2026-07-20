"""Database page application contracts."""

from dataclasses import replace

import pytest

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


def test_restore_rejects_forged_metadata_reservation_atomically() -> None:
    pages = DatabasePageAllocator(10, metadata_pages=2)
    checkpoint = pages.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(),
        free_ranges=(Span(0, 10),),
    )
    before = pages.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        pages.restore(replace(checkpoint, allocator=forged_allocator))

    assert pages.snapshot() == before
    assert pages.allocate_pages("table", 1).pages == Span(2, 3)


@pytest.mark.parametrize("history", [(Span(-100, 100),), (Span(0, 1),)])
def test_restore_rejects_invalid_freed_history_atomically(
    history: tuple[Span, ...],
) -> None:
    pages = DatabasePageAllocator(10, metadata_pages=1)
    checkpoint = pages.checkpoint()
    before = pages.snapshot()

    with pytest.raises(ValueError, match="freed-page history"):
        pages.restore(replace(checkpoint, freed_history=history))

    assert pages.snapshot() == before


def test_restore_rejects_reuse_counter_below_live_reused_extents() -> None:
    pages = DatabasePageAllocator(10, metadata_pages=1)
    first = pages.allocate_pages("first", 2)
    pages.free_pages(first, table_id="first")
    reused = pages.allocate_pages("reused", 1)
    assert reused.reused
    checkpoint = pages.checkpoint()
    before = pages.snapshot()

    with pytest.raises(ValueError, match="reuse_allocations"):
        pages.restore(replace(checkpoint, reuse_allocations=0))

    assert pages.snapshot() == before
