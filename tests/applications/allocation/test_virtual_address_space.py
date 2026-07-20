"""Virtual address-space application contracts."""

from dataclasses import replace

import pytest

from tests.oracles.applications.allocation.virtual_address_space import fixed_layout
from treemendous.applications._shared.allocation import (
    AllocationUnavailableError,
    ForeignAllocationError,
    StaleAllocationError,
)
from treemendous.applications.allocation.virtual_address import VirtualAddressSpace
from treemendous.domain import Span


def test_page_rounding_guards_and_fixed_address() -> None:
    space = VirtualAddressSpace(32, page_size=4096)
    reserved, payload = fixed_layout(4 * 4096, 5000, 4096, 1)
    mapping = space.map(5000, owner="process", address=4 * 4096, guard_pages=1)
    assert mapping.reserved_pages == reserved
    assert mapping.byte_range == payload
    assert mapping.mapped_length == 8192


def test_movable_relocation_rolls_back_on_conflict_and_fixed_rejects() -> None:
    space = VirtualAddressSpace(12, page_size=1024)
    movable = space.map(1024, owner="p", address=2 * 1024, guard_pages=1)
    space.map(1024, owner="other", address=6 * 1024, guard_pages=1)
    before = space.snapshot()
    with pytest.raises(AllocationUnavailableError):
        space.relocate(movable, owner="p", address=6 * 1024)
    assert space.snapshot() == before
    fixed = space.map(1024, owner="p", address=10 * 1024, guard_pages=0, movable=False)
    with pytest.raises(ValueError, match="fixed"):
        space.relocate(fixed, owner="p")


def test_same_address_relocation_validates_owner_and_liveness() -> None:
    space = VirtualAddressSpace(8, page_size=1024)
    mapping = space.map(1024, owner="owner", address=2 * 1024)
    before = space.snapshot()

    with pytest.raises(ForeignAllocationError):
        space.relocate(mapping, owner="wrong", address=mapping.address)
    assert space.snapshot() == before

    space.unmap(mapping, owner="owner")
    unmapped = space.snapshot()
    with pytest.raises(StaleAllocationError):
        space.relocate(mapping, owner="owner", address=mapping.address)
    assert space.snapshot() == unmapped


def test_restore_rejects_forged_reserved_pages_atomically() -> None:
    space = VirtualAddressSpace(8, page_size=1024, reserved_pages=(Span(0, 2),))
    checkpoint = space.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(),
        free_ranges=(Span(0, 8),),
    )
    before = space.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        space.restore(replace(checkpoint, allocator=forged_allocator))

    assert space.snapshot() == before
