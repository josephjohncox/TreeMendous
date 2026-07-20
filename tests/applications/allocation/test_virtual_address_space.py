"""Virtual address-space application contracts."""

import pytest

from tests.oracles.applications.allocation.virtual_address_space import fixed_layout
from treemendous.applications._shared.allocation import AllocationUnavailableError
from treemendous.applications.allocation.virtual_address import VirtualAddressSpace


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
