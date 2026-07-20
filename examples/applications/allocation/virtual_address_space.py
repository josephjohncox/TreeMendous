"""Map and relocate guarded virtual pages."""

from treemendous.applications.allocation.virtual_address import VirtualAddressSpace


def main() -> None:
    space = VirtualAddressSpace(64, page_size=4096)
    mapping = space.map(6000, owner="worker", guard_pages=1)
    moved = space.relocate(mapping, owner="worker", address=16 * 4096)
    print("mapping", moved.byte_range, "reserved pages", moved.reserved_pages)


if __name__ == "__main__":
    main()
