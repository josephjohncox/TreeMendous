"""Allocate contiguous filesystem blocks around reserved metadata."""

from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator


def main() -> None:
    disk = DiskBlockAllocator(100, block_size=4096, metadata_blocks=4)
    extent = disk.allocate_extent("photos.db", 12)
    print("blocks", extent.blocks, "bytes", extent.byte_range)
    disk.free_extent(extent, file_id="photos.db")
    print("largest free extent", disk.snapshot().diagnostics.largest_free_chunk)


if __name__ == "__main__":
    main()
