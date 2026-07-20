# Disk block allocation

`DiskBlockAllocator` models one filesystem block device. Coordinates are block
IDs, not bytes. `block_size` converts an extent to its physical byte range, and
the leading `metadata_blocks` are permanently reserved before user allocation.
Every `FileExtent` is contiguous; a file may own more than one extent and
`extents_for(file_id)` returns them in physical order.

## Allocation lifecycle

`allocate_extent(file_id, block_count)` applies first, best, or worst fit to
free block chunks. Counts must be positive whole blocks. `free_extent` requires
the exact live extent and matching file identity. A wrong file, stale extent,
or overlapping allocation fails without changing free-space geometry. Freed
adjacent ranges coalesce in the shared allocator and become eligible for later
file extents.

Snapshots distinguish reserved metadata, live file extents, and free extents.
Fragmentation metrics are measured in blocks: `largest_free_chunk` therefore
answers the largest immediately allocatable contiguous extent, while multiplying
by `block_size` gives byte capacity. Checkpoints include both allocator handles
and file metadata; restore validates their one-to-one relationship before
commit.

```python
from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator

disk = DiskBlockAllocator(1_000_000, block_size=4096, metadata_blocks=128)
extent = disk.allocate_extent("inode-42", 32)
print(extent.blocks, extent.byte_range)
disk.free_extent(extent, file_id="inode-42")
```

The executable example lives at
`examples/applications/allocation/disk_block_allocation.py`. The focused oracle
uses an independent block bitmap, and the smoke benchmark executes real extent
allocation and release rather than a generic range trace.
