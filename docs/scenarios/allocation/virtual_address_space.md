# Virtual address-space management

`VirtualAddressSpace` manages page-granular mappings over an explicit virtual
page domain. The configured base address and every fixed mapping address must
be page aligned. Byte lengths are rounded upward to pages. Symmetric guard pages
surround the mapped payload and remain inaccessible until unmap.

## Fixed and movable mappings

`map(length, owner=..., address=...)` creates a fixed-address reservation when
`address` is supplied; otherwise it chooses the first suitable page extent.
The independent `movable` flag controls whether `relocate` may replace the
mapping. Relocation can choose another address or accept an exact page-aligned
target. It checkpoints allocator state before releasing the source; if the
target conflicts or is outside the managed domain, the original mapping and
all metadata are restored.

`VirtualMapping.byte_range` describes the page-rounded accessible region.
`reserved_pages` includes guards and is intentionally expressed in page IDs.
`unmap` validates both the exact mapping object and owner. Snapshots expose
mapping order, free page geometry, and page fragmentation. Checkpoint restore
checks page size, guard arithmetic, owner identity, and a one-to-one mapping to
allocator handles before mutation.

```python
from treemendous.applications.allocation.virtual_address import VirtualAddressSpace

space = VirtualAddressSpace(4096, page_size=4096)
mapping = space.map(6000, owner="pid-7", guard_pages=1)
mapping = space.relocate(mapping, owner="pid-7", address=100 * 4096)
space.unmap(mapping, owner="pid-7")
```

The executable example demonstrates relocation. The independent oracle derives
guard and payload spans arithmetically, while the actual-work smoke repeatedly
maps and unmaps page-rounded regions.
