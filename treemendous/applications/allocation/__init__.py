"""Reusable allocation and free-space application engines.

Each engine adds domain invariants to the private contiguous-allocation or
modular-sequence kernel.  Import concrete classes from their scenario module;
the package exports the common entry points for convenience.
"""

from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache
from treemendous.applications.allocation.database_pages import DatabasePageAllocator
from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator
from treemendous.applications.allocation.gpu_arena import GPUMemoryArena
from treemendous.applications.allocation.heap import HeapAllocator
from treemendous.applications.allocation.multipart_upload import MultipartUploadTracker
from treemendous.applications.allocation.ring_buffer import RingBuffer
from treemendous.applications.allocation.virtual_address import VirtualAddressSpace

__all__ = [
    "CDNByteRangeCache",
    "DatabasePageAllocator",
    "DiskBlockAllocator",
    "GPUMemoryArena",
    "HeapAllocator",
    "MultipartUploadTracker",
    "RingBuffer",
    "VirtualAddressSpace",
]
