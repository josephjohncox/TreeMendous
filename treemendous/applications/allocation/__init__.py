"""Reusable allocation and free-space application engines.

Exports are resolved lazily so importing the data-only family manifest does not
also import every application engine.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from treemendous.applications.allocation.cdn_cache import (
        CDNByteRangeCache as CDNByteRangeCache,
    )
    from treemendous.applications.allocation.database_pages import (
        DatabasePageAllocator as DatabasePageAllocator,
    )
    from treemendous.applications.allocation.disk_blocks import (
        DiskBlockAllocator as DiskBlockAllocator,
    )
    from treemendous.applications.allocation.gpu_arena import (
        GPUMemoryArena as GPUMemoryArena,
    )
    from treemendous.applications.allocation.heap import HeapAllocator as HeapAllocator
    from treemendous.applications.allocation.multipart_upload import (
        MultipartUploadTracker as MultipartUploadTracker,
    )
    from treemendous.applications.allocation.ring_buffer import RingBuffer as RingBuffer
    from treemendous.applications.allocation.virtual_address import (
        VirtualAddressSpace as VirtualAddressSpace,
    )

_EXPORTS = {
    "CDNByteRangeCache": ("cdn_cache", "CDNByteRangeCache"),
    "DatabasePageAllocator": ("database_pages", "DatabasePageAllocator"),
    "DiskBlockAllocator": ("disk_blocks", "DiskBlockAllocator"),
    "GPUMemoryArena": ("gpu_arena", "GPUMemoryArena"),
    "HeapAllocator": ("heap", "HeapAllocator"),
    "MultipartUploadTracker": ("multipart_upload", "MultipartUploadTracker"),
    "RingBuffer": ("ring_buffer", "RingBuffer"),
    "VirtualAddressSpace": ("virtual_address", "VirtualAddressSpace"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public engine export without eager family imports."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value
