"""Data-only implementation evidence for the eight allocator scenarios."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final


def _evidence(module: str, artifact: str) -> MappingProxyType[str, str]:
    return MappingProxyType(
        {
            "engine": f"treemendous.applications.allocation.{module}:create_application",
            "example": f"examples/applications/allocation/{artifact}.py",
            "oracle": f"tests/oracles/applications/allocation/{artifact}.py",
            "benchmark": f"tests/performance/applications/allocation/{artifact}.py",
            "docs": f"docs/scenarios/allocation/{artifact}.md",
        }
    )


EVIDENCE: Final = MappingProxyType(
    {
        "heap-free-space": _evidence("heap", "heap_free_space"),
        "disk-block-allocation": _evidence("disk_blocks", "disk_block_allocation"),
        "virtual-address-space": _evidence("virtual_address", "virtual_address_space"),
        "database-page-allocation": _evidence(
            "database_pages", "database_page_allocation"
        ),
        "object-store-multipart-ranges": _evidence(
            "multipart_upload", "object_store_multipart_ranges"
        ),
        "cdn-byte-range-cache": _evidence("cdn_cache", "cdn_byte_range_cache"),
        "gpu-memory-arena": _evidence("gpu_arena", "gpu_memory_arena"),
        "ring-buffer-sequences": _evidence("ring_buffer", "ring_buffer_sequences"),
    }
)
