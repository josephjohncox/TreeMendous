"""Allocate and release an aligned guarded heap block."""

from treemendous.applications.allocation.heap import HeapAllocator


def main() -> None:
    heap = HeapAllocator(256, header_size=8, redzone_size=4)
    block = heap.allocate(48, owner="request-7", alignment=32)
    print("payload", block.payload, "raw", block.raw_handle.span)
    heap.free(block, owner="request-7")
    print("free bytes", heap.snapshot().diagnostics.total_free)


if __name__ == "__main__":
    main()
