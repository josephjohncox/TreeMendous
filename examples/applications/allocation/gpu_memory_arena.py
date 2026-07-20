"""Reclaim a GPU buffer after its owning stream completes."""

from treemendous.applications.allocation.gpu_arena import GPUMemoryArena


def main() -> None:
    arena = GPUMemoryArena(1 << 20, device_alignment=256)
    buffer = arena.allocate(64_000, stream="render", alignment=512)
    arena.defer_free(buffer, stream="render", completion_epoch=5)
    print("deferred", arena.snapshot().deferred_frees)
    arena.advance_completion("render", 5)
    print("free bytes", arena.snapshot().diagnostics.total_free)


if __name__ == "__main__":
    main()
