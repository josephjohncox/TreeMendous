"""Actual stream-deferred GPU arena smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.gpu_arena import GPUMemoryArena


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    arena = GPUMemoryArena(4096)
    started = perf_counter()
    for epoch in range(operations):
        buffer = arena.allocate(128, stream="stream")
        arena.defer_free(buffer, stream="stream", completion_epoch=epoch)
        arena.advance_completion("stream", epoch)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
