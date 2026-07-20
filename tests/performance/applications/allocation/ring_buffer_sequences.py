"""Actual ring produce/consume smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.ring_buffer import RingBuffer


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    ring = RingBuffer(64, sequence_modulus=256)
    started = perf_counter()
    for _ in range(operations):
        ring.produce(8)
        ring.consume(8)
    return {"operations": operations, "seconds": perf_counter() - started}


if __name__ == "__main__":
    print(run_smoke())
