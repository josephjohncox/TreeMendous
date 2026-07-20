"""Produce across a modular wrap under overwrite policy."""

from treemendous.applications.allocation.ring_buffer import FullPolicy, RingBuffer


def main() -> None:
    ring = RingBuffer(
        4, sequence_modulus=8, initial_sequence=6, full_policy=FullPolicy.OVERWRITE
    )
    ring.produce(3, epoch_hint=0)
    result = ring.produce(3, epoch_hint=1)
    snapshot = ring.snapshot()
    print("produced", result.sequences, "overwritten", result.overwritten)
    print("producer", snapshot.producer_cursor, "consumer", snapshot.consumer_cursor)


if __name__ == "__main__":
    main()
