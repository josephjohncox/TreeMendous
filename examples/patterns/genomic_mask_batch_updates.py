"""Update genomic mask geometry atomically without genomic record semantics.

This pattern is outside the 50-engine registry. Coordinates are only half-open
signed integers: contig/build identity, strand and feature metadata, coordinate
conversion, durable provenance, validation, and persistence are not supplied.
"""

from treemendous import MutationResult, Span
from treemendous.exact_batch import (
    BatchLimits,
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)


def main() -> None:
    limits = BatchLimits(
        max_operations=4,
        max_live_intervals=4,
        max_changed_spans=8,
        max_result_bytes=2_048,
        max_work_units=32,
    )
    masks = ExactBatchRangeSet((0, 1_000), initially_available=False, limits=limits)
    results = masks.mutate(
        (
            BatchMutation(MutationOpcode.ADD, 100, 140),
            BatchMutation(MutationOpcode.ADD, 120, 180),
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 130, 150),
            BatchMutation(MutationOpcode.ADD, 130, 150),
        )
    )

    assert results == (
        MutationResult((Span(100, 140),), 40, False),
        MutationResult((Span(140, 180),), 40, False),
        MutationResult((Span(130, 150),), 20, True),
        MutationResult((Span(130, 150),), 20, False),
    )
    assert tuple(
        (interval.start, interval.end) for interval in masks.snapshot().intervals
    ) == ((100, 180),)
    assert masks.limits is limits
    print("masks=100:180 changed=40,40,20,20")


if __name__ == "__main__":
    main()
