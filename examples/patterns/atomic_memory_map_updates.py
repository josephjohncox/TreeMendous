"""Apply atomic memory-map geometry updates without mapping identity or OS calls.

This pattern is outside the 50-engine registry. It supplies ordered half-open
signed-integer geometry only; a real memory manager must provide mapping
identity, page-size/alignment and protection semantics, persistence or a WAL,
and operating-system publication and recovery.
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
    mapped = ExactBatchRangeSet(
        (-4_096, 8_192), initially_available=False, limits=limits
    )
    results = mapped.mutate(
        (
            BatchMutation(MutationOpcode.ADD, -4_096, -2_048),
            BatchMutation(MutationOpcode.ADD, 0, 4_096),
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 1_024, 2_048),
            BatchMutation(MutationOpcode.ADD, 1_024, 2_048),
        )
    )

    assert results == (
        MutationResult((Span(-4_096, -2_048),), 2_048, False),
        MutationResult((Span(0, 4_096),), 4_096, False),
        MutationResult((Span(1_024, 2_048),), 1_024, True),
        MutationResult((Span(1_024, 2_048),), 1_024, False),
    )
    assert tuple(
        (interval.start, interval.end) for interval in mapped.snapshot().intervals
    ) == ((-4_096, -2_048), (0, 4_096))
    assert mapped.limits is limits
    print("mapped=-4096:-2048,0:4096 changed=2048,4096,1024,1024")


if __name__ == "__main__":
    main()
