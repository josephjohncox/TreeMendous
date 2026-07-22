"""Reconcile free-port geometry atomically with no lease identity, TTL, or fencing.

This pattern is outside the 50-engine registry. It demonstrates only ordered,
whole-batch-atomic geometry mutation; a real port lease service must supply
ownership, clocks, durable coordination, and downstream fence enforcement.
"""

from treemendous.exact_batch import (
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)


def main() -> None:
    pool = ExactBatchRangeSet((10_000, 10_008))
    results = pool.mutate(
        [
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 10_001, 10_003),
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 10_005, 10_006),
        ]
    )
    free = ",".join(
        f"{interval.start}-{interval.end}" for interval in pool.snapshot().intervals
    )
    changed = ",".join(str(result.changed_length) for result in results)
    print(f"free={free} changed={changed}")


if __name__ == "__main__":
    main()
