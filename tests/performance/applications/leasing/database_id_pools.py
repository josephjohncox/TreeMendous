"""Correctness-attested benchmark for monotonic database ID leasing."""

from __future__ import annotations

from tests.performance.applications.harness import ApplicationSample
from tests.performance.applications.leasing._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    AdvancingClock,
    LeasingBenchmarkAdapter,
    build_commands,
    clock_reads,
    database_snapshot_extra,
    run_prepared_benchmark,
    validate_parameters,
)
from treemendous.applications.leasing.database_ids import DatabaseIdPool

_SCOPE = "benchmark"
_READS = {
    "acquire": (_SCOPE, _SCOPE),
    "renew": (_SCOPE,),
    "fence": (_SCOPE,),
    "release": (_SCOPE,),
    "expire": (_SCOPE,),
}


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    """Run a bounded deterministic ID lease lifecycle and attest all evidence."""
    validate_parameters(operations, seed)
    draft = build_commands(operations=operations, seed=seed, short_ttl=6, long_ttl=1)
    advancing_reads = 1 + clock_reads(draft, _READS)
    commands = build_commands(
        operations=operations,
        seed=seed,
        short_ttl=6,
        long_ttl=advancing_reads + 100,
    )
    clock = AdvancingClock(advancing_reads)
    capacity = max(operations + 16, 64)
    engine = DatabaseIdPool(_SCOPE, minimum_id=1, maximum_id=capacity, clock=clock)
    adapter = LeasingBenchmarkAdapter(
        scenario_id="database-id-pools",
        primary_scope=_SCOPE,
        scopes=(_SCOPE,),
        domains={_SCOPE: ((1, capacity + 1),)},
        reads_by_action=_READS,
        acquire=lambda owner, ttl: engine.acquire(owner, ttl=ttl),
        renew=lambda handle, ttl: engine.renew(handle, ttl=ttl),
        release=engine.release,
        expire=engine.expire,
        fence=lambda handle: engine.validate_fence(handle, handle.resource.start),
        snapshot=engine.snapshot,
        snapshot_group=lambda snapshot: snapshot.pools,
        diagnostics=engine.diagnostics,
        snapshot_extra=database_snapshot_extra,
        monotonic=True,
    )
    return run_prepared_benchmark(commands=commands, adapter=adapter)


if __name__ == "__main__":
    print(run_benchmark().to_dict())
