"""Correctness-attested benchmark for TCP and UDP port leasing."""

from __future__ import annotations

from tests.performance.applications.harness import ApplicationSample
from tests.performance.applications.leasing._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    AdvancingClock,
    LeasingBenchmarkAdapter,
    build_commands,
    clock_reads,
    no_snapshot_extra,
    run_prepared_benchmark,
    validate_parameters,
)
from treemendous.applications.leasing.tcp_udp_ports import PortLeaseEngine

_SCOPE = "tcp"
_SCOPES = ("tcp", "udp")
_READS = {
    "acquire": (_SCOPE,),
    "renew": (_SCOPE,),
    "fence": (_SCOPE,),
    "release": (_SCOPE,),
    "expire": _SCOPES,
}


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    """Run a bounded deterministic port lifecycle and attest all evidence."""
    validate_parameters(operations, seed)
    draft = build_commands(operations=operations, seed=seed, short_ttl=5, long_ttl=1)
    advancing_reads = len(_SCOPES) + clock_reads(draft, _READS)
    commands = build_commands(
        operations=operations,
        seed=seed,
        short_ttl=5,
        long_ttl=advancing_reads + 100,
    )
    clock = AdvancingClock(advancing_reads)
    engine = PortLeaseEngine(clock=clock, system_ports=None, ephemeral_ports=None)
    adapter = LeasingBenchmarkAdapter(
        scenario_id="tcp-udp-port-leases",
        primary_scope=_SCOPE,
        scopes=_SCOPES,
        domains={scope: ((1, 65536),) for scope in _SCOPES},
        reads_by_action=_READS,
        acquire=lambda owner, ttl: engine.acquire(_SCOPE, owner, ttl=ttl),
        renew=lambda handle, ttl: engine.renew(handle, ttl=ttl),
        release=engine.release,
        expire=engine.expire,
        fence=lambda handle: engine.validate_fence(handle, handle.resource.start),
        snapshot=engine.snapshot,
        snapshot_group=lambda snapshot: snapshot,
        diagnostics=engine.diagnostics,
        snapshot_extra=no_snapshot_extra,
    )
    return run_prepared_benchmark(commands=commands, adapter=adapter)


if __name__ == "__main__":
    print(run_benchmark().to_dict())
