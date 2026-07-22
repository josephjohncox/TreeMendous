#!/usr/bin/env python3
"""Fixed E4-B/D lease-state scaling and publication experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.leasing import (
    Lease,
    LeaseDiagnostics,
    LeasePool,
    LeasePoolSnapshot,
    LeaseState,
)
from treemendous.applications.leasing._common import PoolGroup
from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import Span
from treemendous.rangeset import RangeSet

SCHEMA = "treemendous-lease-state-scaling-experiment-v1"
ACTIVE_LEASES = (128, 512, 2_048, 8_192)
SMOKE_BLOCKS = 10
CONFIRMATION_BLOCKS = 30
FIXED_BLOCKS = (SMOKE_BLOCKS, CONFIRMATION_BLOCKS)
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 8_404_117
READ_CYCLES = 16
WRITE_CYCLES = 1
FENCE_LIMIT = 0.50
SNAPSHOT_LIMIT = 0.50
NO_EXPIRY_LIMIT = 0.50
NORMALIZED_SCALING_LIMIT = 2.0
WRITE_REGRESSION_LIMIT = 1.10
MEMORY_SETTLING_LIMIT = 1.10
_KINDS = (
    "repeated_snapshot",
    "fence_validation",
    "no_expiry_acquire_release",
    "expiry_burst",
)
_GATE_RATIO_ORDER = (
    "fence_n8192_upper",
    "snapshot_n8192_upper",
    "no_expiry_n8192_upper",
    "candidate_no_expiry_n8192_over_n2048",
    "write_cells_max_upper",
    "memory_settling",
)
_SUMMARY_KEYS = {"median", "median_95_low", "median_95_high"}
_BASE_ROW_KEYS = {
    "kind",
    "active_leases",
    "cycles_per_position",
    "validated_blocks",
    "blocks",
    "block_ratios",
    "candidate_ns_per_cycle",
    "baseline_ns_per_cycle",
    "ratio",
    "result_sha256",
    "final_state_sha256",
}
_ROW_EXTRA_KEYS = {
    "repeated_snapshot": set(),
    "fence_validation": {"baseline_linear_lease_visits"},
    "no_expiry_acquire_release": {
        "candidate_active_lease_replays",
        "baseline_active_lease_replays",
    },
    "expiry_burst": {"expired_per_cycle"},
}
BALANCED_BLOCK_METHOD = (
    "each fixed block has two multi-cycle positions, one candidate-first and one "
    "baseline-first; candidate and baseline durations are totaled within the "
    "whole block; bootstrap units are whole blocks"
)
STOP_RULE = (
    "Do not implement a lazy RangeSnapshot dataclass: the accepted exact immutable "
    "projection caches already satisfy the generic unchanged-state use case."
)
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATHS = (
    "tests/performance/experiments/lease_state_scaling.py",
    "treemendous/applications/_shared/leasing.py",
    "treemendous/applications/leasing/_common.py",
)
T = TypeVar("T")


def _seed_pool(
    count: int, *, short_tail: int = 0
) -> tuple[LeasePool, LogicalClock, list[Lease]]:
    clock = LogicalClock()
    pool = LeasePool((0, count + 1), clock=clock)
    leases = [
        pool.acquire(
            f"owner-{index}",
            ttl=2 if index >= count - short_tail else 1_000_000,
            exact_span=(index, index + 1),
        )
        for index in range(count)
    ]
    return pool, clock, leases


def _baseline_snapshot(pool: LeasePool) -> LeasePoolSnapshot:
    """Faithfully construct the pre-cache public snapshot on unchanged state."""
    with pool._lock:
        now = pool._observe_time()
        staged, free, _ = pool._stage_expirations(now)
        leases = tuple(sorted(staged.values(), key=lambda lease: lease.token))
        available = tuple(interval.span for interval in free.intervals())
        states = tuple(lease.state for lease in staged.values())
        diagnostics = LeaseDiagnostics(
            observed_at=now,
            total_capacity=pool._domain.measure,
            available_capacity=sum(span.length for span in available),
            largest_available_span=max((span.length for span in available), default=0),
            active_leases=states.count(LeaseState.ACTIVE),
            expired_leases=states.count(LeaseState.EXPIRED),
            released_leases=states.count(LeaseState.RELEASED),
            issued_tokens=pool._next_fencing_token - 1,
            next_fencing_token=pool._next_fencing_token,
        )
        result = LeasePoolSnapshot(
            now,
            pool.pool_id,
            pool.allowed_spans,
            available,
            leases,
            diagnostics,
        )
        pool._commit(staged, free, now)
        return result


def _baseline_fence_lookup(pool: LeasePool, evidence: Lease, scans: list[int]) -> Lease:
    """Reproduce the former public-snapshot plus linear token scan."""
    snapshot = _baseline_snapshot(pool)
    issued = None
    for lease in snapshot.leases:
        scans[0] += 1
        if lease.token == evidence.token:
            issued = lease
            break
    if issued is None:
        raise AssertionError("seeded fence evidence was not issued")
    if (
        issued.owner != evidence.owner
        or issued.resource != evidence.resource
        or issued.acquired_at != evidence.acquired_at
        or issued.request_id != evidence.request_id
    ):
        raise AssertionError("seeded fence evidence changed")
    return issued


def _elapsed(call: Callable[[], T]) -> tuple[int, T]:
    started = time.perf_counter_ns()
    result = call()
    return time.perf_counter_ns() - started, result


def _repeat(call: Callable[[], T], cycles: int) -> T:
    result = call()
    for _ in range(cycles - 1):
        result = call()
    return result


def _bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians))]


def _summary(values: list[float], seed: int) -> dict[str, float]:
    low, high = _bootstrap(values, seed)
    return {
        "median": float(statistics.median(values)),
        "median_95_low": low,
        "median_95_high": high,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _semantic_digests(kind: str, count: int) -> tuple[str, str]:
    """Return fixed normalized semantics, independent of measured durations."""
    result: Any
    final: Any
    if kind == "repeated_snapshot":
        result = {
            "available": [[count, count + 1]],
            "leases": [
                [index + 1, index, index + 1, "active"] for index in range(count)
            ],
        }
        final = result
    elif kind == "fence_validation":
        result = {
            "token": count,
            "owner": f"owner-{count - 1}",
            "resource": [count - 1, count],
            "state": "active",
        }
        final = {"active_leases": count, "available": [[count, count + 1]]}
    elif kind == "no_expiry_acquire_release":
        result = {
            "resource": [count, count + 1],
            "released_state": "released",
            "active_leases": count,
        }
        final = {"active_leases": count, "available": [[count, count + 1]]}
    elif kind == "expiry_burst":
        burst = max(1, count // 8)
        result = [
            [token, token - 1, token, "expired"]
            for token in range(count - burst + 1, count + 1)
        ]
        final = {
            "active_leases": count - burst,
            "expired_leases": burst,
            "available": [[count - burst, count + 1]],
        }
    else:
        raise ValueError(f"unknown row kind: {kind}")
    return _digest(result), _digest(final)


def _balanced_row(
    *,
    kind: str,
    count: int,
    blocks: int,
    cycles: int,
    candidate: Callable[[], Any],
    baseline: Callable[[], Any],
    validate: Callable[[Any, Any], None],
) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    candidate_totals: list[int] = []
    baseline_totals: list[int] = []
    for block_index in range(blocks):
        positions: list[dict[str, str | int]] = []
        orders = (
            ("candidate", "baseline")
            if block_index % 2 == 0
            else ("baseline", "candidate")
        )
        for first in orders:
            if first == "candidate":
                candidate_ns, candidate_result = _elapsed(
                    lambda: _repeat(candidate, cycles)
                )
                baseline_ns, baseline_result = _elapsed(
                    lambda: _repeat(baseline, cycles)
                )
            else:
                baseline_ns, baseline_result = _elapsed(
                    lambda: _repeat(baseline, cycles)
                )
                candidate_ns, candidate_result = _elapsed(
                    lambda: _repeat(candidate, cycles)
                )
            validate(candidate_result, baseline_result)
            positions.append(
                {
                    "first": first,
                    "candidate_ns": candidate_ns,
                    "baseline_ns": baseline_ns,
                }
            )
        candidate_total = sum(int(item["candidate_ns"]) for item in positions)
        baseline_total = sum(int(item["baseline_ns"]) for item in positions)
        raw.append(
            {
                "block_index": block_index,
                "positions": positions,
                "candidate_ns_total": candidate_total,
                "baseline_ns_total": baseline_total,
                "ratio": candidate_total / baseline_total,
            }
        )
        candidate_totals.append(candidate_total)
        baseline_totals.append(baseline_total)
    ratios = [
        candidate_ns / baseline_ns
        for candidate_ns, baseline_ns in zip(
            candidate_totals, baseline_totals, strict=True
        )
    ]
    seed = BOOTSTRAP_SEED + count * 17 + sum(map(ord, kind))
    divisor = float(2 * cycles)
    result_digest, final_digest = _semantic_digests(kind, count)
    return {
        "kind": kind,
        "active_leases": count,
        "cycles_per_position": cycles,
        "validated_blocks": blocks,
        "blocks": raw,
        "block_ratios": ratios,
        "candidate_ns_per_cycle": _summary(
            [value / divisor for value in candidate_totals], seed + 1
        ),
        "baseline_ns_per_cycle": _summary(
            [value / divisor for value in baseline_totals], seed + 2
        ),
        "ratio": _summary(ratios, seed),
        "result_sha256": result_digest,
        "final_state_sha256": final_digest,
    }


def _read_rows(count: int, blocks: int) -> list[dict[str, Any]]:
    pool, _, leases = _seed_pool(count)
    expected = pool.snapshot()
    scans = [0]
    snapshot_row = _balanced_row(
        kind="repeated_snapshot",
        count=count,
        blocks=blocks,
        cycles=READ_CYCLES,
        candidate=pool.snapshot,
        baseline=lambda: _baseline_snapshot(pool),
        validate=lambda candidate, baseline: _validate_snapshots(
            candidate, baseline, expected
        ),
    )
    evidence = leases[-1]
    fence_row = _balanced_row(
        kind="fence_validation",
        count=count,
        blocks=blocks,
        cycles=READ_CYCLES,
        candidate=lambda: pool._lookup_fence_lease(evidence),
        baseline=lambda: _baseline_fence_lookup(pool, evidence, scans),
        validate=lambda candidate, baseline: _validate_fence(
            candidate, baseline, evidence
        ),
    )
    fence_row["baseline_linear_lease_visits"] = scans[0]
    return [snapshot_row, fence_row]


def _validate_snapshots(
    candidate: LeasePoolSnapshot,
    baseline: LeasePoolSnapshot,
    expected: LeasePoolSnapshot,
) -> None:
    if candidate != baseline or candidate != expected:
        raise AssertionError("snapshot paths diverged")
    if (
        type(candidate) is not LeasePoolSnapshot
        or type(baseline) is not LeasePoolSnapshot
    ):
        raise AssertionError("snapshot exact type changed")


def _validate_fence(candidate: Lease, baseline: Lease, expected: Lease) -> None:
    if candidate != baseline or candidate != expected:
        raise AssertionError("fence lookup paths diverged")


def _observe_active_replays(pool: LeasePool, replays: list[int]) -> None:
    original = pool._build_free

    def rebuild_from_active(leases: dict[int, Lease]) -> Any:
        replays[0] += sum(lease.state is LeaseState.ACTIVE for lease in leases.values())
        return original(leases)

    setattr(pool, "_build_free", rebuild_from_active)


def _candidate_clone_free(pool: LeasePool, source: RangeSet) -> RangeSet:
    """Clone committed canonical free intervals for experiment-only staging."""
    staged = RangeSet(
        BackendAdapter(IntervalManager()),
        domain=pool._domain,
        initially_available=False,
    )
    for interval in source.intervals():
        staged.add(interval.span)
    return staged


def _candidate_stage_expirations(
    pool: LeasePool, now: int
) -> tuple[dict[int, Lease], RangeSet, tuple[Lease, ...]]:
    """Apply the rejected authoritative-free delta without a runtime seam."""
    expired = tuple(
        lease
        for lease in pool._leases.values()
        if lease.state is LeaseState.ACTIVE and lease.expires_at <= now
    )
    if not expired:
        return pool._leases, pool._free, ()
    staged_leases = dict(pool._leases)
    staged_free = _candidate_clone_free(pool, pool._free)
    transitioned = []
    for lease in sorted(expired, key=lambda item: item.token):
        terminal = replace(lease, state=LeaseState.EXPIRED)
        staged_leases[lease.token] = terminal
        staged_free.add(lease.resource)
        transitioned.append(terminal)
    return staged_leases, staged_free, tuple(transitioned)


def _candidate_acquire_exact(pool: LeasePool, owner: str, target: Span) -> Lease:
    """Run Candidate D's clone/delta acquire entirely inside the experiment."""
    with pool._lock:
        now = pool._observe_time()
        leases, free, _ = _candidate_stage_expirations(pool, now)
        allocation_free = _candidate_clone_free(pool, free)
        resource = pool._select_span(
            allocation_free,
            size=target.length,
            alignment=1,
            exact_span=target,
        )
        token = pool._next_fencing_token
        lease = Lease(
            pool.pool_id,
            owner,
            resource,
            token,
            now,
            now + 1_000_000,
        )
        committed = dict(leases)
        committed[token] = lease
        pool._commit(committed, allocation_free, now)
        pool._next_fencing_token = token + 1
        return lease


def _candidate_release(pool: LeasePool, handle: Lease) -> Lease:
    """Run Candidate D's clone/delta release entirely inside the experiment."""
    with pool._lock:
        now = pool._observe_time()
        leases, free, _ = _candidate_stage_expirations(pool, now)
        current = pool._current_for(handle, leases, owner=None)
        released = replace(current, state=LeaseState.RELEASED)
        committed = dict(leases)
        committed[current.token] = released
        staged_free = _candidate_clone_free(pool, free)
        staged_free.add(current.resource)
        pool._commit(committed, staged_free, now)
        return released


def _candidate_expire(pool: LeasePool) -> tuple[Lease, ...]:
    """Run Candidate D's free-delta expiry entirely inside the experiment."""
    with pool._lock:
        now = pool._observe_time()
        leases, free, expired = _candidate_stage_expirations(pool, now)
        pool._commit(leases, free, now)
        return expired


def _write_row(count: int, blocks: int) -> tuple[dict[str, Any], dict[str, int]]:
    candidate_pool, _, _ = _seed_pool(count)
    baseline_pool, _, _ = _seed_pool(count)
    baseline_replays = [0]
    _observe_active_replays(baseline_pool, baseline_replays)
    candidate_replays = [0]
    _observe_active_replays(candidate_pool, candidate_replays)
    target = Span(count, count + 1)

    def candidate_cycle() -> tuple[Span, LeaseState, int]:
        lease = _candidate_acquire_exact(candidate_pool, "candidate", target)
        released = _candidate_release(candidate_pool, lease)
        active = candidate_pool.snapshot().diagnostics.active_leases
        return lease.resource, released.state, active

    def baseline_cycle() -> tuple[Span, LeaseState, int]:
        lease = baseline_pool.acquire("baseline", ttl=1_000_000, exact_span=target)
        released = baseline_pool.release(lease)
        active = baseline_pool.snapshot().diagnostics.active_leases
        return lease.resource, released.state, active

    row = _balanced_row(
        kind="no_expiry_acquire_release",
        count=count,
        blocks=blocks,
        cycles=WRITE_CYCLES,
        candidate=candidate_cycle,
        baseline=baseline_cycle,
        validate=lambda candidate, baseline: _validate_write(
            candidate, baseline, count
        ),
    )
    instrumentation = {
        "candidate_active_lease_replays": candidate_replays[0],
        "baseline_active_lease_replays": baseline_replays[0],
    }
    row.update(instrumentation)
    return row, instrumentation


def _validate_write(candidate: Any, baseline: Any, count: int) -> None:
    expected = (Span(count, count + 1), LeaseState.RELEASED, count)
    if candidate != expected or baseline != expected:
        raise AssertionError("restorative write paths diverged")


def _baseline_expire(pool: LeasePool) -> tuple[Lease, ...]:
    with pool._lock:
        now = pool._observe_time()
        expired = tuple(
            lease
            for lease in pool._leases.values()
            if lease.state is LeaseState.ACTIVE and lease.expires_at <= now
        )
        staged = dict(pool._leases)
        transitioned = []
        for lease in sorted(expired, key=lambda item: item.token):
            terminal = replace(lease, state=LeaseState.EXPIRED)
            staged[lease.token] = terminal
            transitioned.append(terminal)
        free = pool._build_free(staged)
        pool._commit(staged, free, now)
        return tuple(transitioned)


def _expiry_row(count: int, blocks: int) -> dict[str, Any]:
    burst = max(1, count // 8)
    source, _, _ = _seed_pool(count, short_tail=burst)
    checkpoint = source.checkpoint()
    candidate_queue: list[LeasePool] = []
    baseline_queue: list[LeasePool] = []

    def prepare() -> None:
        candidate_clock = LogicalClock()
        baseline_clock = LogicalClock()
        candidate = LeasePool.from_checkpoint(checkpoint, clock=candidate_clock)
        baseline = LeasePool.from_checkpoint(checkpoint, clock=baseline_clock)
        candidate_clock.advance(2)
        baseline_clock.advance(2)
        candidate_queue.append(candidate)
        baseline_queue.append(baseline)

    # Setup is deliberately outside timers. Each block has two positions.
    for _ in range(blocks * 2):
        prepare()

    row = _balanced_row(
        kind="expiry_burst",
        count=count,
        blocks=blocks,
        cycles=1,
        candidate=lambda: _candidate_expire(candidate_queue.pop(0)),
        baseline=lambda: _baseline_expire(baseline_queue.pop(0)),
        validate=lambda candidate, baseline: _validate_expiry(
            candidate, baseline, burst
        ),
    )
    row["expired_per_cycle"] = burst
    return row


def _validate_expiry(
    candidate: tuple[Lease, ...], baseline: tuple[Lease, ...], burst: int
) -> None:
    normalized_candidate = tuple(
        (lease.token, lease.resource, lease.state) for lease in candidate
    )
    normalized_baseline = tuple(
        (lease.token, lease.resource, lease.state) for lease in baseline
    )
    if normalized_candidate != normalized_baseline or len(candidate) != burst:
        raise AssertionError("expiry paths diverged")


def _fence_instrumentation() -> dict[str, int]:
    group = PoolGroup({"scope": (Span(0, 4),)}, clock=LogicalClock())
    handle = group.acquire("scope", "owner", ttl=10)
    pool = group.pool("scope")
    calls = {"public_snapshot_calls": 0, "linear_lease_scans": 0}
    original_snapshot = pool.snapshot
    original_projection = pool._lease_projection

    def snapshot() -> LeasePoolSnapshot:
        calls["public_snapshot_calls"] += 1
        return original_snapshot()

    def projection(leases: dict[int, Lease]) -> tuple[Lease, ...]:
        calls["linear_lease_scans"] += len(leases)
        return original_projection(leases)

    setattr(pool, "snapshot", snapshot)
    setattr(pool, "_lease_projection", projection)
    if not group.validate_fence("key", handle):
        raise AssertionError("valid issued fence was rejected")
    return calls


def _retained_bytes(pool: LeasePool) -> int:
    import sys as _sys

    values: list[object] = [
        pool._free,
        pool._free.intervals(),
        pool._lease_projection_cache,
        pool._available_projection_cache,
        pool._state_counts_cache,
    ]
    return sum(_sys.getsizeof(value) for value in values if value is not None)


def _memory_settling() -> dict[str, Any]:
    pool, _, leases = _seed_pool(128)
    current = leases[0]
    checkpoints: dict[int, int] = {}
    for cycle in range(1, 1_001):
        current = pool.renew(current, ttl=1_000_000)
        pool.snapshot()
        if cycle in (100, 1_000):
            gc.collect()
            checkpoints[cycle] = _retained_bytes(pool)
    ratio = checkpoints[1_000] / checkpoints[100]
    return {
        "restorative_cycles": 1_000,
        "bytes_after_100": checkpoints[100],
        "bytes_after_1000": checkpoints[1_000],
        "settling_ratio": ratio,
        "limit": MEMORY_SETTLING_LIMIT,
    }


def _gate(
    rows: list[dict[str, Any]], instrumentation: dict[str, int], memory: dict[str, Any]
) -> dict[str, Any]:
    by_key = {(row["kind"], row["active_leases"]): row for row in rows}
    fence = by_key[("fence_validation", 8_192)]["ratio"]["median_95_high"]
    snapshot = by_key[("repeated_snapshot", 8_192)]["ratio"]["median_95_high"]
    no_expiry = by_key[("no_expiry_acquire_release", 8_192)]["ratio"]["median_95_high"]
    candidate_8192 = by_key[("no_expiry_acquire_release", 8_192)][
        "candidate_ns_per_cycle"
    ]["median"]
    candidate_2048 = by_key[("no_expiry_acquire_release", 2_048)][
        "candidate_ns_per_cycle"
    ]["median"]
    scaling = candidate_8192 / candidate_2048
    write_rows = [
        row
        for row in rows
        if row["kind"] in {"no_expiry_acquire_release", "expiry_burst"}
    ]
    write_max = max(row["ratio"]["median_95_high"] for row in write_rows)
    b_checks = {
        "fence_zero_public_snapshots": instrumentation["public_snapshot_calls"] == 0,
        "fence_zero_linear_scans": instrumentation["linear_lease_scans"] == 0,
        "fence_n8192_upper_at_most_0_50": fence <= FENCE_LIMIT,
        "snapshot_n8192_upper_at_most_0_50": snapshot <= SNAPSHOT_LIMIT,
    }
    d_checks = {
        "no_expiry_zero_active_replays": all(
            row.get("candidate_active_lease_replays", 0) == 0
            for row in write_rows
            if row["kind"] == "no_expiry_acquire_release"
        ),
        "no_expiry_n8192_upper_at_most_0_50": no_expiry <= NO_EXPIRY_LIMIT,
        "candidate_n8192_over_n2048_at_most_2": scaling <= NORMALIZED_SCALING_LIMIT,
        "write_cells_upper_at_most_1_10": write_max <= WRITE_REGRESSION_LIMIT,
        "retained_memory_settles_within_10_percent": memory["settling_ratio"]
        <= MEMORY_SETTLING_LIMIT,
    }
    b_accepted = all(b_checks.values())
    d_accepted = all(d_checks.values())
    return {
        "candidate_b": {
            "decision": "ACCEPTED" if b_accepted else "REJECTED",
            "checks": b_checks,
        },
        "candidate_d": {
            "decision": "ACCEPTED" if d_accepted else "REJECTED",
            "checks": d_checks,
        },
        "ratios": {
            "fence_n8192_upper": fence,
            "snapshot_n8192_upper": snapshot,
            "no_expiry_n8192_upper": no_expiry,
            "candidate_no_expiry_n8192_over_n2048": scaling,
            "write_cells_max_upper": write_max,
            "memory_settling": memory["settling_ratio"],
        },
    }


def _source_entries() -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "sha256": hashlib.sha256((_ROOT / path).read_bytes()).hexdigest(),
        }
        for path in _SOURCE_PATHS
    ]


def _provenance() -> dict[str, Any]:
    sources = _source_entries()
    source_state = hashlib.sha256(
        "".join(item["sha256"] for item in sources).encode()
    ).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git": {"commit": commit, "source_state_sha256": source_state},
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "sources": sources,
    }


def _methodology(blocks: int = CONFIRMATION_BLOCKS) -> dict[str, Any]:
    return {
        "active_leases": list(ACTIVE_LEASES),
        "blocks": blocks,
        "balanced_block_method": BALANCED_BLOCK_METHOD,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "same_instance_validation_outside_timing": True,
        "baseline": "faithful pre-candidate construction and active-lease replay functions",
        "stop_rule": STOP_RULE,
    }


def run_matrix(
    *, blocks: int = CONFIRMATION_BLOCKS, sizes: tuple[int, ...] = ACTIVE_LEASES
) -> dict[str, Any]:
    if type(blocks) is not int or blocks not in FIXED_BLOCKS:
        raise ValueError(f"blocks must be one of the fixed profiles: {FIXED_BLOCKS}")
    rows: list[dict[str, Any]] = []
    instrumentation = _fence_instrumentation()
    for count in sizes:
        rows.extend(_read_rows(count, blocks))
        write_row, _ = _write_row(count, blocks)
        rows.append(write_row)
        rows.append(_expiry_row(count, blocks))
    memory = _memory_settling()
    gate = _gate(rows, instrumentation, memory) if sizes == ACTIVE_LEASES else None
    return {
        "schema": SCHEMA,
        "blocks": blocks,
        "methodology": _methodology(blocks),
        "instrumentation": instrumentation,
        "rows": rows,
        "memory": memory,
        "gate": gate,
        "provenance": _provenance(),
    }


def _canonical(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Lease state scaling experiment",
        "",
        f"JSON SHA-256: `{digest}`",
        "",
        "| Candidate | Decision |",
        "| --- | --- |",
    ]
    gate = report["gate"]
    if gate is None:
        lines.extend(("| E4-B | diagnostic |", "| E4-D | diagnostic |"))
    else:
        lines.extend(
            (
                f"| E4-B | {gate['candidate_b']['decision']} |",
                f"| E4-D | {gate['candidate_d']['decision']} |",
                "",
                "## Exact gate ratios",
                "",
            )
        )
        for name in _GATE_RATIO_ORDER:
            lines.append(f"- `{name}`: {gate['ratios'][name]:.6f}")
    lines.extend(("", "## Stop rule", "", STOP_RULE, ""))
    return "\n".join(lines)


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(report)
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    markdown.write_text(_markdown(report, digest))
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, markdown, checksum


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], expected[key]) for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _require_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} schema mismatch")
    return value


def verify_artifacts(output: Path) -> dict[str, Any]:
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    expected_checksum = f"{digest}  {output.name}\n"
    if Path(f"{output}.sha256").read_text() != expected_checksum:
        raise ValueError("SHA-256 sidecar mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )
    if encoded != _canonical(report):
        raise ValueError("JSON is not canonical")
    _require_keys(
        report,
        {
            "schema",
            "blocks",
            "methodology",
            "instrumentation",
            "rows",
            "memory",
            "gate",
            "provenance",
        },
        "report",
    )
    if type(report["schema"]) is not str or report["schema"] != SCHEMA:
        raise ValueError("schema mismatch")
    blocks_count = report["blocks"]
    if type(blocks_count) is not int or blocks_count not in FIXED_BLOCKS:
        raise ValueError("fixed block count mismatch")
    if not _exact(report["methodology"], _methodology(blocks_count)):
        raise ValueError("fixed methodology mismatch")

    instrumentation = _require_keys(
        report["instrumentation"],
        {"public_snapshot_calls", "linear_lease_scans"},
        "instrumentation",
    )
    if any(type(value) is not int for value in instrumentation.values()) or not _exact(
        instrumentation, {"public_snapshot_calls": 0, "linear_lease_scans": 0}
    ):
        raise ValueError("instrumentation derivation mismatch")

    memory = _require_keys(
        report["memory"],
        {
            "restorative_cycles",
            "bytes_after_100",
            "bytes_after_1000",
            "settling_ratio",
            "limit",
        },
        "memory",
    )
    if (
        type(memory["restorative_cycles"]) is not int
        or memory["restorative_cycles"] != 1_000
        or type(memory["bytes_after_100"]) is not int
        or type(memory["bytes_after_1000"]) is not int
        or min(memory["bytes_after_100"], memory["bytes_after_1000"]) <= 0
        or type(memory["settling_ratio"]) is not float
        or memory["settling_ratio"]
        != memory["bytes_after_1000"] / memory["bytes_after_100"]
        or type(memory["limit"]) is not float
        or memory["limit"] != MEMORY_SETTLING_LIMIT
    ):
        raise ValueError("memory derivation mismatch")

    rows = report["rows"]
    expected_matrix = [(kind, count) for count in ACTIVE_LEASES for kind in _KINDS]
    if type(rows) is not list or len(rows) != len(expected_matrix):
        raise ValueError("fixed matrix mismatch")
    observed_matrix: list[tuple[str, int]] = []
    for row, expected_identity in zip(rows, expected_matrix, strict=True):
        if type(row) is not dict:
            raise ValueError("row exact type mismatch")
        kind, count = expected_identity
        if "kind" not in row or "active_leases" not in row:
            raise ValueError("per-kind row schema mismatch")
        if type(row["kind"]) is not str or type(row["active_leases"]) is not int:
            raise ValueError("row identity exact type mismatch")
        identity = (row["kind"], row["active_leases"])
        observed_matrix.append(identity)
        if identity != expected_identity:
            raise ValueError("ordered unique matrix mismatch")
        expected_keys = _BASE_ROW_KEYS | _ROW_EXTRA_KEYS[kind]
        if set(row) != expected_keys:
            raise ValueError("per-kind row schema mismatch")
        expected_cycles = (
            WRITE_CYCLES
            if kind == "no_expiry_acquire_release"
            else 1
            if kind == "expiry_burst"
            else READ_CYCLES
        )
        if (
            type(row["cycles_per_position"]) is not int
            or row["cycles_per_position"] != expected_cycles
            or type(row["validated_blocks"]) is not int
            or row["validated_blocks"] != blocks_count
        ):
            raise ValueError("row cycle/block mismatch")
        expected_result, expected_final = _semantic_digests(kind, count)
        if (
            type(row["result_sha256"]) is not str
            or row["result_sha256"] != expected_result
            or type(row["final_state_sha256"]) is not str
            or row["final_state_sha256"] != expected_final
        ):
            raise ValueError("semantic digest mismatch")
        blocks = row["blocks"]
        if type(blocks) is not list or len(blocks) != blocks_count:
            raise ValueError("raw block mismatch")
        candidate_totals: list[int] = []
        baseline_totals: list[int] = []
        for index, block in enumerate(blocks):
            _require_keys(
                block,
                {
                    "block_index",
                    "positions",
                    "candidate_ns_total",
                    "baseline_ns_total",
                    "ratio",
                },
                "block",
            )
            if type(block["block_index"]) is not int or block["block_index"] != index:
                raise ValueError("block exact type/index mismatch")
            positions = block["positions"]
            expected_first = (
                ["candidate", "baseline"]
                if index % 2 == 0
                else ["baseline", "candidate"]
            )
            if type(positions) is not list or len(positions) != 2:
                raise ValueError("balanced block ordering mismatch")
            for position, first in zip(positions, expected_first, strict=True):
                _require_keys(
                    position,
                    {"first", "candidate_ns", "baseline_ns"},
                    "position",
                )
                if type(position["first"]) is not str or position["first"] != first:
                    raise ValueError("balanced block ordering mismatch")
                if any(
                    type(position[key]) is not int or position[key] <= 0
                    for key in ("candidate_ns", "baseline_ns")
                ):
                    raise ValueError("exact timing type mismatch")
            candidate_total = sum(item["candidate_ns"] for item in positions)
            baseline_total = sum(item["baseline_ns"] for item in positions)
            expected_ratio = candidate_total / baseline_total
            if (
                type(block["candidate_ns_total"]) is not int
                or block["candidate_ns_total"] != candidate_total
                or type(block["baseline_ns_total"]) is not int
                or block["baseline_ns_total"] != baseline_total
                or type(block["ratio"]) is not float
                or block["ratio"] != expected_ratio
            ):
                raise ValueError("raw block derivation mismatch")
            candidate_totals.append(candidate_total)
            baseline_totals.append(baseline_total)
        ratios = [a / b for a, b in zip(candidate_totals, baseline_totals, strict=True)]
        seed = BOOTSTRAP_SEED + count * 17 + sum(map(ord, kind))
        divisor = float(2 * expected_cycles)
        for label, value in (
            ("ratio", row["ratio"]),
            ("candidate summary", row["candidate_ns_per_cycle"]),
            ("baseline summary", row["baseline_ns_per_cycle"]),
        ):
            summary = _require_keys(value, _SUMMARY_KEYS, label)
            if any(type(item) is not float for item in summary.values()):
                raise ValueError("summary exact type mismatch")
        if (
            type(row["block_ratios"]) is not list
            or any(type(value) is not float for value in row["block_ratios"])
            or not _exact(row["block_ratios"], ratios)
            or not _exact(row["ratio"], _summary(ratios, seed))
            or not _exact(
                row["candidate_ns_per_cycle"],
                _summary([value / divisor for value in candidate_totals], seed + 1),
            )
            or not _exact(
                row["baseline_ns_per_cycle"],
                _summary([value / divisor for value in baseline_totals], seed + 2),
            )
        ):
            raise ValueError("derived ratio or timing summary mismatch")
        if kind == "fence_validation" and (
            type(row["baseline_linear_lease_visits"]) is not int
            or row["baseline_linear_lease_visits"]
            != count * blocks_count * 2 * READ_CYCLES
        ):
            raise ValueError("fence instrumentation derivation mismatch")
        if kind == "no_expiry_acquire_release" and (
            type(row["candidate_active_lease_replays"]) is not int
            or row["candidate_active_lease_replays"] != 0
            or type(row["baseline_active_lease_replays"]) is not int
            or row["baseline_active_lease_replays"] != count * blocks_count * 4
        ):
            raise ValueError("write instrumentation derivation mismatch")
        if kind == "expiry_burst" and (
            type(row["expired_per_cycle"]) is not int
            or row["expired_per_cycle"] != max(1, count // 8)
        ):
            raise ValueError("expiry semantics mismatch")
    if observed_matrix != expected_matrix or len(set(observed_matrix)) != len(
        expected_matrix
    ):
        raise ValueError("ordered unique matrix mismatch")

    expected_gate = _gate(rows, instrumentation, memory)
    if not _exact(report["gate"], expected_gate):
        raise ValueError("gate mismatch")
    if not _exact(report["provenance"], _provenance()):
        raise ValueError("source/runtime provenance mismatch")
    if output.with_suffix(".md").read_text() != _markdown(report, digest):
        raise ValueError("Markdown mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=CONFIRMATION_BLOCKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_artifacts(args.output)
        print(f"verified {args.output}")
        return
    report = run_matrix(blocks=args.blocks)
    paths = write_artifacts(report, args.output)
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
