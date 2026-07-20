"""Shared correctness-first machinery for leasing application benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)

MAX_OPERATIONS = 10_000
DEFAULT_OPERATIONS = 128
DEFAULT_SEED = 42

Action = Literal["acquire", "renew", "fence", "release", "expire"]


class AdvancingClock:
    """Deterministic clock that advances only for the prepared workload.

    Pool construction consumes the first reads. Every clock read made by a
    timed public leasing call then advances by one. Reads made while observing
    the completed engine are frozen at the final workload timestamp.
    """

    def __init__(self, advancing_reads: int) -> None:
        self._advancing_reads = advancing_reads
        self._reads = 0

    def now(self) -> int:
        """Return the next prepared timestamp, then freeze at the last one."""
        value = min(self._reads, self._advancing_reads - 1)
        self._reads += 1
        return value


@dataclass(frozen=True)
class LeaseCommand:
    """One already-generated public leasing call."""

    action: Action
    slot: str
    owner: str = ""
    ttl: int = 0


@dataclass(frozen=True)
class LeasingBenchmarkAdapter:
    """Scenario-specific public calls and observable state adapters."""

    scenario_id: str
    primary_scope: str
    scopes: tuple[str, ...]
    domains: Mapping[str, tuple[tuple[int, int], ...]]
    reads_by_action: Mapping[str, tuple[str, ...]]
    acquire: Callable[[str, int], Any]
    renew: Callable[[Any, int], Any]
    release: Callable[[Any], Any]
    expire: Callable[[], tuple[Any, ...]]
    fence: Callable[[Any], bool]
    snapshot: Callable[[], Any]
    snapshot_group: Callable[[Any], Any]
    diagnostics: Callable[[], Any]
    snapshot_extra: Callable[[Any], Any]
    monotonic: bool = False
    metadata_labels: tuple[str, str] | None = None


@dataclass
class _OracleLease:
    scope: str
    owner: str
    start: int
    end: int
    token: int
    acquired_at: int
    expires_at: int
    state: str = "active"
    revision: int = 1


class _NaiveOracle:
    """Small list-based lease model independent of production range code."""

    def __init__(
        self,
        *,
        scopes: tuple[str, ...],
        domains: Mapping[str, tuple[tuple[int, int], ...]],
        monotonic: bool,
    ) -> None:
        self.scopes = scopes
        self.domains = domains
        self.monotonic = monotonic
        self.leases: dict[str, list[_OracleLease]] = {scope: [] for scope in scopes}
        self.next_token = {scope: 1 for scope in scopes}
        self.next_monotonic = {scope: domains[scope][0][0] for scope in scopes}
        self.reusable: dict[str, set[int]] = {scope: set() for scope in scopes}
        self.handles: dict[str, _OracleLease] = {}
        self.fences: dict[tuple[str, int], int] = {}

    def expire_scope(self, scope: str, now: int) -> tuple[_OracleLease, ...]:
        expired: list[_OracleLease] = []
        for lease in self.leases[scope]:
            if lease.state == "active" and lease.expires_at <= now:
                lease.state = "expired"
                expired.append(lease)
                if self.monotonic:
                    self.reusable[scope].update(range(lease.start, lease.end))
        return tuple(expired)

    def _first_available(self, scope: str) -> int:
        occupied = {
            lease.start for lease in self.leases[scope] if lease.state == "active"
        }
        for start, end in self.domains[scope]:
            for value in range(start, end):
                if value not in occupied:
                    return value
        raise RuntimeError("naive leasing oracle exhausted its bounded domain")

    def acquire(self, scope: str, owner: str, ttl: int, now: int) -> _OracleLease:
        if self.monotonic:
            start = self.next_monotonic[scope]
            self.next_monotonic[scope] += 1
        else:
            start = self._first_available(scope)
        token = self.next_token[scope]
        self.next_token[scope] += 1
        lease = _OracleLease(scope, owner, start, start + 1, token, now, now + ttl)
        self.leases[scope].append(lease)
        return lease

    def renew(self, lease: _OracleLease, ttl: int, now: int) -> _OracleLease:
        lease.expires_at = now + ttl
        lease.revision += 1
        return lease

    def release(self, lease: _OracleLease) -> _OracleLease:
        lease.state = "released"
        if self.monotonic:
            self.reusable[lease.scope].update(range(lease.start, lease.end))
        return lease

    def fence(self, lease: _OracleLease) -> bool:
        key = (lease.scope, lease.start)
        previous = self.fences.get(key, 0)
        accepted = lease.token >= previous
        if accepted:
            self.fences[key] = lease.token
        return accepted


def validate_parameters(operations: int, seed: int) -> None:
    """Reject unbounded or nondeterministically typed benchmark inputs."""
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if operations <= 0:
        raise ValueError("operations must be positive")
    if operations > MAX_OPERATIONS:
        raise ValueError(f"operations must not exceed {MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def build_commands(
    *, operations: int, seed: int, short_ttl: int, long_ttl: int
) -> tuple[LeaseCommand, ...]:
    """Build a bounded deterministic acquire/renew/fence/release trace."""
    random = Random(seed)
    commands: list[LeaseCommand] = []
    for index in range(operations):
        cycle, phase = divmod(index, 8)
        expiring = f"expiring-{cycle}"
        releasable = f"releasable-{cycle}"
        active = f"active-{cycle}"
        owner = f"owner-{cycle}-{random.randrange(1_000_000):06d}"
        if phase == 0:
            command = LeaseCommand("acquire", expiring, owner, short_ttl)
        elif phase == 1:
            command = LeaseCommand("acquire", releasable, owner, long_ttl)
        elif phase == 2:
            command = LeaseCommand("renew", releasable, ttl=long_ttl)
        elif phase == 3:
            command = LeaseCommand("fence", releasable)
        elif phase == 4:
            command = LeaseCommand("release", releasable)
        elif phase == 5:
            command = LeaseCommand("expire", expiring)
        elif phase == 6:
            command = LeaseCommand("acquire", active, owner, long_ttl)
        else:
            command = LeaseCommand("fence", active)
        commands.append(command)
    return tuple(commands)


def clock_reads(
    commands: tuple[LeaseCommand, ...],
    reads_by_action: Mapping[str, tuple[str, ...]],
) -> int:
    """Count clock reads made by the prepared public-call trace."""
    return sum(len(reads_by_action[command.action]) for command in commands)


def _inner_lease(handle: Any) -> tuple[str, Any]:
    inner = getattr(handle, "inner", handle)
    return inner.scope, inner.lease


def _lease_record_evidence(scope: str, lease: Any) -> dict[str, Any]:
    return {
        "scope": scope,
        "owner": lease.owner,
        "resource": (lease.resource.start, lease.resource.end),
        "token": lease.token,
        "acquired_at": lease.acquired_at,
        "expires_at": lease.expires_at,
        "revision": lease.revision,
        "state": lease.state.value,
        "request_id": lease.request_id,
    }


def _lease_evidence(handle: Any) -> dict[str, Any]:
    scope, lease = _inner_lease(handle)
    return _lease_record_evidence(scope, lease)


def _oracle_lease_evidence(lease: _OracleLease) -> dict[str, Any]:
    return {
        "scope": lease.scope,
        "owner": lease.owner,
        "resource": (lease.start, lease.end),
        "token": lease.token,
        "acquired_at": lease.acquired_at,
        "expires_at": lease.expires_at,
        "revision": lease.revision,
        "state": lease.state,
        "request_id": None,
    }


def _result_evidence(action: Action, value: Any) -> tuple[str, Any]:
    if action == "expire":
        return action, tuple(_lease_evidence(handle) for handle in value)
    if action == "fence":
        return action, value
    return action, _lease_evidence(value)


def execute_commands(
    commands: tuple[LeaseCommand, ...], adapter: LeasingBenchmarkAdapter
) -> tuple[tuple[Action, Any], ...]:
    """Execute only prepared public leasing calls and retain their raw results."""
    handles: dict[str, Any] = {}
    results: list[tuple[Action, Any]] = []
    for command in commands:
        if command.action == "acquire":
            value = adapter.acquire(command.owner, command.ttl)
            handles[command.slot] = value
        elif command.action == "renew":
            value = adapter.renew(handles[command.slot], command.ttl)
            handles[command.slot] = value
        elif command.action == "release":
            value = adapter.release(handles[command.slot])
            handles[command.slot] = value
        elif command.action == "fence":
            value = adapter.fence(handles[command.slot])
        else:
            value = adapter.expire()
        results.append((command.action, value))
    return tuple(results)


def _span_evidence(spans: Any) -> tuple[tuple[int, int], ...]:
    return tuple((span.start, span.end) for span in spans)


def _diagnostic_evidence(diagnostic: Any) -> dict[str, int]:
    return {
        "observed_at": diagnostic.observed_at,
        "total_capacity": diagnostic.total_capacity,
        "available_capacity": diagnostic.available_capacity,
        "largest_available_span": diagnostic.largest_available_span,
        "active_leases": diagnostic.active_leases,
        "expired_leases": diagnostic.expired_leases,
        "released_leases": diagnostic.released_leases,
        "issued_tokens": diagnostic.issued_tokens,
        "next_fencing_token": diagnostic.next_fencing_token,
    }


def _group_state_evidence(group: Any) -> tuple[tuple[str, Any], ...]:
    pools: list[tuple[str, Any]] = []
    for scope, pool in group.pools:
        leases = tuple(_lease_record_evidence(scope, lease) for lease in pool.leases)
        pools.append(
            (
                scope,
                {
                    "observed_at": pool.observed_at,
                    "allowed_spans": _span_evidence(pool.allowed_spans),
                    "available_spans": _span_evidence(pool.available_spans),
                    "leases": leases,
                    "active_leases": tuple(
                        lease for lease in leases if lease["state"] == "active"
                    ),
                    "tokens": tuple(lease["token"] for lease in leases),
                    "expiries": tuple(
                        (lease["token"], lease["expires_at"], lease["state"])
                        for lease in leases
                    ),
                },
            )
        )
    return tuple(pools)


def _group_counter_evidence(group: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (scope, _diagnostic_evidence(diagnostic)) for scope, diagnostic in group.pools
    )


def observe_engine(
    raw: tuple[tuple[Action, Any], ...], adapter: LeasingBenchmarkAdapter
) -> ApplicationOutcome:
    """Observe the same timed engine instance after the timer has stopped."""
    results = tuple(_result_evidence(action, value) for action, value in raw)
    snapshot = adapter.snapshot()
    return ApplicationOutcome(
        results,
        {
            "pools": _group_state_evidence(adapter.snapshot_group(snapshot)),
            "extra": adapter.snapshot_extra(snapshot),
        },
        _group_counter_evidence(adapter.diagnostics()),
    )


def _available_spans(
    domains: tuple[tuple[int, int], ...], leases: list[_OracleLease]
) -> tuple[tuple[int, int], ...]:
    occupied = sorted(lease.start for lease in leases if lease.state == "active")
    available: list[tuple[int, int]] = []
    for domain_start, domain_end in domains:
        cursor = domain_start
        for value in occupied:
            if value < domain_start or value >= domain_end:
                continue
            if cursor < value:
                available.append((cursor, value))
            cursor = value + 1
        if cursor < domain_end:
            available.append((cursor, domain_end))
    return tuple(available)


def _oracle_group_state(
    oracle: _NaiveOracle, observed_at: int
) -> tuple[tuple[str, Any], ...]:
    pools: list[tuple[str, Any]] = []
    for scope in oracle.scopes:
        leases = tuple(_oracle_lease_evidence(lease) for lease in oracle.leases[scope])
        pools.append(
            (
                scope,
                {
                    "observed_at": observed_at,
                    "allowed_spans": oracle.domains[scope],
                    "available_spans": _available_spans(
                        oracle.domains[scope], oracle.leases[scope]
                    ),
                    "leases": leases,
                    "active_leases": tuple(
                        lease for lease in leases if lease["state"] == "active"
                    ),
                    "tokens": tuple(lease["token"] for lease in leases),
                    "expiries": tuple(
                        (lease["token"], lease["expires_at"], lease["state"])
                        for lease in leases
                    ),
                },
            )
        )
    return tuple(pools)


def _oracle_counters(
    oracle: _NaiveOracle, observed_at: int
) -> tuple[tuple[str, Any], ...]:
    counters: list[tuple[str, Any]] = []
    for scope in oracle.scopes:
        leases = oracle.leases[scope]
        available = _available_spans(oracle.domains[scope], leases)
        total_capacity = sum(end - start for start, end in oracle.domains[scope])
        available_capacity = sum(end - start for start, end in available)
        counters.append(
            (
                scope,
                {
                    "observed_at": observed_at,
                    "total_capacity": total_capacity,
                    "available_capacity": available_capacity,
                    "largest_available_span": max(
                        (end - start for start, end in available), default=0
                    ),
                    "active_leases": sum(lease.state == "active" for lease in leases),
                    "expired_leases": sum(lease.state == "expired" for lease in leases),
                    "released_leases": sum(
                        lease.state == "released" for lease in leases
                    ),
                    "issued_tokens": len(leases),
                    "next_fencing_token": oracle.next_token[scope],
                },
            )
        )
    return tuple(counters)


def _merged_reusable(values: set[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    ordered = sorted(values)
    merged: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            merged.append((start, previous + 1))
            start = value
        previous = value
    merged.append((start, previous + 1))
    return tuple(merged)


def oracle_outcome(
    commands: tuple[LeaseCommand, ...], adapter: LeasingBenchmarkAdapter
) -> ApplicationOutcome:
    """Replay the prepared commands through the independent list-based model."""
    oracle = _NaiveOracle(
        scopes=adapter.scopes,
        domains=adapter.domains,
        monotonic=adapter.monotonic,
    )
    next_time = len(adapter.scopes)
    results: list[tuple[str, Any]] = []
    for command in commands:
        newly_expired: list[_OracleLease] = []
        now = next_time - 1
        for scope in adapter.reads_by_action[command.action]:
            now = next_time
            next_time += 1
            expired = oracle.expire_scope(scope, now)
            if command.action == "expire":
                newly_expired.extend(expired)
        if command.action == "acquire":
            value: Any = oracle.acquire(
                adapter.primary_scope, command.owner, command.ttl, now
            )
            oracle.handles[command.slot] = value
        elif command.action == "renew":
            value = oracle.renew(oracle.handles[command.slot], command.ttl, now)
        elif command.action == "release":
            value = oracle.release(oracle.handles[command.slot])
        elif command.action == "fence":
            value = oracle.fence(oracle.handles[command.slot])
        else:
            value = tuple(newly_expired)
        evidence: Any
        if command.action == "expire":
            evidence = tuple(_oracle_lease_evidence(lease) for lease in value)
        elif command.action == "fence":
            evidence = value
        else:
            evidence = _oracle_lease_evidence(value)
        results.append((command.action, evidence))

    observed_at = next_time - 1
    extra: Any = ()
    if adapter.monotonic:
        scope = adapter.primary_scope
        extra = {
            "next_monotonic_id": oracle.next_monotonic[scope],
            "reusable_spans": _merged_reusable(oracle.reusable[scope]),
            "committed": (),
        }
    elif adapter.metadata_labels is not None:
        size_class, hazard = adapter.metadata_labels
        extra = tuple(
            {
                "lease": _oracle_lease_evidence(lease),
                "size_class": size_class,
                "hazard": hazard,
            }
            for scope in adapter.scopes
            for lease in oracle.leases[scope]
        )
    return ApplicationOutcome(
        tuple(results),
        {"pools": _oracle_group_state(oracle, observed_at), "extra": extra},
        _oracle_counters(oracle, observed_at),
    )


def run_prepared_benchmark(
    *, commands: tuple[LeaseCommand, ...], adapter: LeasingBenchmarkAdapter
) -> ApplicationSample:
    """Time the public calls and attest their same-instance evidence."""
    return run_application_case(
        scenario_id=adapter.scenario_id,
        operations=len(commands),
        execute=lambda: execute_commands(commands, adapter),
        observe=lambda raw: observe_engine(raw, adapter),
        oracle=lambda: oracle_outcome(commands, adapter),
    )


def database_snapshot_extra(snapshot: Any) -> dict[str, Any]:
    """Normalize database-only cursor and reusable state."""
    return {
        "next_monotonic_id": snapshot.next_monotonic_id,
        "reusable_spans": _span_evidence(snapshot.reusable_spans),
        "committed": tuple(
            {
                "resource": (batch.resource.start, batch.resource.end),
                "token": batch.token,
                "committed_at": batch.committed_at,
            }
            for batch in snapshot.committed
        ),
    }


def warehouse_snapshot_extra(snapshot: Any) -> tuple[dict[str, Any], ...]:
    """Normalize warehouse compatibility metadata alongside every lease."""
    return tuple(
        {
            "lease": _lease_evidence(handle),
            "size_class": handle.size_class,
            "hazard": handle.hazard,
        }
        for handle in snapshot.leases
    )


def no_snapshot_extra(snapshot: Any) -> tuple[()]:
    """Return the common empty scenario-specific state marker."""
    del snapshot
    return ()
