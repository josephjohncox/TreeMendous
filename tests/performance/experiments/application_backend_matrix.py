#!/usr/bin/env python3
"""Concrete-application backend qualification experiment.

The experiment is intentionally outside :mod:`treemendous`.  It temporarily
replaces only the private RangeSet factories owned by the application kernels;
public constructors and stable runtime behavior are never changed.  Every
replacement delegates to one semantically probed :class:`BackendRegistry`.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sysconfig
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from treemendous.applications._shared import allocation, claiming, leasing
from treemendous.applications._shared.allocation import ContiguousAllocator, FitPolicy
from treemendous.applications._shared.claiming import ClaimLedger
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator
from treemendous.applications.leasing._common import PoolGroup
from treemendous.applications.partitioning._runtime import PartitionRuntime
from treemendous.backends.registry import BackendRegistry
from treemendous.backends.types import (
    Available,
    BackendSpec,
    Capability,
    Invalid,
    Maturity,
    Unavailable,
)
from treemendous.domain import Span
from treemendous.rangeset import RangeSet

SCHEMA = "treemendous-application-backend-matrix-v1"
BASELINE_BACKEND = "py_boundary"
ENGINE_NAMES = (
    "contiguous_allocator",
    "disk_block_allocator",
    "pool_group",
    "claim_ledger",
    "partition_runtime",
)
MINIMUM_BLOCKS = 20
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 591_337
PRIMARY_UPPER_95_LIMIT = 0.90
CELL_UPPER_95_LIMIT = 1.10
CURRENT_DEFAULT_UPPER_95_LIMIT = 1.10
BUILD_FLAG_NAMES = (
    "BOOST_ROOT",
    "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
    "TREE_MENDOUS_GLIBCXX_DEBUG",
    "TREE_MENDOUS_LOCAL_NATIVE",
    "TREE_MENDOUS_SANITIZERS",
    "TREE_MENDOUS_WITH_ICL",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATHS = (
    "tests/performance/experiments/application_backend_matrix.py",
    "treemendous/backends/registry.py",
    "treemendous/backends/selection.py",
    "treemendous/applications/_shared/allocation.py",
    "treemendous/applications/_shared/leasing.py",
    "treemendous/applications/_shared/claiming.py",
    "treemendous/applications/leasing/_common.py",
    "treemendous/applications/partitioning/_runtime.py",
    "treemendous/applications/allocation/disk_blocks.py",
)
T = TypeVar("T")


class QualificationError(ValueError):
    """A requested backend failed the experiment's fixed eligibility rules."""


@dataclasses.dataclass
class _FactoryTracker:
    backend: str
    expected_type: type[Any]
    objects: list[Any] = dataclasses.field(default_factory=list)

    def record(self, ranges: RangeSet) -> RangeSet:
        implementation = ranges._adapter.implementation
        if type(implementation) is not self.expected_type:
            raise AssertionError(
                f"factory for {self.backend} constructed {type(implementation)!r}"
            )
        self.objects.append(implementation)
        return ranges

    def evidence(self) -> dict[str, Any]:
        if not self.objects:
            raise AssertionError("application trace did not call a patched factory")
        ids = [id(item) for item in self.objects]
        if len(ids) != len(set(ids)):
            raise AssertionError("factory implementation identities were reused")
        return {
            "calls": len(ids),
            "implementation_ids": ids,
            "implementation_module": self.expected_type.__module__,
            "implementation_type": self.expected_type.__qualname__,
        }


class _PatchedFactoryContext:
    """Temporarily route private application factories through one registry spec."""

    def __init__(self, registry: BackendRegistry, spec: BackendSpec) -> None:
        self._registry = registry
        self._spec = spec
        self._counter = 0
        self._tracker = _FactoryTracker(spec.id, spec.loader())
        self._originals: tuple[Any, ...] | None = None

    def _new_ranges(self, domain: Any) -> RangeSet:
        return self._tracker.record(
            self._registry.create(
                domain,
                backend=self._spec.id,
                initially_available=True,
            )
        )

    def _new_allocation_ranges(self, domain: Any) -> RangeSet:
        return self._new_ranges(domain)

    def _new_claim_ranges(self, domain: Any, claims: tuple[Any, ...] = ()) -> RangeSet:
        ranges = self._new_ranges(domain)
        for claim in claims:
            if claim.state in {
                claiming.ClaimState.ACTIVE,
                claiming.ClaimState.COMPLETED,
            }:
                mutation = ranges.discard(claim.span, require_covered=True)
                if not mutation.fully_covered:
                    raise claiming.ClaimInvariantError(
                        "consuming claims overlap or leave the domain"
                    )
        return ranges

    def _next_hex(self, _length: int | None = 16) -> str:
        self._counter += 1
        return f"experiment-{self._counter:08d}"

    def _next_uuid(self) -> UUID:
        self._counter += 1
        return UUID(int=self._counter)

    def __enter__(self) -> _FactoryTracker:
        context = self

        def new_lease_ranges(instance: Any) -> RangeSet:
            return context._new_ranges(instance._domain)

        self._originals = (
            allocation._new_range_set,
            leasing.LeasePool._new_range_set,
            claiming._new_free,
            allocation.uuid4,
            leasing.secrets.token_hex,
            claiming.secrets.token_hex,
        )
        setattr(allocation, "_new_range_set", self._new_allocation_ranges)
        setattr(leasing.LeasePool, "_new_range_set", new_lease_ranges)
        setattr(claiming, "_new_free", self._new_claim_ranges)
        setattr(allocation, "uuid4", self._next_uuid)
        setattr(leasing.secrets, "token_hex", self._next_hex)
        setattr(claiming.secrets, "token_hex", self._next_hex)
        return self._tracker

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        del exc_type, exc_value, traceback
        originals = self._originals
        if originals is None:
            raise RuntimeError("patched factory context was not entered")
        (
            old_allocation,
            old_lease,
            old_claim,
            old_uuid,
            old_lease_hex,
            old_claim_hex,
        ) = originals
        setattr(allocation, "_new_range_set", old_allocation)
        setattr(leasing.LeasePool, "_new_range_set", old_lease)
        setattr(claiming, "_new_free", old_claim)
        setattr(allocation, "uuid4", old_uuid)
        setattr(leasing.secrets, "token_hex", old_lease_hex)
        setattr(claiming.secrets, "token_hex", old_claim_hex)
        self._originals = None


def _patched_factories(
    registry: BackendRegistry, spec: BackendSpec
) -> _PatchedFactoryContext:
    """Return a scoped private-factory patch for one qualified backend."""
    return _PatchedFactoryContext(registry, spec)


def _rejection_reasons(spec: BackendSpec, state: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    if isinstance(state, Unavailable):
        reasons.append(f"unavailable: {state.reason}")
    elif isinstance(state, Invalid):
        reasons.append(f"invalid: {state.error}")
    elif not isinstance(state, Available):
        reasons.append("unavailable: missing semantic probe")
    if spec.maturity is not Maturity.STABLE:
        reasons.append("experimental")
    if spec.coordinate_bits != 64:
        reasons.append("coordinate width is not signed 64-bit")
    if not spec.deterministic:
        reasons.append("nondeterministic")
    if Capability.CORE not in spec.capabilities:
        reasons.append("missing CORE capability declaration")
    if (
        isinstance(state, Available)
        and Capability.CORE not in state.validated_capabilities
    ):
        reasons.append("CORE capability was not semantically validated")
    return tuple(reasons)


def _qualify_backend(registry: BackendRegistry, backend: str) -> BackendSpec:
    spec = next((item for item in registry.specs if item.id == backend), None)
    if spec is None:
        raise QualificationError(f"unknown backend: {backend}")
    reasons = _rejection_reasons(spec, registry.states[backend])
    if reasons:
        raise QualificationError(f"backend {backend} rejected: {'; '.join(reasons)}")
    # Construction must traverse the canonical registry resolution path.  The
    # temporary object is deliberately discarded before application tracing.
    registry.create((-(2**63), -(2**63) + 1), backend=backend)
    return spec


def _eligible_specs(registry: BackendRegistry) -> tuple[BackendSpec, ...]:
    result = []
    for spec in registry.specs:
        try:
            result.append(_qualify_backend(registry, spec.id))
        except QualificationError:
            pass
    if not any(spec.id == BASELINE_BACKEND for spec in result):
        raise RuntimeError("qualified py_boundary baseline is unavailable")
    return tuple(result)


def _freeze(value: Any) -> Any:
    """Convert application evidence to exact, canonical JSON values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("application evidence contains a non-finite float")
        return value
    if isinstance(value, enum.Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _freeze(value.value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _freeze(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            str(key): _freeze(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_freeze(item) for item in value]
    raise TypeError(f"cannot freeze application evidence of type {type(value)!r}")


def _observed_call(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"result": _freeze(call())}
    except Exception as error:  # noqa: BLE001 - exceptions are experiment evidence
        return {
            "exception": {
                "module": type(error).__module__,
                "type": type(error).__qualname__,
                "message": str(error),
            }
        }


def _contiguous_action() -> Callable[[], Any]:
    allocator = ContiguousAllocator((-64, 256), reserved=((-64, -56), (120, 128)))
    seed_left = allocator.allocate(13, owner="seed-left", alignment=4)
    seed_right = allocator.allocate(9, owner="seed-right", alignment=8)
    allocator.free(seed_left, owner="seed-left")

    def action() -> Any:
        calls = []
        calls.append(
            _observed_call(
                lambda: allocator.allocate(
                    11,
                    owner="alpha",
                    alignment=4,
                    policy=FitPolicy.BEST,
                    idempotency_key="a",
                )
            )
        )
        calls.append(
            _observed_call(
                lambda: allocator.reserve(
                    80, 7, owner="beta", alignment=8, idempotency_key="b"
                )
            )
        )
        checkpoint = allocator.checkpoint()
        temporary = allocator.allocate(5, owner="temporary", policy=FitPolicy.WORST)
        allocator.free(temporary, owner="temporary")
        allocator.reserve_hole((144, 151))
        allocator.restore(checkpoint)
        calls.append(_observed_call(lambda: allocator.reserve(80, 2, owner="conflict")))
        replay = allocator.allocate(3, owner="replay", alignment=2)
        calls.append({"result": _freeze(replay)})
        return {
            "calls": calls,
            "checkpoint": _freeze(checkpoint),
            "diagnostics": _freeze(allocator.diagnostics()),
            "snapshot": _freeze(allocator.snapshot()),
            "final_geometry": _freeze(allocator.snapshot().free_ranges),
            "seed_right": _freeze(seed_right),
        }

    return action


def _disk_action() -> Callable[[], Any]:
    disk = DiskBlockAllocator(256, metadata_blocks=4, fit_policy=FitPolicy.BEST)
    old = disk.allocate_extent("seed", 9)
    disk.allocate_extent("resident", 13)
    disk.free_extent(old, file_id="seed")

    def action() -> Any:
        calls = [
            _observed_call(lambda: disk.allocate_extent("alpha", 17)),
            _observed_call(
                lambda: disk.allocate_extent("beta", 7, policy=FitPolicy.WORST)
            ),
        ]
        checkpoint = disk.checkpoint()
        temporary = disk.allocate_extent("temporary", 5)
        disk.free_extent(temporary, file_id="temporary")
        disk.restore(checkpoint)
        calls.append(_observed_call(lambda: disk.allocate_extent("oversized", 10_000)))
        calls.append(_observed_call(lambda: disk.allocate_extent("replay", 3)))
        snapshot = disk.snapshot()
        return {
            "calls": calls,
            "checkpoint": _freeze(checkpoint),
            "diagnostics": _freeze(snapshot.diagnostics),
            "snapshot": _freeze(snapshot),
            "final_geometry": _freeze(snapshot.free_extents),
        }

    return action


def _pool_group_action() -> Callable[[], Any]:
    clock = LogicalClock(100)
    group = PoolGroup(
        {"alpha": (Span(-32, 64),), "beta": (Span(100, 180),)}, clock=clock
    )

    def action() -> Any:
        first = group.acquire("alpha", "worker-a", ttl=50, size=7, request_id="a")
        second = group.acquire("beta", "worker-b", ttl=40, size=5, request_id="b")
        renewed = group.renew(first, ttl=80)
        released = group.release(second)
        checkpoint = group.checkpoint()
        restored = PoolGroup.restore(checkpoint, clock=clock)
        replay = restored.acquire("alpha", "worker-c", ttl=20, size=3, request_id="c")
        failure = _observed_call(lambda: restored.release(renewed))
        return {
            "results": _freeze((first, second, renewed, released, replay)),
            "exception": failure,
            "snapshot": _freeze(group.snapshot()),
            "diagnostics": _freeze(group.diagnostics()),
            "checkpoint": _freeze(checkpoint),
            "restored_snapshot": _freeze(restored.snapshot()),
            "restored_checkpoint": _freeze(restored.checkpoint()),
            "final_geometry": _freeze(
                tuple(
                    (scope, pool.snapshot().available_spans)
                    for scope, pool in sorted(restored.pools.items())
                )
            ),
        }

    return action


def _claim_ledger_action() -> Callable[[], Any]:
    clock = LogicalClock(100)
    ledger = ClaimLedger((-16, 128), clock=clock)

    def action() -> Any:
        first = ledger.claim_next("worker-a", 11, request_id="a", ttl=50)
        second = ledger.claim_next("worker-b", 7, request_id="b")
        renewed = ledger.renew(first, ttl=70)
        abandoned = ledger.abandon(second)
        checkpoint = ledger.checkpoint()
        restored = ClaimLedger.from_checkpoint(checkpoint, clock=clock)
        replay = restored.claim_next("worker-c", 5, request_id="c")
        failure = _observed_call(lambda: restored.complete(first))
        completed = restored.complete(replay, result={"rows": 5})
        snapshot = restored.snapshot()
        return {
            "results": _freeze((first, second, renewed, abandoned, replay, completed)),
            "exception": failure,
            "snapshot": _freeze(snapshot),
            "diagnostics": _freeze(snapshot.diagnostics),
            "checkpoint": _freeze(checkpoint),
            "restored_checkpoint": _freeze(restored.checkpoint()),
            "events": _freeze(restored.events()),
            "violations": _freeze(restored.invariant_violations()),
            "final_geometry": _freeze(snapshot.available),
        }

    return action


def _partition_runtime_action() -> Callable[[], Any]:
    clock = LogicalClock(100)
    runtime = PartitionRuntime(96, clock=clock)

    def action() -> Any:
        first = runtime.claim("worker-a", 13, request_id="a")
        completed = runtime.complete(first, "indexed", {"rows": 13})
        second = runtime.claim("worker-b", 9, request_id="b")
        abandoned = runtime.abandon(second)
        checkpoint = runtime.checkpoint()
        restored = PartitionRuntime.from_checkpoint(checkpoint, clock=clock)
        replay = restored.claim("worker-c", 7, request_id="c")
        replay_completed = restored.complete(replay, "replayed", {"rows": 7})
        audit, restored_checkpoint = restored.audit_snapshot(restored.ledger.snapshot)
        failure = _observed_call(lambda: restored.complete(first, "stale"))
        return {
            "results": _freeze(
                (first, completed, second, abandoned, replay, replay_completed)
            ),
            "exception": failure,
            "snapshot": _freeze(audit),
            "diagnostics": _freeze(audit.diagnostics),
            "checkpoint": _freeze(checkpoint),
            "restored_checkpoint": _freeze(restored_checkpoint),
            "events": _freeze(restored.log.snapshot()),
            "final_geometry": _freeze(audit.available),
        }

    return action


_ENGINE_FACTORIES: dict[str, Callable[[], Callable[[], Any]]] = {
    "contiguous_allocator": _contiguous_action,
    "disk_block_allocator": _disk_action,
    "pool_group": _pool_group_action,
    "claim_ledger": _claim_ledger_action,
    "partition_runtime": _partition_runtime_action,
}


def _execute_once(
    registry: BackendRegistry, spec: BackendSpec, engine: str
) -> tuple[int, Any, dict[str, Any]]:
    with _patched_factories(registry, spec) as tracker:
        action = _ENGINE_FACTORIES[engine]()  # setup is outside the timer
        started = time.perf_counter_ns()
        semantic = action()
        elapsed = time.perf_counter_ns() - started
        factory_evidence = tracker.evidence()
    return elapsed, semantic, factory_evidence


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _bootstrap(values: Sequence[float], seed: int) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    rng = random.Random(seed)
    try:
        medians = sorted(
            statistics.median(rng.choices(values, k=len(values)))
            for _ in range(BOOTSTRAP_RESAMPLES)
        )
        median = float(statistics.median(values))
        lower = float(medians[int(0.025 * len(medians))])
        upper = float(medians[int(0.975 * len(medians))])
    except (IndexError, TypeError, ValueError, statistics.StatisticsError) as exc:
        raise ValueError("bootstrap values are invalid") from exc
    return {
        "median": median,
        "median_95_low": lower,
        "median_95_high": upper,
    }


def _correctness_rows(
    registry: BackendRegistry,
    specs: Sequence[BackendSpec],
    engines: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline: dict[str, Any] = {}
    baseline_factories: dict[str, Any] = {}
    baseline_spec = next(spec for spec in specs if spec.id == BASELINE_BACKEND)
    for engine in engines:
        _, semantic, factories = _execute_once(registry, baseline_spec, engine)
        baseline[engine] = semantic
        baseline_factories[engine] = factories

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for engine in engines:
            if spec.id == BASELINE_BACKEND:
                semantic = baseline[engine]
                factories = baseline_factories[engine]
            else:
                _, semantic, factories = _execute_once(registry, spec, engine)
            if semantic != baseline[engine]:
                raise AssertionError(
                    f"{spec.id}/{engine} differs from exact py_boundary evidence"
                )
            rows.append(
                {
                    "backend": spec.id,
                    "engine": engine,
                    "semantic": semantic,
                    "semantic_sha256": _sha(semantic),
                    "baseline_sha256": _sha(baseline[engine]),
                    "factory": factories,
                }
            )
    return rows, baseline


def _measure_row(
    registry: BackendRegistry,
    baseline: BackendSpec,
    candidate: BackendSpec,
    engine: str,
    blocks: int,
    expected: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    raw_blocks = []
    ratios = []
    for block in range(blocks):
        labels = (
            ("candidate", "baseline") if block % 2 == 0 else ("baseline", "candidate")
        )
        order = (candidate, baseline) if block % 2 == 0 else (baseline, candidate)
        elapsed: dict[str, int] = {}
        for label, spec in zip(labels, order, strict=True):
            duration, semantic, _ = _execute_once(registry, spec, engine)
            # Correctness is checked on the exact instance after its timed call.
            if semantic != expected:
                raise AssertionError(
                    f"timed {spec.id}/{engine} result differs from baseline"
                )
            elapsed[label] = duration
        ratio = elapsed["candidate"] / max(1, elapsed["baseline"])
        ratios.append(ratio)
        raw_blocks.append(
            {
                "block": block,
                "order": list(labels),
                "candidate_ns": elapsed["candidate"],
                "baseline_ns": elapsed["baseline"],
                "ratio": ratio,
            }
        )
    seed = BOOTSTRAP_SEED + sum(ord(char) for char in candidate.id + engine + phase)
    return {
        "phase": phase,
        "backend": candidate.id,
        "engine": engine,
        "blocks": raw_blocks,
        "ratios": ratios,
        "ratio": _bootstrap(ratios, seed),
        "validated_blocks": blocks,
        "semantic_sha256": _sha(expected),
    }


def _gate(
    specs: Sequence[BackendSpec],
    initial: Sequence[dict[str, Any]],
    confirmation: Sequence[dict[str, Any]],
    engines: Sequence[str],
) -> dict[str, Any]:
    by_backend = {
        spec.id: [row for row in initial if row["backend"] == spec.id] for spec in specs
    }
    initial_candidates = []
    for backend, rows in by_backend.items():
        if backend == BASELINE_BACKEND:
            continue
        if (
            len(rows) == len(engines)
            and min(row["ratio"]["median_95_high"] for row in rows)
            <= PRIMARY_UPPER_95_LIMIT
            and max(row["ratio"]["median_95_high"] for row in rows)
            <= CELL_UPPER_95_LIMIT
        ):
            initial_candidates.append(backend)
    confirmed = []
    for backend in initial_candidates:
        rows = [row for row in confirmation if row["backend"] == backend]
        if (
            len(rows) == len(engines)
            and min(row["ratio"]["median_95_high"] for row in rows)
            <= PRIMARY_UPPER_95_LIMIT
            and max(row["ratio"]["median_95_high"] for row in rows)
            <= CELL_UPPER_95_LIMIT
        ):
            confirmed.append(backend)
    default_rows = by_backend[BASELINE_BACKEND]
    default_pass = (
        len(default_rows) == len(engines)
        and max(row["ratio"]["median_95_high"] for row in default_rows)
        <= CURRENT_DEFAULT_UPPER_95_LIMIT
    )
    qualified = confirmed if default_pass else []
    return {
        "decision": "QUALIFIED" if qualified else "REJECTED",
        "primary_upper_95_limit": PRIMARY_UPPER_95_LIMIT,
        "cell_upper_95_limit": CELL_UPPER_95_LIMIT,
        "current_default_upper_95_limit": CURRENT_DEFAULT_UPPER_95_LIMIT,
        "current_default_pass": default_pass,
        "initial_candidates": initial_candidates,
        "confirmed_backends": confirmed,
        "qualified_backends": qualified,
        "runtime_injection_retained": False,
    }


def _negative_case(
    case: str, registry: BackendRegistry, backend: str
) -> dict[str, Any]:
    try:
        _qualify_backend(registry, backend)
    except Exception as error:  # noqa: BLE001 - fail-closed evidence
        return {
            "case": case,
            "backend": backend,
            "accepted": False,
            "fallback_backend": None,
            "exception_type": type(error).__qualname__,
            "reason": str(error),
        }
    raise AssertionError(f"negative qualification case {case!r} was accepted")


def _negative_evidence(registry: BackendRegistry) -> list[dict[str, Any]]:
    specs = registry.specs
    baseline = next(spec for spec in specs if spec.id == BASELINE_BACKEND)
    unavailable = BackendRegistry(
        specs,
        {**registry.states, baseline.id: Unavailable("synthetic unavailable probe")},
    )
    invalid = BackendRegistry(
        specs, {**registry.states, baseline.id: Invalid("synthetic invalid probe")}
    )
    experimental = next(
        spec for spec in specs if spec.maturity is Maturity.EXPERIMENTAL
    )
    width = next(spec for spec in specs if spec.coordinate_bits != 64)
    nondeterministic = next(spec for spec in specs if not spec.deterministic)
    return [
        _negative_case("unknown", registry, "does-not-exist"),
        _negative_case("unavailable", unavailable, baseline.id),
        _negative_case("invalid", invalid, baseline.id),
        _negative_case("experimental", registry, experimental.id),
        _negative_case("32-bit", registry, width.id),
        _negative_case("nondeterministic", registry, nondeterministic.id),
    ]


def _methodology(blocks: int, engines: Sequence[str]) -> dict[str, Any]:
    return {
        "baseline_backend": BASELINE_BACKEND,
        "engines": list(engines),
        "minimum_blocks": MINIMUM_BLOCKS,
        "blocks": blocks,
        "balanced_order": "candidate/baseline alternates by block",
        "setup_timing": "application construction and seed state excluded",
        "correctness_timing": "same timed instance checked after timing",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_upper_95_limit": PRIMARY_UPPER_95_LIMIT,
        "cell_upper_95_limit": CELL_UPPER_95_LIMIT,
        "current_default_upper_95_limit": CURRENT_DEFAULT_UPPER_95_LIMIT,
    }


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    changed = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    staged = sorted(
        path for path in git("diff", "--cached", "--name-only").splitlines() if path
    )
    digest = hashlib.sha256()
    digest.update(git("rev-parse", "HEAD").strip().encode())
    for relative in changed:
        path = _REPOSITORY_ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "head_tree": git("rev-parse", "HEAD^{tree}").strip(),
        "clean_worktree": not bool(status),
        "changed_paths": changed,
        "staged_files": staged,
        "source_state_sha256": digest.hexdigest(),
    }


def _file(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _backend_provenance(specs: Sequence[BackendSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        implementation = spec.loader()
        path = Path(inspect.getfile(implementation))
        rows.append(
            {
                "id": spec.id,
                "module": implementation.__module__,
                "type": implementation.__qualname__,
                "runtime": spec.runtime.value,
                "coordinate_bits": spec.coordinate_bits,
                "deterministic": spec.deterministic,
                "binary": _file(path),
            }
        )
    return rows


def _compiler_version() -> str:
    compiler = os.environ.get("CXX", "c++")
    try:
        result = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.splitlines()[0] if result.stdout else "unavailable"


def _provenance(specs: Sequence[BackendSpec]) -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "sources": [_file(_REPOSITORY_ROOT / relative) for relative in _SOURCE_PATHS],
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "build": {
            "command": os.environ.get(
                "TREE_MENDOUS_BUILD_COMMAND", "uv sync --all-extras"
            ),
            "cxx": os.environ.get("CXX", "c++"),
            "cxx_version": _compiler_version(),
            "cc": str(sysconfig.get_config_var("CC") or "unknown"),
            "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
            "flags": {name: os.environ.get(name, "") for name in BUILD_FLAG_NAMES},
        },
        "backends": _backend_provenance(specs),
    }


def run_matrix(
    *,
    blocks: int = MINIMUM_BLOCKS,
    backend_ids: Sequence[str] | None = None,
    engines: Sequence[str] = ENGINE_NAMES,
) -> dict[str, Any]:
    """Run the bounded correctness and paired qualification matrix."""
    if type(blocks) is not int or blocks < MINIMUM_BLOCKS:
        raise ValueError(f"blocks must be an integer >= {MINIMUM_BLOCKS}")
    expected_engines = tuple(engine for engine in ENGINE_NAMES if engine in engines)
    if not engines or tuple(engines) != expected_engines:
        raise ValueError(
            "engines must be an ordered unique nonempty subset of the fixed matrix"
        )
    registry = BackendRegistry.discover()
    eligible = _eligible_specs(registry)
    if backend_ids is not None:
        requested = tuple(backend_ids)
        if BASELINE_BACKEND not in requested:
            requested = (BASELINE_BACKEND, *requested)
        eligible = tuple(spec for spec in eligible if spec.id in requested)
        missing = set(requested) - {spec.id for spec in eligible}
        if missing:
            raise QualificationError(
                f"requested backends are not qualified: {sorted(missing)}"
            )
    correctness, baseline_semantics = _correctness_rows(registry, eligible, engines)
    baseline = next(spec for spec in eligible if spec.id == BASELINE_BACKEND)
    initial = [
        _measure_row(
            registry,
            baseline,
            spec,
            engine,
            blocks,
            baseline_semantics[engine],
            phase="initial",
        )
        for spec in eligible
        for engine in engines
    ]
    preliminary = _gate(eligible, initial, (), engines)
    confirmation = [
        _measure_row(
            registry,
            baseline,
            spec,
            engine,
            blocks,
            baseline_semantics[engine],
            phase="confirmation",
        )
        for spec in eligible
        if spec.id in preliminary["initial_candidates"]
        for engine in engines
    ]
    gate = _gate(eligible, initial, confirmation, engines)
    return {
        "schema": SCHEMA,
        "methodology": _methodology(blocks, engines),
        "eligible_backends": [spec.id for spec in eligible],
        "negative_requests": _negative_evidence(registry),
        "correctness": correctness,
        "benchmarks": initial,
        "confirmation": confirmation,
        "gate": gate,
        "provenance": _provenance(eligible),
    }


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Concrete application backend qualification experiment",
        "",
        f"Decision: **{report['gate']['decision']}**",
        f"Runtime injection retained: **{report['gate']['runtime_injection_retained']}**",
        f"JSON SHA-256: `{digest}`",
        "",
        "| phase | backend | engine | median ratio | upper 95% |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in [*report["benchmarks"], *report["confirmation"]]:
        lines.append(
            f"| {row['phase']} | {row['backend']} | {row['engine']} | "
            f"{row['ratio']['median']:.4f} | {row['ratio']['median_95_high']:.4f} |"
        )
    lines.extend(("", "## Negative requests", ""))
    lines.extend(
        f"- `{row['case']}` / `{row['backend']}`: {row['reason']}"
        for row in report["negative_requests"]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report, digest))
    checksum = Path(f"{output}.sha256")
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, markdown, checksum


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _finite(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid number: {value}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite number: {value}")
    return result


def _exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _verify_benchmark_row(row: Any, blocks: int) -> None:
    keys = {
        "phase",
        "backend",
        "engine",
        "blocks",
        "ratios",
        "ratio",
        "validated_blocks",
        "semantic_sha256",
    }
    if type(row) is not dict or set(row) != keys:
        raise ValueError("benchmark row schema mismatch")
    if (
        type(row["phase"]) is not str
        or row["phase"] not in {"initial", "confirmation"}
        or type(row["backend"]) is not str
        or type(row["engine"]) is not str
        or row["engine"] not in ENGINE_NAMES
    ):
        raise ValueError("benchmark row identity exact type mismatch")
    if type(row["validated_blocks"]) is not int or row["validated_blocks"] != blocks:
        raise ValueError("validated block count mismatch")
    raw = row["blocks"]
    if type(raw) is not list or len(raw) != blocks:
        raise ValueError("raw block count mismatch")
    ratios = []
    for index, block in enumerate(raw):
        if type(block) is not dict or set(block) != {
            "block",
            "order",
            "candidate_ns",
            "baseline_ns",
            "ratio",
        }:
            raise ValueError("raw block schema mismatch")
        expected_order = (
            ["candidate", "baseline"] if index % 2 == 0 else ["baseline", "candidate"]
        )
        if block["block"] != index or block["order"] != expected_order:
            raise ValueError("balanced block ordering mismatch")
        if (
            type(block["candidate_ns"]) is not int
            or type(block["baseline_ns"]) is not int
            or min(block["candidate_ns"], block["baseline_ns"]) <= 0
        ):
            raise ValueError("block duration exact type/value mismatch")
        expected = block["candidate_ns"] / block["baseline_ns"]
        if type(block["ratio"]) is not float or block["ratio"] != expected:
            raise ValueError("raw block ratio mismatch")
        ratios.append(expected)
    if not _exact(row["ratios"], ratios):
        raise ValueError("derived ratio samples mismatch")
    seed = BOOTSTRAP_SEED + sum(
        ord(char) for char in row["backend"] + row["engine"] + row["phase"]
    )
    if not _exact(row["ratio"], _bootstrap(ratios, seed)):
        raise ValueError("derived ratio interval mismatch")


def verify_artifacts(output: Path) -> dict[str, Any]:
    """Strictly parse and recompute the canonical artifact triplet."""
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if Path(f"{output}.sha256").read_text() != f"{digest}  {output.name}\n":
        raise ValueError("checksum mismatch")
    try:
        report = json.loads(
            encoded,
            object_pairs_hook=_duplicates,
            parse_constant=_nonfinite,
            parse_float=_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("application-backend JSON is invalid") from exc
    top = {
        "schema",
        "methodology",
        "eligible_backends",
        "negative_requests",
        "correctness",
        "benchmarks",
        "confirmation",
        "gate",
        "provenance",
    }
    if type(report) is not dict or set(report) != top or report["schema"] != SCHEMA:
        raise ValueError("report schema mismatch")
    methodology = report["methodology"]
    if type(methodology) is not dict:
        raise ValueError("methodology schema mismatch")
    blocks = methodology.get("blocks")
    engines = methodology.get("engines")
    if type(blocks) is not int or blocks < MINIMUM_BLOCKS or type(engines) is not list:
        raise ValueError("methodology exact type/value mismatch")
    expected_engines = [engine for engine in ENGINE_NAMES if engine in engines]
    if engines != expected_engines:
        raise ValueError("methodology engine order/uniqueness mismatch")
    if not _exact(methodology, _methodology(blocks, engines)):
        raise ValueError("fixed methodology mismatch")
    eligible_ids = report["eligible_backends"]
    if (
        type(eligible_ids) is not list
        or not eligible_ids
        or any(type(item) is not str for item in eligible_ids)
    ):
        raise ValueError("eligible backend schema mismatch")
    registry = BackendRegistry.discover()
    if len(set(eligible_ids)) != len(eligible_ids):
        raise ValueError("eligible backend order/uniqueness mismatch")
    available_order = [spec.id for spec in _eligible_specs(registry)]
    expected_eligible_order = [item for item in available_order if item in eligible_ids]
    if eligible_ids != expected_eligible_order or BASELINE_BACKEND not in eligible_ids:
        raise ValueError("eligible backend order/uniqueness mismatch")
    specs = tuple(_qualify_backend(registry, backend) for backend in eligible_ids)
    if not _exact(report["negative_requests"], _negative_evidence(registry)):
        raise ValueError("negative fail-closed evidence mismatch")
    expected_pairs = [
        (backend, engine) for backend in eligible_ids for engine in engines
    ]
    observed_pairs: list[tuple[str, str]] = []
    semantics: dict[tuple[str, str], Any] = {}
    if type(report["correctness"]) is not list:
        raise ValueError("correctness list type mismatch")
    for row in report["correctness"]:
        keys = {
            "backend",
            "engine",
            "semantic",
            "semantic_sha256",
            "baseline_sha256",
            "factory",
        }
        if (
            type(row) is not dict
            or set(row) != keys
            or type(row["backend"]) is not str
            or type(row["engine"]) is not str
        ):
            raise ValueError("correctness row schema/type mismatch")
        pair = (row["backend"], row["engine"])
        observed_pairs.append(pair)
        semantics[pair] = row["semantic"]
        if row["semantic_sha256"] != _sha(row["semantic"]) or row[
            "baseline_sha256"
        ] != _sha(row["semantic"]):
            raise ValueError("correctness digest/parity mismatch")
        factory = row["factory"]
        if type(factory) is not dict or set(factory) != {
            "calls",
            "implementation_ids",
            "implementation_module",
            "implementation_type",
        }:
            raise ValueError("factory evidence schema mismatch")
        if (
            type(factory["calls"]) is not int
            or factory["calls"] <= 0
            or type(factory["implementation_ids"]) is not list
            or len(factory["implementation_ids"]) != factory["calls"]
            or len(set(factory["implementation_ids"])) != factory["calls"]
            or any(type(item) is not int for item in factory["implementation_ids"])
        ):
            raise ValueError("factory call/identity evidence mismatch")
        spec = next(item for item in specs if item.id == row["backend"])
        implementation = spec.loader()
        if (
            factory["implementation_module"] != implementation.__module__
            or factory["implementation_type"] != implementation.__qualname__
        ):
            raise ValueError("factory backend provenance mismatch")
    if observed_pairs != expected_pairs or len(set(observed_pairs)) != len(
        expected_pairs
    ):
        raise ValueError("correctness matrix order/uniqueness mismatch")
    for backend in eligible_ids:
        for engine in engines:
            if not _exact(
                semantics[(backend, engine)], semantics[(BASELINE_BACKEND, engine)]
            ):
                raise ValueError("exact py_boundary semantic parity mismatch")
    if (
        type(report["benchmarks"]) is not list
        or type(report["confirmation"]) is not list
    ):
        raise ValueError("benchmark/confirmation list type mismatch")
    for row in report["benchmarks"]:
        _verify_benchmark_row(row, blocks)
        if row["phase"] != "initial":
            raise ValueError("benchmark phase mismatch")
        expected_sha = _sha(semantics[(BASELINE_BACKEND, row["engine"])])
        if row["semantic_sha256"] != expected_sha:
            raise ValueError("timed semantic digest mismatch")
    expected_initial = [
        (backend, engine) for backend in eligible_ids for engine in engines
    ]
    observed_initial = [(row["backend"], row["engine"]) for row in report["benchmarks"]]
    if observed_initial != expected_initial or len(set(observed_initial)) != len(
        expected_initial
    ):
        raise ValueError("benchmark matrix order/uniqueness mismatch")
    initial_gate = _gate(specs, report["benchmarks"], (), engines)
    confirmation_pairs = [
        (row.get("backend"), row.get("engine")) if type(row) is dict else (None, None)
        for row in report["confirmation"]
    ]
    expected_confirmation = [
        (backend, engine)
        for backend in initial_gate["initial_candidates"]
        for engine in engines
    ]
    if confirmation_pairs != expected_confirmation or len(
        set(confirmation_pairs)
    ) != len(expected_confirmation):
        raise ValueError("confirmation matrix order/uniqueness mismatch")
    for row in report["confirmation"]:
        _verify_benchmark_row(row, blocks)
        if row["phase"] != "confirmation":
            raise ValueError("benchmark phase mismatch")
        expected_sha = _sha(semantics[(BASELINE_BACKEND, row["engine"])])
        if row["semantic_sha256"] != expected_sha:
            raise ValueError("timed semantic digest mismatch")
    expected_gate = _gate(specs, report["benchmarks"], report["confirmation"], engines)
    if not _exact(report["gate"], expected_gate):
        raise ValueError("recomputed gate mismatch")
    if not _exact(report["provenance"], _provenance(specs)):
        raise ValueError("source/runtime/backend/binary provenance mismatch")
    if output.with_suffix(".md").read_text() != _markdown(report, digest):
        raise ValueError("Markdown mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=MINIMUM_BLOCKS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/experiments/application-backend-matrix.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        report = verify_artifacts(args.output)
        print(f"verified {args.output}: {report['gate']['decision']}")
        return
    report = run_matrix(blocks=args.blocks)
    paths = write_artifacts(report, args.output)
    verify_artifacts(args.output)
    print(f"{report['gate']['decision']}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
