#!/usr/bin/env python3
"""Ordered, payload-aware ``RangeSet`` transaction experiment.

This module is deliberately outside :mod:`treemendous`.  The candidate uses
private ``RangeSet`` state to test copy/stage/publish economics without adding a
stable API.  It never compensates a partially applied source mutation: all
fallible work targets a fresh backend, followed by one ``__dict__`` reference
publication while the original lock is held.
"""

from __future__ import annotations

import argparse
import gc
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
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from treemendous.backends.adapters import BackendAdapter
from treemendous.backends.catalog import CATALOG
from treemendous.backends.types import Capability, Maturity
from treemendous.domain import IntervalResult, MutationResult, RangeSnapshot, Span
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    PayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.rangeset import _MISSING, RangeSet

SCHEMA = "treemendous-rangeset-transaction-experiment-v1"
BACKEND = "py_boundary"
BUILD_FLAG_NAMES = (
    "BOOST_ROOT",
    "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
    "TREE_MENDOUS_GLIBCXX_DEBUG",
    "TREE_MENDOUS_LOCAL_NATIVE",
    "TREE_MENDOUS_SANITIZERS",
    "TREE_MENDOUS_WITH_ICL",
)
MINIMUM_SAMPLES = 15
BATCH_SIZES = (0, 1, 4, 16, 64)
INTERVAL_COUNTS = (64, 1_000)
PAYLOAD_KINDS = ("none", "uniform", "join", "ordered")
TRACE_KINDS = ("restorative", "non_restorative")
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 76_031
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _TransactionOperation:
    """One immutable experiment operation descriptor."""

    action: Literal["add", "discard", "strict-discard"]
    span: Span
    payload: Any = _MISSING

    def __post_init__(self) -> None:
        if self.action not in ("add", "discard", "strict-discard"):
            raise ValueError(f"unsupported transaction action: {self.action!r}")
        if self.action != "add" and self.payload is not _MISSING:
            raise ValueError("discard operations cannot carry payload")


class _UnsupportedAdapterError(ValueError):
    """Raised when an experiment source has no exact catalog factory proof."""


BackendFactory = Callable[[], Any]
ResultFactory = Callable[[tuple[Span, ...], int, bool], Any]


def _proven_backend_factory(source: RangeSet) -> tuple[str, BackendFactory]:
    """Prove factory provenance by exact implementation class and stable spec."""
    implementation_type = type(source._adapter.implementation)
    matches = []
    for spec in CATALOG:
        if not (
            spec.maturity is Maturity.STABLE
            and spec.deterministic
            and Capability.CORE in spec.capabilities
        ):
            continue
        try:
            catalog_type = spec.loader()
        except (ImportError, OSError):
            continue
        if catalog_type is implementation_type:
            matches.append(spec)
    if len(matches) != 1:
        raise _UnsupportedAdapterError(
            "transaction experiment requires one exact stable deterministic "
            "CORE catalog factory; direct/custom adapters are unsupported"
        )
    spec = matches[0]

    def factory() -> Any:
        return spec.loader()(**dict(spec.constructor_args))

    return spec.id, factory


def _coerce_operation(row: Any) -> _TransactionOperation:
    if isinstance(row, _TransactionOperation):
        return row
    if not isinstance(row, tuple) or len(row) not in (2, 3):
        raise TypeError("transaction rows must be _TransactionOperation or 2/3-tuples")
    action = row[0]
    raw_span = row[1]
    span = raw_span if isinstance(raw_span, Span) else Span(*raw_span)
    payload = row[2] if len(row) == 3 else _MISSING
    return _TransactionOperation(action, span, payload)


def _clone_source_into_stage(
    source: RangeSet,
    factory: BackendFactory,
    timings: dict[str, int] | None = None,
) -> RangeSet:
    """Build a fresh backend and clone complete geometry/payload/event state."""
    factory_started = time.perf_counter_ns()
    implementation = factory()
    staged = RangeSet(
        BackendAdapter(implementation),
        domain=source._domain,
        initially_available=False,
        payload_policy=source._payload_policy,
        payload_cloner=source._payload_cloner,
    )

    # Read the raw geometry rather than trusting a possibly deferred cache.
    geometry = tuple(source._adapter.intervals())
    backend_load_started = time.perf_counter_ns()
    for interval in geometry:
        staged._adapter.release(interval.start, interval.end)
    backend_load_finished = time.perf_counter_ns()
    staged._geometry_cache = geometry
    staged._geometry_cache_valid = True
    staged._pending_geometry_update = None
    staged._total_free = source._total_free

    if source._payload_segments is not None:
        with source._payload_processing():
            staged._owned_payload_identity = source._clone_payload(
                source._owned_payload_identity
            )
            staged._payload_segments = source._clone_segments(source._payload_segments)
            staged._ordered_events = (
                None
                if source._ordered_events is None
                else source._clone_events(source._ordered_events)
            )
    if timings is not None:
        timings["backend_load_ns"] = backend_load_finished - backend_load_started
        timings["stage_ns"] = time.perf_counter_ns() - factory_started
    return staged


def _replay(
    staged: RangeSet, operations: tuple[_TransactionOperation, ...]
) -> tuple[MutationResult, ...]:
    rows: list[MutationResult] = []
    for operation in operations:
        if operation.action == "add":
            if operation.payload is _MISSING:
                rows.append(staged.add(operation.span))
            else:
                rows.append(staged.add(operation.span, operation.payload))
        else:
            rows.append(
                staged.discard(
                    operation.span,
                    require_covered=operation.action == "strict-discard",
                )
            )
    return tuple(rows)


def _rangeset_transaction(
    source: RangeSet,
    operations: Iterable[_TransactionOperation | tuple[Any, ...]],
    *,
    _result_factory: ResultFactory | None = None,
    _timings: dict[str, int] | None = None,
) -> tuple[MutationResult, ...]:
    """Stage ordered scalar mutations and atomically publish the staged state.

    Every iterable, policy, clone, backend, and result-construction callback is
    completed while the original state is guarded and before publication.
    Failure simply drops the staged object; the source backend is never touched.
    """
    if not isinstance(source, RangeSet):
        raise TypeError("source must be an exact RangeSet")

    lock = source._lock
    lock.acquire()
    original_state = source.__dict__
    published = False
    activity_lock = source._payload_activity_lock
    guarded_payload = source._payload_policy is not None
    try:
        if source._authoritative_mutation_active or source._payload_is_active():
            raise RuntimeError(
                "reentrant transaction on the same RangeSet is not allowed"
            )
        source._authoritative_mutation_active = True
        if guarded_payload:
            with activity_lock:
                source._payload_activity += 1

        # Factory proof deliberately precedes iteration/materialization/staging.
        total_started = time.perf_counter_ns()
        _, factory = _proven_backend_factory(source)
        materialize_started = time.perf_counter_ns()
        materialized = tuple(_coerce_operation(row) for row in operations)
        stage_timings: dict[str, int] = {}
        staged = _clone_source_into_stage(source, factory, stage_timings)
        replay_started = time.perf_counter_ns()
        scalar_rows = _replay(staged, materialized)

        make_result = MutationResult if _result_factory is None else _result_factory
        prepared = tuple(
            make_result(row.changed, row.changed_length, row.fully_covered)
            for row in scalar_rows
        )
        if any(type(row) is not MutationResult for row in prepared):
            raise TypeError(
                "transaction result factory must return exact MutationResult"
            )
        if _timings is not None:
            now = time.perf_counter_ns()
            _timings.update(stage_timings)
            _timings["materialize_ns"] = (
                replay_started - materialize_started - stage_timings["stage_ns"]
            )
            _timings["replay_result_ns"] = now - replay_started
            _timings["prepublish_total_ns"] = now - total_started

        # Prepare the published dictionary completely.  The source retains its
        # synchronization objects; publication itself is one noexcept reference
        # assignment and allocates nothing.
        staged._lock = lock
        staged._payload_activity_lock = activity_lock
        staged._payload_activity = 0
        staged._authoritative_mutation_active = False
        published_state = staged.__dict__
        source.__dict__ = published_state
        published = True
        return prepared
    finally:
        if not published:
            original_state["_authoritative_mutation_active"] = False
            if guarded_payload:
                with activity_lock:
                    original_state["_payload_activity"] -= 1
        lock.release()


# ---------------------------------------------------------------------------
# Bounded paired benchmark and strict artifact triplet.


def _policy(kind: str) -> PayloadPolicy[Any] | None:
    if kind == "none":
        return None
    if kind == "uniform":
        return UniformPayloadPolicy()
    if kind == "join":
        return JoinPayloadPolicy(lambda left, right: left | right, frozenset())
    if kind == "ordered":
        return OrderedPayloadPolicy(
            lambda left, right: left + right,
            (),
            event_key_fn=lambda value: value,
        )
    raise ValueError(f"unknown payload kind: {kind}")


def _payload(kind: str, index: int) -> Any:
    if kind == "uniform":
        return index
    if kind == "join":
        return frozenset({index})
    if kind == "ordered":
        return (index,)
    return _MISSING


def _seed_ranges(backend: str, count: int, payload_kind: str) -> RangeSet:
    from treemendous import create_range_set

    ranges = create_range_set(
        (0, count * 4 + 4),
        backend=backend,
        initially_available=False,
        payload_policy=_policy(payload_kind),
    )
    # Direct construction avoids O(N^3) ordered-policy setup while preserving
    # the exact valid internal state scalar add would create for disjoint spans.
    geometry = tuple(IntervalResult(i * 4, i * 4 + 2) for i in range(count))
    for interval in geometry:
        ranges._adapter.release(interval.start, interval.end)
    ranges._geometry_cache = geometry
    ranges._total_free = count * 2
    if payload_kind != "none":
        segments = [
            IntervalResult(item.start, item.end, data=_payload(payload_kind, i))
            for i, item in enumerate(geometry)
        ]
        ranges._payload_segments = segments
        if payload_kind == "ordered":
            ranges._ordered_events = [
                ranges._ordered_event(item.span, item.data) for item in segments
            ]
    return ranges


def _trace(
    count: int, batch_size: int, payload_kind: str, trace_kind: str
) -> tuple[_TransactionOperation, ...]:
    operations: list[_TransactionOperation] = []
    for index in range(batch_size):
        target = (index // 2) % count
        base = target * 4
        action: Literal["add", "discard", "strict-discard"]
        if trace_kind == "restorative":
            action = "discard" if index % 2 == 0 else "add"
            span = Span(base, base + 2)
        else:
            action = "add" if index % 3 else "strict-discard"
            span = (
                Span(base + 2, base + 3)
                if action == "add"
                else Span(base + 3, base + 4)
            )
        payload = _payload(payload_kind, target) if action == "add" else _MISSING
        operations.append(_TransactionOperation(action, span, payload))
    return tuple(operations)


def _scalar(
    ranges: RangeSet, operations: Sequence[_TransactionOperation]
) -> tuple[MutationResult, ...]:
    return _replay(ranges, tuple(operations))


def _freeze_results(rows: Sequence[MutationResult]) -> list[Any]:
    return [
        [
            [[span.start, span.end] for span in row.changed],
            row.changed_length,
            row.fully_covered,
        ]
        for row in rows
    ]


def _freeze_snapshot(snapshot: RangeSnapshot) -> list[Any]:
    return [
        [
            item.start,
            item.end,
            item.data
            if isinstance(item.data, (str, int, float, bool)) or item.data is None
            else repr(item.data),
        ]
        for item in snapshot.intervals
    ]


def _bootstrap_interval(values: Sequence[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        medians.append(
            statistics.median(rng.choice(values) for _ in range(len(values)))
        )
    medians.sort()
    return medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians))]


def _measure_case(
    backend: str,
    count: int,
    batch_size: int,
    payload_kind: str,
    trace_kind: str,
    samples: int,
) -> dict[str, Any]:
    operations = _trace(count, batch_size, payload_kind, trace_kind)
    scalar_ns: list[int] = []
    transaction_ns: list[int] = []
    ratios: list[float] = []
    peak_bytes: list[int] = []
    stage_ns: list[int] = []
    backend_load_ns: list[int] = []
    prepublish_total_ns: list[int] = []
    oracle_ranges = _seed_ranges(backend, count, payload_kind)
    oracle_rows = _scalar(oracle_ranges, operations)
    oracle_snapshot = oracle_ranges.snapshot()
    result_digest = _checksum(_freeze_results(oracle_rows))
    final_digest = _checksum(_freeze_snapshot(oracle_snapshot))

    for sample in range(samples):
        first_transaction = sample % 2 == 0
        order = (
            ("transaction", "scalar")
            if first_transaction
            else ("scalar", "transaction")
        )
        observed: dict[str, tuple[tuple[MutationResult, ...], RangeSnapshot]] = {}
        elapsed: dict[str, int] = {}
        transaction_timings: dict[str, int] = {}
        for method in order:
            ranges = _seed_ranges(backend, count, payload_kind)
            gc.collect()
            if tracemalloc.is_tracing():
                raise AssertionError(
                    "tracemalloc must be inactive during latency timing"
                )
            started = time.perf_counter_ns()
            rows = (
                _rangeset_transaction(ranges, operations, _timings=transaction_timings)
                if method == "transaction"
                else _scalar(ranges, operations)
            )
            elapsed[method] = time.perf_counter_ns() - started
            observed[method] = (rows, ranges.snapshot())
        if observed["transaction"] != (oracle_rows, oracle_snapshot) or observed[
            "scalar"
        ] != (oracle_rows, oracle_snapshot):
            raise AssertionError("timed result/final state differs from scalar oracle")

        # Memory is a separate untimed transaction replay.  Tracing never
        # contaminates either latency path, and this extra run is validated.
        memory_ranges = _seed_ranges(backend, count, payload_kind)
        gc.collect()
        tracemalloc.start()
        before_peak = tracemalloc.get_traced_memory()[1]
        memory_rows = _rangeset_transaction(memory_ranges, operations)
        memory_peak = max(0, tracemalloc.get_traced_memory()[1] - before_peak)
        tracemalloc.stop()
        if (memory_rows, memory_ranges.snapshot()) != (oracle_rows, oracle_snapshot):
            raise AssertionError("memory replay result/final state differs from oracle")
        scalar_ns.append(elapsed["scalar"])
        transaction_ns.append(elapsed["transaction"])
        ratios.append(elapsed["transaction"] / max(1, elapsed["scalar"]))
        peak_bytes.append(memory_peak)
        stage_ns.append(transaction_timings["stage_ns"])
        backend_load_ns.append(transaction_timings["backend_load_ns"])
        prepublish_total_ns.append(transaction_timings["prepublish_total_ns"])

    lower, upper = _bootstrap_interval(
        ratios,
        BOOTSTRAP_SEED + count + batch_size + sum(map(ord, payload_kind + trace_kind)),
    )
    return {
        "backend": backend,
        "interval_count": count,
        "batch_size": batch_size,
        "payload": payload_kind,
        "trace": trace_kind,
        "scalar_ns_samples": scalar_ns,
        "transaction_ns_samples": transaction_ns,
        "paired_ratios": ratios,
        "ratio_median": statistics.median(ratios),
        "ratio_95_low": lower,
        "ratio_95_high": upper,
        "transaction_peak_bytes_samples": peak_bytes,
        "stage_ns_samples": stage_ns,
        "backend_load_ns_samples": backend_load_ns,
        "prepublish_total_ns_samples": prepublish_total_ns,
        "result_sha256": result_digest,
        "final_state_sha256": final_digest,
    }


def _checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=_REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    status = run("git", "status", "--porcelain=v1", "--untracked-files=all")
    changed = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    staged = sorted(
        line[3:]
        for line in status.splitlines()
        if len(line) >= 4 and line[0] not in " ?"
    )
    source_material = []
    for path in changed:
        candidate = _REPOSITORY_ROOT / path
        source_material.append(
            [
                path,
                hashlib.sha256(candidate.read_bytes()).hexdigest()
                if candidate.is_file()
                else None,
            ]
        )
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "head_tree": run("git", "rev-parse", "HEAD^{tree}"),
        "clean_worktree": not status,
        "changed_paths": changed,
        "staged_files": staged,
        "source_state_sha256": _checksum(source_material),
    }


def _file_provenance(relative: str) -> dict[str, str]:
    path = (_REPOSITORY_ROOT / relative).resolve()
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _compiler_version() -> str:
    compiler = os.environ.get("CXX", "c++")
    try:
        completed = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.splitlines()[0] if completed.stdout else "unavailable"


def _runtime_provenance() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
    }


def _build_provenance() -> dict[str, Any]:
    return {
        "command": os.environ.get("TREE_MENDOUS_BUILD_COMMAND", "uv sync --all-extras"),
        "cxx": os.environ.get("CXX", "c++"),
        "cxx_version": _compiler_version(),
        "cc": str(sysconfig.get_config_var("CC") or "unknown"),
        "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
        "flags": {name: os.environ.get(name, "") for name in BUILD_FLAG_NAMES},
    }


def _binary_provenance() -> dict[str, Any]:
    ranges = _seed_ranges(BACKEND, 1, "none")
    implementation_type = type(ranges._adapter.implementation)
    path = Path(inspect.getfile(implementation_type)).resolve()
    try:
        display_path = str(path.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "id": BACKEND,
        "module": implementation_type.__module__,
        "type": implementation_type.__qualname__,
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _provenance() -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "sources": [
            _file_provenance("tests/performance/experiments/rangeset_transaction.py"),
            _file_provenance("treemendous/rangeset.py"),
        ],
        "runtime": _runtime_provenance(),
        "build": _build_provenance(),
        "backend": _binary_provenance(),
    }


def _gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target = {
        row["payload"]: row["ratio_95_high"]
        for row in rows
        if row["batch_size"] == 16
        and row["interval_count"] == 1_000
        and row["trace"] == "restorative"
    }
    target_pass = set(target) == set(PAYLOAD_KINDS) and all(
        value <= 1.0 for value in target.values()
    )
    application = [
        row["ratio_95_high"]
        for row in rows
        if row["trace"] == "non_restorative" and row["batch_size"] > 0
    ]
    application_pass = bool(application) and min(application) <= 0.90
    supported = [row for row in rows if row["batch_size"] > 0]
    regression_pass = bool(supported) and all(
        row["ratio_95_high"] <= 1.10 for row in supported
    )
    accepted = target_pass and application_pass and regression_pass
    return {
        "b16_n1000_upper95_by_payload": target,
        "b16_n1000_limit": 1.0,
        "b16_n1000_pass": target_pass,
        "non_restorative_best_upper95": min(application) if application else None,
        "non_restorative_limit": 0.90,
        "non_restorative_pass": application_pass,
        "all_supported_limit": 1.10,
        "all_supported_worst_upper95": max(
            (row["ratio_95_high"] for row in supported), default=None
        ),
        "all_supported_pass": regression_pass,
        "decision": "ACCEPTED" if accepted else "REJECTED",
    }


def run_matrix(
    *,
    backend: str = "py_boundary",
    samples: int = MINIMUM_SAMPLES,
    interval_counts: Sequence[int] = INTERVAL_COUNTS,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    payload_kinds: Sequence[str] = PAYLOAD_KINDS,
    trace_kinds: Sequence[str] = TRACE_KINDS,
) -> dict[str, Any]:
    if backend != BACKEND:
        raise ValueError(f"transaction experiment backend must be {BACKEND!r}")
    if type(samples) is not int or samples < MINIMUM_SAMPLES:
        raise ValueError(f"samples must be an integer >= {MINIMUM_SAMPLES}")
    dimensions = (
        tuple(interval_counts),
        tuple(batch_sizes),
        tuple(payload_kinds),
        tuple(trace_kinds),
    )
    full_dimensions = (INTERVAL_COUNTS, BATCH_SIZES, PAYLOAD_KINDS, TRACE_KINDS)
    bounded_dimensions = ((64,), (0, 1, 4), PAYLOAD_KINDS, TRACE_KINDS)
    if dimensions == full_dimensions:
        profile = "full"
    elif dimensions == bounded_dimensions:
        profile = "bounded"
    else:
        raise ValueError(
            "transaction matrix must use the fixed full or bounded profile"
        )
    rows = [
        _measure_case(backend, n, b, payload, trace, samples)
        for n in interval_counts
        for b in batch_sizes
        for payload in payload_kinds
        for trace in trace_kinds
    ]
    return {
        "schema": SCHEMA,
        "samples": samples,
        "matrix": {
            "profile": profile,
            "batch_sizes": list(batch_sizes),
            "interval_counts": list(interval_counts),
            "payloads": list(payload_kinds),
            "traces": list(trace_kinds),
            "paired_order": "AB/BA alternating",
            "timed_boundary": "mutation calls only; setup and exact validation outside",
        },
        "rows": rows,
        "gate": _gate(rows),
        "provenance": _provenance(),
    }


def _markdown(report: dict[str, Any], json_digest: str) -> str:
    gate = report["gate"]
    lines = [
        "# RangeSet transaction experiment",
        "",
        f"Decision: **{gate['decision']}**",
        f"JSON SHA-256: `{json_digest}`",
        "",
        "| N | B | payload | trace | median ratio | upper 95% |",
        "| ---: | ---: | --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['interval_count']} | {row['batch_size']} | {row['payload']} | "
        f"{row['trace']} | {row['ratio_median']:.4f} | {row['ratio_95_high']:.4f} |"
        for row in report["rows"]
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite number: {value}")
    return parsed


def _validate_finite(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys/type mismatch")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_case_digests(
    backend: str, count: int, batch_size: int, payload: str, trace: str
) -> tuple[str, str]:
    ranges = _seed_ranges(backend, count, payload)
    rows = _scalar(ranges, _trace(count, batch_size, payload, trace))
    return (
        _checksum(_freeze_results(rows)),
        _checksum(_freeze_snapshot(ranges.snapshot())),
    )


def verify_artifacts(output: Path) -> dict[str, Any]:
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    expected_sidecar = f"{digest}  {output.name}\n"
    if Path(f"{output}.sha256").read_text() != expected_sidecar:
        raise ValueError("checksum mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
        parse_float=_finite_float,
    )
    _validate_finite(report)
    _require_keys(
        report,
        {"schema", "samples", "matrix", "rows", "gate", "provenance"},
        "report",
    )
    if report["schema"] != SCHEMA:
        raise ValueError("schema mismatch")
    samples = report["samples"]
    if type(samples) is not int or samples < MINIMUM_SAMPLES:
        raise ValueError("invalid sample count/type")
    matrix = _require_keys(
        report["matrix"],
        {
            "profile",
            "batch_sizes",
            "interval_counts",
            "payloads",
            "traces",
            "paired_order",
            "timed_boundary",
        },
        "matrix/methodology",
    )
    expected_dimensions = {
        "full": {
            "batch_sizes": list(BATCH_SIZES),
            "interval_counts": list(INTERVAL_COUNTS),
            "payloads": list(PAYLOAD_KINDS),
            "traces": list(TRACE_KINDS),
        },
        "bounded": {
            "batch_sizes": [0, 1, 4],
            "interval_counts": [64],
            "payloads": list(PAYLOAD_KINDS),
            "traces": list(TRACE_KINDS),
        },
    }
    if type(matrix["profile"]) is not str:
        raise ValueError("matrix profile exact type mismatch")
    dimensions = expected_dimensions.get(matrix["profile"])
    if dimensions is None or any(
        not _exact_json_equal(matrix[key], value) for key, value in dimensions.items()
    ):
        raise ValueError("fixed matrix membership mismatch")
    if (
        matrix["paired_order"] != "AB/BA alternating"
        or matrix["timed_boundary"]
        != "mutation calls only; setup and exact validation outside"
    ):
        raise ValueError("fixed matrix methodology mismatch")
    expected_identities = [
        (n, b, payload, trace)
        for n in matrix["interval_counts"]
        for b in matrix["batch_sizes"]
        for payload in matrix["payloads"]
        for trace in matrix["traces"]
    ]
    rows = report["rows"]
    if type(rows) is not list or len(rows) != len(expected_identities):
        raise ValueError("row matrix shape mismatch")
    row_keys = {
        "backend",
        "interval_count",
        "batch_size",
        "payload",
        "trace",
        "scalar_ns_samples",
        "transaction_ns_samples",
        "paired_ratios",
        "ratio_median",
        "ratio_95_low",
        "ratio_95_high",
        "transaction_peak_bytes_samples",
        "stage_ns_samples",
        "backend_load_ns_samples",
        "prepublish_total_ns_samples",
        "result_sha256",
        "final_state_sha256",
    }
    observed_identities = []
    for row in rows:
        _require_keys(row, row_keys, "row")
        if (
            type(row["interval_count"]) is not int
            or type(row["batch_size"]) is not int
            or type(row["backend"]) is not str
            or type(row["payload"]) is not str
            or type(row["trace"]) is not str
        ):
            raise ValueError("row identity exact type mismatch")
        identity = (
            row["interval_count"],
            row["batch_size"],
            row["payload"],
            row["trace"],
        )
        observed_identities.append(identity)
        if row["backend"] != BACKEND:
            raise ValueError("row backend identity mismatch")
        integer_sample_keys = (
            "scalar_ns_samples",
            "transaction_ns_samples",
            "transaction_peak_bytes_samples",
            "stage_ns_samples",
            "backend_load_ns_samples",
            "prepublish_total_ns_samples",
        )
        for key in (*integer_sample_keys, "paired_ratios"):
            values = row[key]
            if type(values) is not list or len(values) != samples:
                raise ValueError("raw sample shape mismatch")
        if any(
            type(value) is not int or value < 0
            for key in integer_sample_keys
            for value in row[key]
        ) or any(
            value <= 0
            for key in ("scalar_ns_samples", "transaction_ns_samples")
            for value in row[key]
        ):
            raise ValueError("elapsed/resource sample exact type mismatch")
        for load, stage, total, elapsed in zip(
            row["backend_load_ns_samples"],
            row["stage_ns_samples"],
            row["prepublish_total_ns_samples"],
            row["transaction_ns_samples"],
            strict=True,
        ):
            if not (load <= stage <= total <= elapsed):
                raise ValueError("transaction timing components are inconsistent")
        ratios = [
            transaction / max(1, scalar)
            for scalar, transaction in zip(
                row["scalar_ns_samples"], row["transaction_ns_samples"], strict=True
            )
        ]
        if not _exact_json_equal(ratios, row["paired_ratios"]):
            raise ValueError("raw paired ratios mismatch")
        low, high = _bootstrap_interval(
            ratios,
            BOOTSTRAP_SEED
            + row["interval_count"]
            + row["batch_size"]
            + sum(map(ord, row["payload"] + row["trace"])),
        )
        derived = (statistics.median(ratios), low, high)
        recorded = (row["ratio_median"], row["ratio_95_low"], row["ratio_95_high"])
        if not _exact_json_equal(list(derived), list(recorded)):
            raise ValueError("derived interval mismatch")
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in row["paired_ratios"]
        ):
            raise ValueError("ratio exact type/non-finite mismatch")
        expected_result, expected_final = _expected_case_digests(
            row["backend"], *identity
        )
        if row["result_sha256"] != expected_result:
            raise ValueError("result digest mismatch")
        if row["final_state_sha256"] != expected_final:
            raise ValueError("final-state digest mismatch")
    if not _exact_json_equal(
        [list(identity) for identity in observed_identities],
        [list(identity) for identity in expected_identities],
    ):
        raise ValueError("fixed matrix membership/order mismatch")
    _require_keys(
        report["gate"],
        {
            "b16_n1000_upper95_by_payload",
            "b16_n1000_limit",
            "b16_n1000_pass",
            "non_restorative_best_upper95",
            "non_restorative_limit",
            "non_restorative_pass",
            "all_supported_limit",
            "all_supported_worst_upper95",
            "all_supported_pass",
            "decision",
        },
        "gate",
    )
    expected_gate = _gate(rows)
    if not _exact_json_equal(report["gate"], expected_gate):
        raise ValueError("gate mismatch")
    provenance = _require_keys(
        report["provenance"],
        {"git", "sources", "runtime", "build", "backend"},
        "provenance",
    )
    _require_keys(
        provenance["git"],
        {
            "commit",
            "head_tree",
            "clean_worktree",
            "changed_paths",
            "staged_files",
            "source_state_sha256",
        },
        "git provenance",
    )
    if not _exact_json_equal(provenance["git"], _git_metadata()):
        raise ValueError("git/source-state provenance mismatch")
    expected_sources = [
        _file_provenance("tests/performance/experiments/rangeset_transaction.py"),
        _file_provenance("treemendous/rangeset.py"),
    ]
    if not _exact_json_equal(provenance["sources"], expected_sources):
        raise ValueError("source provenance path/hash mismatch")
    if not _exact_json_equal(provenance["runtime"], _runtime_provenance()):
        raise ValueError("runtime provenance does not match current runtime")
    if not _exact_json_equal(provenance["build"], _build_provenance()):
        raise ValueError("build provenance does not match current build")
    if not _exact_json_equal(provenance["backend"], _binary_provenance()):
        raise ValueError("active backend identity/module/type/path/hash mismatch")
    if output.with_suffix(".md").read_text() != _markdown(report, digest):
        raise ValueError("Markdown mismatch")
    canonical = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if canonical != encoded:
        raise ValueError("JSON is not canonical")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="py_boundary")
    parser.add_argument("--samples", type=int, default=MINIMUM_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--bounded", action="store_true", help="N=64, B=0/1/4 correctness slice"
    )
    args = parser.parse_args(argv)
    if args.verify:
        report = verify_artifacts(args.output)
    else:
        report = run_matrix(
            backend=args.backend,
            samples=args.samples,
            interval_counts=(64,) if args.bounded else INTERVAL_COUNTS,
            batch_sizes=(0, 1, 4) if args.bounded else BATCH_SIZES,
        )
        write_artifacts(report, args.output)
        verify_artifacts(args.output)
    print(
        json.dumps(
            {"artifact": str(args.output), "decision": report["gate"]["decision"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
