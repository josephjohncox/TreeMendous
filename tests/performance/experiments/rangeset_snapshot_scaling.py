#!/usr/bin/env python3
"""Paired scaling evidence for the geometry-only ``RangeSet`` snapshot cache."""

from __future__ import annotations

import argparse
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

from treemendous import IntervalResult, RangeSnapshot, Span, create_range_set
from treemendous.rangeset import RangeSet

SCHEMA = "treemendous-rangeset-snapshot-scaling-experiment-v2"
BACKEND = "py_boundary"
BUILD_FLAG_NAMES = (
    "BOOST_ROOT",
    "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
    "TREE_MENDOUS_GLIBCXX_DEBUG",
    "TREE_MENDOUS_LOCAL_NATIVE",
    "TREE_MENDOUS_SANITIZERS",
    "TREE_MENDOUS_WITH_ICL",
)
MINIMUM_BLOCKS = 30
CONFIRMATION_BLOCKS = 40
INTERVAL_COUNTS = (100, 1_000, 10_000)
UNCHANGED_READS = 16
PILOT_TARGET_NS = 5_000_000
PILOT_MIN_ITERATIONS = 2
PILOT_MAX_ITERATIONS = 64
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 91_337
UNCHANGED_N10000_LIMIT = 0.25
CACHED_SCALING_LIMIT = 1.50
WRITE_OBSERVE_LIMIT = 1.10
BALANCED_BLOCK_METHOD = (
    "each block contains one cached-first and one uncached-first ordering; "
    "ordering-pair sequence alternates AB/BA then BA/AB by block; cached and "
    "uncached elapsed times are totaled across their two block positions before "
    "one block ratio is computed; bootstrap resampling units are whole blocks"
)
PILOT_METHOD = (
    "outside retained blocks, start at two timed iterations and double one "
    "common cached/uncached iteration count until both timed positions reach at "
    "least 5 ms or the deterministic cap of "
    "64; record every pilot duration and exclude all pilot data from samples "
    "and gates"
)
TIMED_BOUNDARY = (
    "public cached snapshot versus faithful uncached RangeSnapshot construction; "
    "pilot, setup, and same-instance validation outside retained-block timing"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
T = TypeVar("T")


def _seed_ranges(backend: str, count: int) -> RangeSet:
    ranges = create_range_set(
        (0, count * 3),
        backend=backend,
        initially_available=False,
    )
    for index in range(count):
        ranges.add(Span(index * 3, index * 3 + 1))
    return ranges


def _uncached_snapshot(ranges: RangeSet) -> RangeSnapshot:
    """Faithfully construct the public value while deliberately bypassing reuse."""
    with ranges._lock:
        return RangeSnapshot(
            ranges.intervals(),
            ranges._total_free,
            ranges._domain,
        )


def _snapshot_digest(snapshot: RangeSnapshot) -> str:
    material = {
        "domain": (
            None
            if snapshot.domain is None
            else [[span.start, span.end] for span in snapshot.domain.spans]
        ),
        "intervals": [
            [interval.start, interval.end, interval.length, interval.data]
            for interval in snapshot.intervals
        ],
        "total_free": snapshot.total_free,
    }
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_snapshot_digest(count: int) -> str:
    material = {
        "domain": [[0, count * 3]],
        "intervals": [[index * 3, index * 3 + 1, 1, None] for index in range(count)],
        "total_free": count,
    }
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_exact_snapshot(
    snapshot: RangeSnapshot, *, count: int, removed: bool = False
) -> None:
    expected_count = count - int(removed)
    assert type(snapshot) is RangeSnapshot
    assert len(snapshot.intervals) == expected_count
    assert snapshot.total_free == expected_count
    assert snapshot.domain is not None
    assert snapshot.domain.spans == (Span(0, count * 3),)
    assert all(type(item) is IntervalResult for item in snapshot.intervals)
    expected_last = count - 2 if removed else count - 1
    assert snapshot.intervals[-1] == IntervalResult(
        expected_last * 3, expected_last * 3 + 1
    )


def _elapsed(call: Callable[[], T]) -> tuple[int, T]:
    started = time.perf_counter_ns()
    result = call()
    return time.perf_counter_ns() - started, result


def _cached_burst(ranges: RangeSet, reads: int) -> RangeSnapshot:
    snapshot = ranges.snapshot()
    for _ in range(reads - 1):
        snapshot = ranges.snapshot()
    return snapshot


def _uncached_burst(ranges: RangeSet, reads: int) -> RangeSnapshot:
    snapshot = _uncached_snapshot(ranges)
    for _ in range(reads - 1):
        snapshot = _uncached_snapshot(ranges)
    return snapshot


def _restorative_cycle(
    ranges: RangeSet, snapshotter: Callable[[RangeSet], RangeSnapshot], count: int
) -> tuple[Any, RangeSnapshot, Any, RangeSnapshot]:
    target = Span((count - 1) * 3, (count - 1) * 3 + 1)
    removed = ranges.discard(target, require_covered=True)
    after_remove = snapshotter(ranges)
    restored = ranges.add(target)
    after_restore = snapshotter(ranges)
    return removed, after_remove, restored, after_restore


def _repeat(call: Callable[[], T], iterations: int) -> T:
    result = call()
    for _ in range(iterations - 1):
        result = call()
    return result


def _bootstrap_interval(values: Sequence[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians))]


def _summary(values: Sequence[int | float], seed: int) -> dict[str, float]:
    floats = [float(value) for value in values]
    low, high = _bootstrap_interval(floats, seed)
    return {
        "median": float(statistics.median(values)),
        "median_95_low": low,
        "median_95_high": high,
    }


def _ratio_summary(
    cached: Sequence[int], uncached: Sequence[int], seed: int
) -> tuple[list[float], dict[str, float]]:
    ratios = [
        candidate / baseline
        for candidate, baseline in zip(cached, uncached, strict=True)
    ]
    low, high = _bootstrap_interval(ratios, seed)
    return ratios, {
        "median": float(statistics.median(ratios)),
        "median_95_low": low,
        "median_95_high": high,
    }


def _validate_cycle(
    result: tuple[Any, RangeSnapshot, Any, RangeSnapshot], count: int
) -> None:
    removed, after_remove, restored, after_restore = result
    target = Span((count - 1) * 3, (count - 1) * 3 + 1)
    assert removed.changed == (target,)
    assert removed.changed_length == 1 and removed.fully_covered
    assert restored.changed == (target,)
    assert restored.changed_length == 1 and not restored.fully_covered
    _assert_exact_snapshot(after_remove, count=count, removed=True)
    _assert_exact_snapshot(after_restore, count=count)


def _pilot(
    cached_call: Callable[[], T],
    uncached_call: Callable[[], T],
    validate: Callable[[T, T], None],
) -> tuple[int, dict[str, Any]]:
    iterations = PILOT_MIN_ITERATIONS
    measurements: list[dict[str, int]] = []
    while True:
        cached_ns, cached_result = _elapsed(lambda: _repeat(cached_call, iterations))
        uncached_ns, uncached_result = _elapsed(
            lambda: _repeat(uncached_call, iterations)
        )
        validate(cached_result, uncached_result)
        measurements.append(
            {
                "iterations": iterations,
                "cached_ns": cached_ns,
                "uncached_ns": uncached_ns,
            }
        )
        if (
            min(cached_ns, uncached_ns) >= PILOT_TARGET_NS
            or iterations == PILOT_MAX_ITERATIONS
        ):
            break
        iterations = min(iterations * 2, PILOT_MAX_ITERATIONS)
    return iterations, {
        "target_position_ns": PILOT_TARGET_NS,
        "maximum_iterations": PILOT_MAX_ITERATIONS,
        "order": "cached_then_uncached",
        "measurements": measurements,
        "selected_iterations": iterations,
        "target_reached_by_both": min(cached_ns, uncached_ns) >= PILOT_TARGET_NS,
        "excluded_from_blocks_and_gates": True,
    }


def _balanced_blocks(
    cached_call: Callable[[], T],
    uncached_call: Callable[[], T],
    validate: Callable[[T, T], None],
    *,
    block_count: int,
    iterations: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    blocks: list[dict[str, Any]] = []
    cached_totals: list[int] = []
    uncached_totals: list[int] = []
    for block_index in range(block_count):
        orderings = (
            ("cached_first", "uncached_first")
            if block_index % 2 == 0
            else ("uncached_first", "cached_first")
        )
        positions: list[dict[str, Any]] = []
        for ordering in orderings:
            if ordering == "cached_first":
                cached_ns, cached_result = _elapsed(
                    lambda: _repeat(cached_call, iterations)
                )
                uncached_ns, uncached_result = _elapsed(
                    lambda: _repeat(uncached_call, iterations)
                )
            else:
                uncached_ns, uncached_result = _elapsed(
                    lambda: _repeat(uncached_call, iterations)
                )
                cached_ns, cached_result = _elapsed(
                    lambda: _repeat(cached_call, iterations)
                )
            validate(cached_result, uncached_result)
            positions.append(
                {
                    "ordering": ordering,
                    "cached_ns": cached_ns,
                    "uncached_ns": uncached_ns,
                }
            )
        cached_total = sum(position["cached_ns"] for position in positions)
        uncached_total = sum(position["uncached_ns"] for position in positions)
        ratio = cached_total / uncached_total
        blocks.append(
            {
                "block_index": block_index,
                "positions": positions,
                "cached_ns_total": cached_total,
                "uncached_ns_total": uncached_total,
                "ratio": ratio,
            }
        )
        cached_totals.append(cached_total)
        uncached_totals.append(uncached_total)
    ratios, ratio_summary = _ratio_summary(cached_totals, uncached_totals, seed)
    divisor = 2 * iterations
    cached_per_iteration = [value / divisor for value in cached_totals]
    uncached_per_iteration = [value / divisor for value in uncached_totals]
    return (
        blocks,
        ratios,
        _summary(cached_per_iteration, seed + 1),
        _summary(uncached_per_iteration, seed + 2),
        ratio_summary,
    )


def _measure_unchanged(
    backend: str, count: int, block_count: int, reads: int
) -> dict[str, Any]:
    ranges = _seed_ranges(backend, count)
    cached_reference = ranges.snapshot()
    faithful = _uncached_snapshot(ranges)
    assert cached_reference == faithful
    assert cached_reference is ranges.snapshot()
    _assert_exact_snapshot(cached_reference, count=count)

    def cached_call() -> RangeSnapshot:
        return _cached_burst(ranges, reads)

    def uncached_call() -> RangeSnapshot:
        return _uncached_burst(ranges, reads)

    def validate(cached_result: RangeSnapshot, uncached_result: RangeSnapshot) -> None:
        assert cached_result is cached_reference
        assert uncached_result == cached_reference
        assert uncached_result is not cached_reference
        _assert_exact_snapshot(cached_result, count=count)

    iterations, pilot = _pilot(cached_call, uncached_call, validate)
    seed = BOOTSTRAP_SEED + count
    blocks, ratios, cached_ns, uncached_ns, ratio = _balanced_blocks(
        cached_call,
        uncached_call,
        validate,
        block_count=block_count,
        iterations=iterations,
        seed=seed,
    )
    return {
        "kind": "unchanged_read_burst",
        "interval_count": count,
        "operations_per_iteration": reads,
        "iterations_per_position": iterations,
        "pilot": pilot,
        "blocks": blocks,
        "block_ratios": ratios,
        "cached_ns_per_iteration": cached_ns,
        "uncached_ns_per_iteration": uncached_ns,
        "ratio": ratio,
        "validated_blocks": block_count,
        "final_snapshot_sha256": _snapshot_digest(cached_reference),
    }


def _measure_write_observe(
    backend: str, count: int, block_count: int
) -> dict[str, Any]:
    cached_ranges = _seed_ranges(backend, count)
    uncached_ranges = _seed_ranges(backend, count)
    cached_initial = cached_ranges.snapshot()
    uncached_initial = _uncached_snapshot(uncached_ranges)
    assert cached_initial == uncached_initial

    def cached_call() -> tuple[Any, RangeSnapshot, Any, RangeSnapshot]:
        return _restorative_cycle(cached_ranges, RangeSet.snapshot, count)

    def uncached_call() -> tuple[Any, RangeSnapshot, Any, RangeSnapshot]:
        return _restorative_cycle(uncached_ranges, _uncached_snapshot, count)

    def validate(
        cached_result: tuple[Any, RangeSnapshot, Any, RangeSnapshot],
        uncached_result: tuple[Any, RangeSnapshot, Any, RangeSnapshot],
    ) -> None:
        _validate_cycle(cached_result, count)
        _validate_cycle(uncached_result, count)
        assert cached_result == uncached_result
        assert cached_result[3] is cached_ranges.snapshot()
        assert cached_result[3] is not cached_initial
        assert uncached_result[3] is not uncached_ranges.snapshot()
        assert cached_ranges.intervals() == uncached_ranges.intervals()

    iterations, pilot = _pilot(cached_call, uncached_call, validate)
    seed = BOOTSTRAP_SEED + count * 2
    blocks, ratios, cached_ns, uncached_ns, ratio = _balanced_blocks(
        cached_call,
        uncached_call,
        validate,
        block_count=block_count,
        iterations=iterations,
        seed=seed,
    )
    return {
        "kind": "write_then_observe_restorative",
        "interval_count": count,
        "operations_per_iteration": 4,
        "iterations_per_position": iterations,
        "pilot": pilot,
        "blocks": blocks,
        "block_ratios": ratios,
        "cached_ns_per_iteration": cached_ns,
        "uncached_ns_per_iteration": uncached_ns,
        "ratio": ratio,
        "validated_blocks": block_count,
        "final_snapshot_sha256": _snapshot_digest(cached_ranges.snapshot()),
    }


def _gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unchanged = {
        row["interval_count"]: row
        for row in rows
        if row["kind"] == "unchanged_read_burst"
    }
    write_rows = [
        row for row in rows if row["kind"] == "write_then_observe_restorative"
    ]
    target = unchanged.get(10_000)
    unchanged_upper = None if target is None else target["ratio"]["median_95_high"]
    unchanged_pass = (
        type(unchanged_upper) is float and unchanged_upper <= UNCHANGED_N10000_LIMIT
    )
    n1000 = unchanged.get(1_000)
    if target is None or n1000 is None:
        scaling_ratio = None
    else:
        scaling_ratio = (
            target["cached_ns_per_iteration"]["median"]
            / n1000["cached_ns_per_iteration"]["median"]
        )
    scaling_pass = (
        type(scaling_ratio) is float and scaling_ratio <= CACHED_SCALING_LIMIT
    )
    write_upper = max(
        (row["ratio"]["median_95_high"] for row in write_rows), default=None
    )
    write_pass = (
        len(write_rows) == len(INTERVAL_COUNTS)
        and type(write_upper) is float
        and write_upper <= WRITE_OBSERVE_LIMIT
    )
    accepted = unchanged_pass and scaling_pass and write_pass
    return {
        "unchanged_n10000_upper95": unchanged_upper,
        "unchanged_n10000_limit": UNCHANGED_N10000_LIMIT,
        "unchanged_n10000_pass": unchanged_pass,
        "cached_n10000_over_n1000_median": scaling_ratio,
        "cached_scaling_limit": CACHED_SCALING_LIMIT,
        "cached_scaling_pass": scaling_pass,
        "write_observe_worst_upper95": write_upper,
        "write_observe_limit": WRITE_OBSERVE_LIMIT,
        "write_observe_pass": write_pass,
        "decision": "ACCEPTED" if accepted else "REJECTED",
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

    status = git("status", "--porcelain=v1")
    changed_paths = sorted(
        line
        for line in git(
            "ls-files", "--modified", "--others", "--exclude-standard"
        ).splitlines()
        if line
    )
    staged_files = sorted(
        line for line in git("diff", "--cached", "--name-only").splitlines() if line
    )
    source_state = hashlib.sha256(status.encode())
    source_state.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    )
    for relative in changed_paths:
        candidate = _REPOSITORY_ROOT / relative
        source_state.update(relative.encode())
        source_state.update(b"\0")
        if candidate.is_file():
            source_state.update(hashlib.sha256(candidate.read_bytes()).digest())
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "head_tree": git("rev-parse", "HEAD^{tree}").strip(),
        "clean_worktree": not status,
        "changed_paths": changed_paths,
        "staged_files": staged_files,
        "source_state_sha256": source_state.hexdigest(),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _file_provenance(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": _display_path(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
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


def _backend_provenance() -> dict[str, str]:
    ranges = _seed_ranges(BACKEND, 1)
    implementation_type = type(ranges._adapter.implementation)
    implementation_path = Path(inspect.getfile(implementation_type))
    return {
        "id": BACKEND,
        "module": implementation_type.__module__,
        "type": implementation_type.__qualname__,
        **_file_provenance(implementation_path),
    }


def _provenance() -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "sources": [
            _file_provenance(Path(__file__)),
            _file_provenance(_REPOSITORY_ROOT / "treemendous" / "rangeset.py"),
        ],
        "runtime": _runtime_provenance(),
        "build": _build_provenance(),
        "backend": _backend_provenance(),
    }


def _methodology() -> dict[str, Any]:
    return {
        "interval_counts": list(INTERVAL_COUNTS),
        "unchanged_reads_per_iteration": UNCHANGED_READS,
        "confirmation_blocks": CONFIRMATION_BLOCKS,
        "minimum_blocks": MINIMUM_BLOCKS,
        "balanced_block_method": BALANCED_BLOCK_METHOD,
        "pilot_method": PILOT_METHOD,
        "pilot_target_ns": PILOT_TARGET_NS,
        "pilot_minimum_iterations": PILOT_MIN_ITERATIONS,
        "pilot_maximum_iterations": PILOT_MAX_ITERATIONS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "timed_boundary": TIMED_BOUNDARY,
    }


def run_matrix(
    *,
    backend: str = BACKEND,
    blocks: int = CONFIRMATION_BLOCKS,
) -> dict[str, Any]:
    if backend != BACKEND:
        raise ValueError(f"snapshot experiment backend must be {BACKEND!r}")
    if type(blocks) is not int or blocks != CONFIRMATION_BLOCKS:
        raise ValueError(
            f"blocks must equal fixed confirmation count {CONFIRMATION_BLOCKS}"
        )
    rows: list[dict[str, Any]] = []
    for count in INTERVAL_COUNTS:
        rows.append(_measure_unchanged(backend, count, blocks, UNCHANGED_READS))
        rows.append(_measure_write_observe(backend, count, blocks))
    return {
        "schema": SCHEMA,
        "blocks": blocks,
        "methodology": _methodology(),
        "rows": rows,
        "gate": _gate(rows),
        "provenance": _provenance(),
    }


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# RangeSet geometry snapshot cache experiment",
        "",
        f"Decision: **{report['gate']['decision']}**",
        f"JSON SHA-256: `{digest}`",
        "",
        "| workload | N | iterations | cached median ns/iteration | uncached median ns/iteration | ratio | upper 95% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['kind']} | {row['interval_count']} | "
        f"{row['iterations_per_position']} | "
        f"{row['cached_ns_per_iteration']['median']:.0f} | "
        f"{row['uncached_ns_per_iteration']['median']:.0f} | "
        f"{row['ratio']['median']:.4f} | "
        f"{row['ratio']['median_95_high']:.4f} |"
        for row in report["rows"]
    )
    gate = report["gate"]
    lines.extend(
        (
            "",
            f"- N=10000 unchanged upper-95: {gate['unchanged_n10000_upper95']}",
            "- cached N=10000/N=1000 median: "
            f"{gate['cached_n10000_over_n1000_median']}",
            "- write-then-observe worst upper-95: "
            f"{gate['write_observe_worst_upper95']}",
        )
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


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
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


def _is_git_oid(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_summary(summary: Any, samples: list[int | float], seed: int) -> None:
    _require_keys(summary, {"median", "median_95_low", "median_95_high"}, "summary")
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in summary.values()
    ):
        raise ValueError("summary exact type/non-finite mismatch")
    if not _exact_equal(summary, _summary(samples, seed)):
        raise ValueError("derived summary mismatch")


def verify_artifacts(output: Path) -> dict[str, Any]:
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if Path(f"{output}.sha256").read_text() != f"{digest}  {output.name}\n":
        raise ValueError("checksum mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
        parse_float=_finite_float,
    )
    _require_keys(
        report,
        {"schema", "blocks", "methodology", "rows", "gate", "provenance"},
        "report",
    )
    if report["schema"] != SCHEMA:
        raise ValueError("schema mismatch")
    blocks = report["blocks"]
    if type(blocks) is not int or blocks != CONFIRMATION_BLOCKS:
        raise ValueError("fixed block count exact type/value mismatch")
    methodology = _require_keys(
        report["methodology"], set(_methodology()), "methodology"
    )
    if not _exact_equal(methodology, _methodology()):
        raise ValueError("fixed matrix/methodology mismatch")
    expected_pairs = [
        (kind, count)
        for count in INTERVAL_COUNTS
        for kind in ("unchanged_read_burst", "write_then_observe_restorative")
    ]
    if type(report["rows"]) is not list or len(report["rows"]) != len(expected_pairs):
        raise ValueError("row shape mismatch")
    observed_pairs: list[tuple[str, int]] = []
    row_keys = {
        "kind",
        "interval_count",
        "operations_per_iteration",
        "iterations_per_position",
        "pilot",
        "blocks",
        "block_ratios",
        "cached_ns_per_iteration",
        "uncached_ns_per_iteration",
        "ratio",
        "validated_blocks",
        "final_snapshot_sha256",
    }
    pilot_keys = {
        "target_position_ns",
        "maximum_iterations",
        "order",
        "measurements",
        "selected_iterations",
        "target_reached_by_both",
        "excluded_from_blocks_and_gates",
    }
    for row in report["rows"]:
        _require_keys(row, row_keys, "row")
        if type(row["kind"]) is not str or type(row["interval_count"]) is not int:
            raise ValueError("row identity exact type mismatch")
        observed_pairs.append((row["kind"], row["interval_count"]))
        expected_operations = (
            methodology["unchanged_reads_per_iteration"]
            if row["kind"] == "unchanged_read_burst"
            else 4
        )
        if row["operations_per_iteration"] != expected_operations:
            raise ValueError("row operation count mismatch")
        iterations = row["iterations_per_position"]
        if (
            type(iterations) is not int
            or iterations < PILOT_MIN_ITERATIONS
            or iterations > PILOT_MAX_ITERATIONS
        ):
            raise ValueError("position iteration count mismatch")
        if row["validated_blocks"] != blocks:
            raise ValueError("validated block count mismatch")

        pilot = _require_keys(row["pilot"], pilot_keys, "pilot")
        if (
            pilot["target_position_ns"] != PILOT_TARGET_NS
            or pilot["maximum_iterations"] != PILOT_MAX_ITERATIONS
            or pilot["order"] != "cached_then_uncached"
            or pilot["selected_iterations"] != iterations
            or pilot["excluded_from_blocks_and_gates"] is not True
            or type(pilot["target_reached_by_both"]) is not bool
        ):
            raise ValueError("pilot fixed methodology/type mismatch")
        measurements = pilot["measurements"]
        if type(measurements) is not list or not measurements:
            raise ValueError("pilot measurement shape mismatch")
        expected_iteration = PILOT_MIN_ITERATIONS
        for index, measurement in enumerate(measurements):
            _require_keys(
                measurement,
                {"iterations", "cached_ns", "uncached_ns"},
                "pilot measurement",
            )
            if (
                measurement["iterations"] != expected_iteration
                or type(measurement["cached_ns"]) is not int
                or measurement["cached_ns"] <= 0
                or type(measurement["uncached_ns"]) is not int
                or measurement["uncached_ns"] <= 0
            ):
                raise ValueError("pilot duration/iteration mismatch")
            reached = (
                min(measurement["cached_ns"], measurement["uncached_ns"])
                >= PILOT_TARGET_NS
            )
            if index < len(measurements) - 1 and (
                reached or expected_iteration == PILOT_MAX_ITERATIONS
            ):
                raise ValueError("pilot stopping derivation mismatch")
            expected_iteration = min(expected_iteration * 2, PILOT_MAX_ITERATIONS)
        final_pilot = measurements[-1]
        reached = (
            min(final_pilot["cached_ns"], final_pilot["uncached_ns"]) >= PILOT_TARGET_NS
        )
        if (
            final_pilot["iterations"] != iterations
            or pilot["target_reached_by_both"] is not reached
            or (not reached and iterations != PILOT_MAX_ITERATIONS)
        ):
            raise ValueError("pilot selection derivation mismatch")

        raw_blocks = row["blocks"]
        if type(raw_blocks) is not list or len(raw_blocks) != blocks:
            raise ValueError("raw block shape mismatch")
        cached_totals: list[int] = []
        uncached_totals: list[int] = []
        derived_ratios: list[float] = []
        for block_index, block in enumerate(raw_blocks):
            _require_keys(
                block,
                {
                    "block_index",
                    "positions",
                    "cached_ns_total",
                    "uncached_ns_total",
                    "ratio",
                },
                "block",
            )
            expected_orderings = (
                ["cached_first", "uncached_first"]
                if block_index % 2 == 0
                else ["uncached_first", "cached_first"]
            )
            if (
                block["block_index"] != block_index
                or type(block["positions"]) is not list
            ):
                raise ValueError("block index/position shape mismatch")
            if len(block["positions"]) != 2:
                raise ValueError("balanced block position count mismatch")
            for position, ordering in zip(
                block["positions"], expected_orderings, strict=True
            ):
                _require_keys(
                    position, {"ordering", "cached_ns", "uncached_ns"}, "position"
                )
                if (
                    position["ordering"] != ordering
                    or type(position["cached_ns"]) is not int
                    or position["cached_ns"] <= 0
                    or type(position["uncached_ns"]) is not int
                    or position["uncached_ns"] <= 0
                ):
                    raise ValueError("balanced block ordering/duration mismatch")
            cached_total = sum(position["cached_ns"] for position in block["positions"])
            uncached_total = sum(
                position["uncached_ns"] for position in block["positions"]
            )
            ratio = cached_total / uncached_total
            if (
                block["cached_ns_total"] != cached_total
                or block["uncached_ns_total"] != uncached_total
                or type(block["ratio"]) is not float
                or not math.isfinite(block["ratio"])
                or block["ratio"] != ratio
            ):
                raise ValueError("raw block total/ratio derivation mismatch")
            cached_totals.append(cached_total)
            uncached_totals.append(uncached_total)
            derived_ratios.append(ratio)

        seed = BOOTSTRAP_SEED + row["interval_count"] * (
            1 if row["kind"] == "unchanged_read_burst" else 2
        )
        divisor = 2 * iterations
        cached_per_iteration = [value / divisor for value in cached_totals]
        uncached_per_iteration = [value / divisor for value in uncached_totals]
        _verify_summary(row["cached_ns_per_iteration"], cached_per_iteration, seed + 1)
        _verify_summary(
            row["uncached_ns_per_iteration"], uncached_per_iteration, seed + 2
        )
        ratios, ratio_summary = _ratio_summary(cached_totals, uncached_totals, seed)
        if not _exact_equal(row["block_ratios"], derived_ratios) or not _exact_equal(
            row["block_ratios"], ratios
        ):
            raise ValueError("raw block ratios mismatch")
        if not _exact_equal(row["ratio"], ratio_summary):
            raise ValueError("derived block ratio interval mismatch")
        if not _is_sha256(row["final_snapshot_sha256"]):
            raise ValueError("snapshot digest exact type/value mismatch")
        if row["final_snapshot_sha256"] != _expected_snapshot_digest(
            row["interval_count"]
        ):
            raise ValueError("snapshot digest semantic mismatch")
    if observed_pairs != expected_pairs:
        raise ValueError("row matrix membership/order mismatch")
    _require_keys(
        report["gate"],
        {
            "unchanged_n10000_upper95",
            "unchanged_n10000_limit",
            "unchanged_n10000_pass",
            "cached_n10000_over_n1000_median",
            "cached_scaling_limit",
            "cached_scaling_pass",
            "write_observe_worst_upper95",
            "write_observe_limit",
            "write_observe_pass",
            "decision",
        },
        "gate",
    )
    expected_gate = _gate(report["rows"])
    if not _exact_equal(report["gate"], expected_gate):
        raise ValueError("gate mismatch")
    provenance = _require_keys(
        report["provenance"],
        {"git", "sources", "runtime", "build", "backend"},
        "provenance",
    )
    git = _require_keys(
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
    if (
        not _is_git_oid(git["commit"])
        or not _is_git_oid(git["head_tree"])
        or type(git["clean_worktree"]) is not bool
        or type(git["changed_paths"]) is not list
        or type(git["staged_files"]) is not list
        or any(type(value) is not str for value in git["changed_paths"])
        or any(type(value) is not str for value in git["staged_files"])
        or not _is_sha256(git["source_state_sha256"])
    ):
        raise ValueError("git provenance exact type/value mismatch")
    if not _exact_equal(git, _git_metadata()):
        raise ValueError("git/source-state provenance does not match local checkout")
    if not _exact_equal(provenance["runtime"], _runtime_provenance()):
        raise ValueError("runtime provenance does not match current runtime")
    if not _exact_equal(provenance["build"], _build_provenance()):
        raise ValueError("build provenance does not match current build")
    expected_sources = [
        _file_provenance(Path(__file__)),
        _file_provenance(_REPOSITORY_ROOT / "treemendous" / "rangeset.py"),
    ]
    if not _exact_equal(provenance["sources"], expected_sources):
        raise ValueError("source provenance path/hash mismatch")
    if not _exact_equal(provenance["backend"], _backend_provenance()):
        raise ValueError("active backend identity/module/type/path/hash mismatch")
    canonical = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if canonical != encoded:
        raise ValueError("JSON is not canonical")
    if output.with_suffix(".md").read_text() != _markdown(report, digest):
        raise ValueError("Markdown mismatch")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--blocks", type=int, default=CONFIRMATION_BLOCKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        report = verify_artifacts(args.output)
    else:
        report = run_matrix(backend=args.backend, blocks=args.blocks)
        write_artifacts(report, args.output)
        verify_artifacts(args.output)
    print(
        json.dumps(
            {"artifact": str(args.output), "gate": report["gate"]}, sort_keys=True
        )
    )
    return 0 if args.verify or report["gate"]["decision"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
