#!/usr/bin/env python3
"""Strict vector-versus-segmented ExactBatch storage qualification evidence.

The controller never imports either native extension.  It launches isolated
workers with one checkout at the front of ``sys.path``, so the baseline and
candidate may safely expose the same ``treemendous.cpp._exact_batch`` name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "treemendous-exact-batch-storage-matrix-v1"
WORKER_SCHEMA = "treemendous-exact-batch-storage-worker-v1"
BASELINE_COMMIT = "2a384f7"
DEFAULT_BASELINE = Path("/private/tmp/treemendous-e4-baseline")
INTERVAL_COUNTS = (64, 1_000, 10_000, 100_000)
BATCH_SIZES = (0, 1, 16, 256)
LOCALITIES = ("head", "middle", "tail", "random")
DIAGNOSTIC_SHAPES = (
    *LOCALITIES,
    "strict_only",
    "duplicate_only",
    "wide_1",
    "wide_10",
    "wide_100",
)
BLOCK_COUNTS = (127, 128, 129)
BLOCK_SHAPES = ("split", "coalesce")
PROMOTION_BLOCKS = 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 672_128
K = 128
ARCHIVE_NAME = "exact-batch-storage-segmented-tuned-rejection.json"
ARCHIVE_SOURCE_ARTIFACT = (
    "build/experiments/exact-batch-storage-small-n-tuning-smoke.json"
)
SEGMENTED_PATCH = Path(
    "tests/performance/experiments/fixtures/exact_batch_segmented_tuned.patch"
)
SEGMENTED_PATCH_SHA256 = (
    "6b30ab72f11aa4fd0eec8ed05f534e1b96a68a2918f1b605b5aba866d6adb432"
)
BASELINE_FULL_COMMIT = "2a384f74d29949fefd0286a147b30c1ef0a190d4"
BASELINE_SOURCE_SHA256 = (
    "0dbbacc095e142f82235a4e0e6f8c73d6bd2abe70d10965219482b09a86bf0c8"
)
SEGMENTED_SOURCE_SHA256 = (
    "f5a368f011bcbbe9f49ba954b7014268ab7c711ecfb1d46b8b4d9da6a8858267"
)
BASELINE_BINARY_SHA256 = (
    "f48e8c03fbfc7a6f434753d84b1545080bc5e9df123cb4fd30d9da28769fb372"
)
SEGMENTED_BINARY_SHA256 = (
    "657bf80912e21be53572f10fd4375dbbdac1965e522072063f67ac24495002c5"
)
ARCHIVED_BASELINE_ROOT = "/private/tmp/treemendous-e4-baseline"
ARCHIVED_CANDIDATE_ROOT = "/Users/joseph/dev/Tree-Mendous"
ARCHIVED_TREE = "598588fba24adb869e5f26f16d9b6ec088a77e9c"
ARCHIVED_BASELINE_STATE_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
ARCHIVED_CANDIDATE_STATE_SHA256 = (
    "14674cdc450c8ba916946854ea38feeacc598ab6a3133d35179d016d9dc3f621"
)
ARCHIVED_BASELINE_CHANGED_PATHS_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
ARCHIVED_CANDIDATE_CHANGED_PATHS_SHA256 = (
    "d6bfb847dbbd2cd197df0835c939217a8b61f5780aba0720724d0013e1379dba"
)
ARCHIVED_PACKAGE_SHA256 = (
    "198f6ee8b1afd90a2fac90b925fe894e7ea3d2085aa16d9c3f2f0cc4df35b332"
)
ARCHIVED_BUILD = {
    "cc": "clang",
    "cflags": "-fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall",
    "command": "python setup.py build_ext --inplace --force",
    "cxx": "c++",
    "cxx_version": "Apple clang version 21.0.0 (clang-2100.1.1.101)",
}
ARCHIVED_RUNTIME = {
    "architecture": "64bit",
    "cpu_count": 18,
    "implementation": "CPython",
    "machine": "arm64",
    "platform": "macOS-26.5.1-arm64-arm-64bit",
    "python": "3.12.7",
    "python_compiler": "Clang 16.0.0 (clang-1600.0.26.3)",
}
COUNTER_KEYS = (
    "directory_entries_copied",
    "blocks_rebuilt",
    "intervals_copied",
    "block_delta",
    "generation",
    "live_count",
    "block_count",
    "transient_estimated_bytes",
    "retained_estimated_bytes",
)
RESOURCE_LIMITS = {
    "max_operations": 1_000_000,
    "max_live_intervals": 200_000,
    "max_changed_spans": 2_000_000,
    "max_result_bytes": 256 * 1024 * 1024,
    "max_work_units": 100_000_000,
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

Operation = tuple[int, int, int]


def _checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio_summary(ratios: list[float]) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    boot = [
        statistics.median(rng.choices(ratios, k=len(ratios)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return {
        "candidate_over_vector_samples": ratios,
        "candidate_over_vector_median": statistics.median(ratios),
        "candidate_over_vector_confidence_95": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
    }


def _packed(rows: tuple[Operation, ...]) -> bytes:
    import struct

    operation = struct.Struct("@qqq")
    return b"".join(operation.pack(*row) for row in rows)


def _regular_domain(count: int) -> tuple[tuple[int, int], ...]:
    return tuple((index * 8, index * 8 + 6) for index in range(count))


def _local_index(count: int, locality: str) -> int:
    if locality == "head":
        return 0
    if locality == "middle":
        return count // 2
    if locality == "tail":
        return count - 1
    if locality == "random":
        return random.Random(90_128 + count).randrange(count)
    raise ValueError(locality)


def _repeat_restorative(
    cycle: tuple[Operation, ...], batch_size: int
) -> tuple[Operation, ...]:
    if batch_size == 0:
        return ()
    if batch_size == 1:
        # A one-row restorative workload must be a no-op.
        return (cycle[-1],)
    return tuple(cycle[index % len(cycle)] for index in range(batch_size))


def _diagnostic_definition(count: int, batch_size: int, shape: str) -> dict[str, Any]:
    setup: tuple[Operation, ...] = ()
    cycle: tuple[Operation, ...]
    if shape.startswith("wide_"):
        percentage = int(shape.removeprefix("wide_"))
        width = 10_000
        domain = (
            (0, width),
            *tuple((width + 2 + i * 8, width + 8 + i * 8) for i in range(count - 1)),
        )
        extent = width * percentage // 100
        start = (width - extent) // 2
        end = start + extent
        cycle = ((1, start, end), (0, start, end), (0, start, end), (0, start, end))
    else:
        domain = _regular_domain(count)
        index = _local_index(count, shape) if shape in LOCALITIES else count // 2
        lower, _ = domain[index]
        if shape == "strict_only":
            setup = ((1, lower + 2, lower + 4),)
            cycle = ((2, lower + 2, lower + 4),)  # strict rejection
        elif shape == "duplicate_only":
            cycle = ((0, lower + 1, lower + 2),)  # already covered
        else:
            cycle = (
                (1, lower + 2, lower + 4),
                (0, lower + 2, lower + 4),
                (0, lower + 1, lower + 5),
                (0, lower + 1, lower + 5),
            )
    rows = _repeat_restorative(cycle, batch_size)
    return {
        "case_id": f"matrix-n{count}-b{batch_size}-{shape}",
        "interval_count": count,
        "batch_size": batch_size,
        "shape": shape,
        "domain": domain,
        "initially_available": True,
        "setup": setup,
        "rows": rows,
    }


def _block_definition(count: int, shape: str) -> dict[str, Any]:
    domain = ((0, count * 4 + 8),)
    setup = tuple((0, i * 4, i * 4 + 3) for i in range(count))
    middle = (count // 2) * 4
    if shape == "split":
        rows = ((1, middle + 1, middle + 2), (0, middle + 1, middle + 2))
    else:
        rows = ((0, middle + 3, middle + 4), (1, middle + 3, middle + 4))
    return {
        "case_id": f"block-k{count}-{shape}",
        "interval_count": count,
        "batch_size": 2,
        "shape": f"block_{shape}",
        "domain": domain,
        "initially_available": False,
        "setup": setup,
        "rows": rows,
    }


def diagnostic_definitions(profile: str = "full") -> tuple[dict[str, Any], ...]:
    counts = INTERVAL_COUNTS if profile == "full" else (64,)
    batches = BATCH_SIZES if profile == "full" else (0, 16)
    cases = tuple(
        _diagnostic_definition(count, batch, shape)
        for count in counts
        for batch in batches
        for shape in DIAGNOSTIC_SHAPES
    )
    blocks = tuple(
        _block_definition(count, shape)
        for count in BLOCK_COUNTS
        for shape in BLOCK_SHAPES
    )
    return (*cases, *blocks)


def _restorative64() -> tuple[Operation, ...]:
    rows: list[Operation] = []
    for index in (0, 7, 23, 41):
        start = index * 8
        rows.extend(
            (
                (1, start + 2, start + 6),
                (0, start + 2, start + 6),
                (0, start + 1, start + 5),
                (0, start + 1, start + 5),
            )
        )
    return tuple(rows)


def promotion_definitions(profile: str = "full") -> tuple[dict[str, Any], ...]:
    counts = INTERVAL_COUNTS if profile == "full" else (64,)
    cases: list[dict[str, Any]] = []
    local_rows: tuple[Operation, ...] = tuple(
        row
        for index in (0, 33_333, 66_666, 99_999)
        for row in (
            (1, index * 8 + 2, index * 8 + 4),
            (0, index * 8 + 2, index * 8 + 4),
            (0, index * 8 + 1, index * 8 + 5),
            (0, index * 8 + 1, index * 8 + 5),
        )
    )
    for count in counts:
        domain = _regular_domain(count)
        if count == 64:
            rows = _restorative64()
            cases.append(
                {
                    "case_id": "promotion-restorative-n64-b16",
                    "layer": "mutate",
                    "interval_count": count,
                    "domain": domain,
                    "initially_available": True,
                    "setup": (),
                    "rows": rows,
                    "iterations": 200,
                }
            )
        if count == 100_000:
            cases.append(
                {
                    "case_id": "promotion-local-n100000-b16",
                    "layer": "mutate",
                    "interval_count": count,
                    "domain": domain,
                    "initially_available": True,
                    "setup": (),
                    "rows": local_rows,
                    "iterations": 3,
                }
            )
            for percentage in (1, 10, 100):
                definition = _diagnostic_definition(count, 16, f"wide_{percentage}")
                cases.append(
                    {
                        **definition,
                        "case_id": f"promotion-wide-n100000-p{percentage}",
                        "layer": "mutate",
                        "iterations": 3,
                    }
                )
        cases.append(
            {
                "case_id": f"promotion-construction-n{count}",
                "layer": "construction",
                "interval_count": count,
                "domain": domain,
                "initially_available": True,
                "setup": (),
                "rows": (),
                "iterations": 1,
            }
        )
        cases.append(
            {
                "case_id": f"promotion-snapshot-n{count}",
                "layer": "snapshot",
                "interval_count": count,
                "domain": domain,
                "initially_available": True,
                "setup": (),
                "rows": (),
                "iterations": 3 if count >= 10_000 else 30,
            }
        )
        materialize_rows = (
            (1, domain[count // 2][0] + 2, domain[count // 2][0] + 4),
            (0, domain[count // 2][0] + 2, domain[count // 2][0] + 4),
        )
        cases.append(
            {
                "case_id": f"promotion-materialization-n{count}",
                "layer": "materialization",
                "interval_count": count,
                "domain": domain,
                "initially_available": True,
                "setup": (),
                "rows": materialize_rows,
                "iterations": 100,
            }
        )
    return tuple(cases)


def _is_descendant(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _file_metadata(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _purge_checkout_imports(root: Path) -> None:
    """Make the requested checkout the worker's only importable project root."""
    requested = root.resolve()
    current = _REPOSITORY_ROOT.resolve()
    for name in tuple(sys.modules):
        if (
            name == "treemendous"
            or name.startswith("treemendous.")
            or name == "tests"
            or name.startswith("tests.")
        ):
            del sys.modules[name]

    retained: list[str] = []
    for entry in sys.path:
        resolved = Path(entry or os.getcwd()).resolve()
        if resolved == requested:
            continue
        if _is_descendant(resolved, current) or _is_descendant(resolved, requested):
            continue
        retained.append(entry)
    sys.path[:] = [str(requested), *retained]

    # Editable installs register a meta-path finder from a site-packages .pth.
    # Workers also start with -I -S, but remove any such finder defensively.
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not type(finder).__module__.startswith("__editable__")
    ]
    for name in tuple(sys.modules):
        if name.startswith("__editable__"):
            del sys.modules[name]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()


def _worker_imports(root: Path) -> tuple[Any, Any, Any, Any, Any]:
    from treemendous import Span, create_range_set
    from treemendous.exact_batch import BatchLimits, ExactBatchRangeSet, MutationOpcode

    return Span, create_range_set, BatchLimits, ExactBatchRangeSet, MutationOpcode


def _import_metadata(root: Path) -> dict[str, dict[str, str]]:
    package = importlib.import_module("treemendous")
    native = importlib.import_module("treemendous.cpp._exact_batch")
    requested = root.resolve()
    package_file = package.__file__
    native_file = native.__file__
    if package_file is None or native_file is None:
        raise AssertionError("worker imports must resolve to concrete files")
    package_path = Path(package_file).resolve()
    native_path = Path(native_file).resolve()
    if not _is_descendant(package_path, requested):
        raise AssertionError(
            f"treemendous imported outside requested root {requested}: {package_path}"
        )
    if not _is_descendant(native_path, requested):
        raise AssertionError(
            f"native exact-batch module imported outside requested root {requested}: {native_path}"
        )
    return {
        "treemendous": _file_metadata(package_path),
        "native": _file_metadata(native_path),
    }


def _peak_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _geometry(snapshot: Any) -> list[list[int]]:
    return [[item.start, item.end] for item in snapshot.intervals]


def _serialize_results(results: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "changed": [[span.start, span.end] for span in result.changed],
            "changed_length": result.changed_length,
            "fully_covered": result.fully_covered,
        }
        for result in results
    ]


def _worker_case(definition: dict[str, Any], root: Path) -> dict[str, Any]:
    Span, create_range_set, BatchLimits, ExactBatchRangeSet, MutationOpcode = (
        _worker_imports(root)
    )
    limits = BatchLimits(**RESOURCE_LIMITS)
    domain = definition["domain"]
    setup = definition["setup"]
    rows = definition["rows"]

    def new_exact() -> Any:
        manager = ExactBatchRangeSet(
            domain, initially_available=definition["initially_available"], limits=limits
        )
        if setup:
            manager.mutate_packed(_packed(setup)).materialize()
        return manager

    def apply_scalar(
        manager: Any, operations: tuple[Operation, ...]
    ) -> tuple[Any, ...]:
        output = []
        for opcode, start, end in operations:
            span = Span(start, end)
            if opcode == MutationOpcode.ADD:
                output.append(manager.add(span))
            else:
                output.append(
                    manager.discard(
                        span,
                        require_covered=opcode
                        == MutationOpcode.DISCARD_REQUIRE_COVERED,
                    )
                )
        return tuple(output)

    scalar = create_range_set(
        domain,
        backend="cpp_boundary",
        initially_available=definition["initially_available"],
    )
    if setup:
        apply_scalar(scalar, setup)
    initial_geometry = _geometry(scalar.snapshot())
    expected = apply_scalar(scalar, rows)
    expected_final = _geometry(scalar.snapshot())
    work_units = 0
    work_scalar = create_range_set(
        domain,
        backend="cpp_boundary",
        initially_available=definition["initially_available"],
    )
    if setup:
        apply_scalar(work_scalar, setup)
    for row in rows:
        work_units += 1 + len(work_scalar.snapshot().intervals)
        apply_scalar(work_scalar, (row,))

    layer = definition.get("layer", "mutate")
    iterations = definition.get("iterations", 1)
    exact = new_exact()
    packed_rows = _packed(rows)
    result = None
    observed_snapshot = None
    materialized = None
    if layer == "construction":
        started = time.perf_counter_ns()
        for _ in range(iterations):
            exact = new_exact()
        elapsed = time.perf_counter_ns() - started
        observed_snapshot = exact.snapshot()
    elif layer == "mutate":
        started = time.perf_counter_ns()
        for _ in range(iterations):
            result = exact.mutate_packed(packed_rows)
        elapsed = time.perf_counter_ns() - started
        if result is None:
            raise AssertionError("timed mutation produced no result")
        materialized = result.materialize()
        observed_snapshot = exact.snapshot()
    elif layer == "snapshot":
        started = time.perf_counter_ns()
        for _ in range(iterations):
            observed_snapshot = exact.snapshot()
        elapsed = time.perf_counter_ns() - started
    elif layer == "materialization":
        result = exact.mutate_packed(packed_rows)
        started = time.perf_counter_ns()
        for _ in range(iterations):
            materialized = result.materialize()
        elapsed = time.perf_counter_ns() - started
        observed_snapshot = exact.snapshot()
    else:
        raise AssertionError(f"unknown layer {layer}")

    if observed_snapshot is None or _geometry(observed_snapshot) != expected_final:
        raise AssertionError(
            f"{definition['case_id']}: timed final snapshot differs from oracle"
        )
    if layer in ("mutate", "materialization") and materialized != expected:
        raise AssertionError(
            f"{definition['case_id']}: timed result differs from oracle"
        )
    if expected_final != initial_geometry:
        raise AssertionError(f"{definition['case_id']}: workload is not restorative")
    native = exact._manager
    counters = (
        native._storage_counters() if hasattr(native, "_storage_counters") else None
    )
    if counters is not None and (
        set(counters) != set(COUNTER_KEYS)
        or any(type(value) is not int for value in counters.values())
    ):
        raise AssertionError("candidate counter schema changed")
    return {
        "case_id": definition["case_id"],
        "layer": layer,
        "interval_count": definition["interval_count"],
        "batch_size": len(rows),
        "iterations": iterations,
        "elapsed_ns": elapsed,
        "ns_per_iteration": elapsed / iterations,
        "oracle_digest": _checksum(
            {"results": _serialize_results(expected), "final": expected_final}
        ),
        "legacy_work_units": work_units,
        "counters": counters,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _binary_metadata(root: Path) -> dict[str, str]:
    binaries = sorted((root / "treemendous/cpp").glob("_exact_batch*.so"))
    if len(binaries) != 1:
        raise ValueError(
            f"expected one exact-batch binary under {root}, found {len(binaries)}"
        )
    path = binaries[0].resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _worker_report(root: Path, mode: str, profile: str) -> dict[str, Any]:
    root = root.resolve()
    _purge_checkout_imports(root)
    _worker_imports(root)
    imports = _import_metadata(root)
    definitions = (
        diagnostic_definitions(profile)
        if mode == "diagnostic"
        else promotion_definitions(profile)
    )
    cells = [_worker_case(definition, root) for definition in definitions]
    retained = None
    if mode == "diagnostic" and profile == "full" and cells[0]["counters"] is not None:
        definition = next(
            item
            for item in promotion_definitions()
            if item["case_id"] == "promotion-local-n100000-b16"
        )
        _, _, BatchLimits, ExactBatchRangeSet, _ = _worker_imports(root)
        manager = ExactBatchRangeSet(
            definition["domain"],
            initially_available=True,
            limits=BatchLimits(**RESOURCE_LIMITS),
        )
        packed_rows = _packed(definition["rows"])
        manager.mutate_packed(packed_rows)
        warm = manager._manager._storage_counters()["retained_estimated_bytes"]
        for _ in range(1_000):
            manager.mutate_packed(packed_rows)
        final = manager._manager._storage_counters()["retained_estimated_bytes"]
        retained = {
            "post_warmup_bytes": warm,
            "after_1000_batches_bytes": final,
            "ratio": final / warm,
        }
    return {
        "schema": WORKER_SCHEMA,
        "mode": mode,
        "profile": profile,
        "root": str(root),
        "imports": imports,
        "binary": imports["native"],
        "cells": cells,
        "retained_memory": retained,
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def _assert_distinct_binaries(
    first: dict[str, str], second: dict[str, str], first_label: str, second_label: str
) -> None:
    if Path(first["path"]).resolve() == Path(second["path"]).resolve():
        raise ValueError(
            f"{first_label}/{second_label} resolved to the same native binary path: {first['path']}"
        )
    if first["sha256"] == second["sha256"]:
        raise ValueError(
            f"{first_label}/{second_label} resolved to the same native binary SHA-256: {first['sha256']}"
        )


def _validate_worker_report(
    report: dict[str, Any], root: Path, comparison_root: Path | None = None
) -> dict[str, Any]:
    requested = root.resolve()
    if Path(report.get("root", "")).resolve() != requested:
        raise ValueError(
            f"worker root mismatch: requested {requested}, reported {report.get('root')}"
        )
    imports = report.get("imports")
    if type(imports) is not dict or set(imports) != {"treemendous", "native"}:
        raise ValueError("worker import metadata schema mismatch")
    for label in ("treemendous", "native"):
        metadata = imports[label]
        if type(metadata) is not dict or set(metadata) != {"path", "sha256"}:
            raise ValueError(f"worker {label} import metadata schema mismatch")
        path = Path(metadata["path"]).resolve()
        if not _is_descendant(path, requested):
            raise ValueError(
                f"worker {label} import is outside requested root {requested}: {path}"
            )
        if _file_metadata(path) != metadata:
            raise ValueError(f"worker {label} import path/hash mismatch")
    if report.get("binary") != imports["native"]:
        raise ValueError("worker binary metadata does not match imported native module")
    if report["binary"] != _binary_metadata(requested):
        raise ValueError(
            "worker imported native module does not match requested-root binary"
        )
    if comparison_root is not None:
        _assert_distinct_binaries(
            report["binary"],
            _binary_metadata(comparison_root.resolve()),
            str(requested),
            str(comparison_root.resolve()),
        )
    return report


def _run_worker(
    root: Path,
    mode: str,
    profile: str,
    *,
    comparison_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    script = Path(__file__).resolve()
    bootstrap = (
        "import runpy,sys; "
        "root,script=sys.argv[1:3]; args=sys.argv[3:]; "
        "sys.path.insert(0,root); sys.argv=[script,*args]; "
        "runpy.run_path(script,run_name='__main__')"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap,
            str(root),
            str(script),
            "--worker",
            "--root",
            str(root),
            "--mode",
            mode,
            "--profile",
            profile,
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        completed.stdout,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    return _validate_worker_report(report, root, comparison_root)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _source_provenance(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain=v1")
    diff = _git(root, "diff", "--binary", "HEAD")
    changed = sorted(
        line
        for line in _git(
            root, "ls-files", "--modified", "--others", "--exclude-standard"
        ).splitlines()
        if line
    )
    digest = hashlib.sha256((status + diff).encode())
    for relative in changed:
        path = root / relative
        digest.update(relative.encode() + b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "root": str(root.resolve()),
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "tree": _git(root, "rev-parse", "HEAD^{tree}").strip(),
        "clean": not status,
        "changed_paths": changed,
        "source_state_sha256": digest.hexdigest(),
        "exact_batch_source_sha256": hashlib.sha256(
            (root / "treemendous/cpp/exact_batch_bindings.cpp").read_bytes()
        ).hexdigest(),
        "binary": _binary_metadata(root),
    }


def _provenance(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline_binary = _binary_metadata(baseline)
    candidate_binary = _binary_metadata(candidate)
    _assert_distinct_binaries(
        baseline_binary, candidate_binary, "baseline", "candidate"
    )
    compiler = os.environ.get("CXX", "c++")
    try:
        compiler_version = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        compiler_version = "unavailable"
    return {
        "baseline": _source_provenance(baseline),
        "candidate": _source_provenance(candidate),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "cpu_count": os.cpu_count() or 0,
        },
        "build": {
            "command": "python setup.py build_ext --inplace --force",
            "cxx": compiler,
            "cxx_version": compiler_version,
            "cc": str(sysconfig.get_config_var("CC") or "unknown"),
            "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
        },
    }


def _promotion_rows(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_ids = [cell["case_id"] for cell in blocks[0]["baseline"]["cells"]]
    output = []
    for case_id in case_ids:
        raw = []
        oracle = None
        work_units = None
        candidate_counters = []
        layer = ""
        count = 0
        for block in blocks:
            baseline = next(
                cell
                for cell in block["baseline"]["cells"]
                if cell["case_id"] == case_id
            )
            candidate = next(
                cell
                for cell in block["candidate"]["cells"]
                if cell["case_id"] == case_id
            )
            if (
                baseline["oracle_digest"] != candidate["oracle_digest"]
                or baseline["legacy_work_units"] != candidate["legacy_work_units"]
            ):
                raise AssertionError(
                    f"{case_id}: baseline/candidate oracle or legacy work differs"
                )
            oracle = baseline["oracle_digest"]
            work_units = baseline["legacy_work_units"]
            layer = baseline["layer"]
            count = baseline["interval_count"]
            raw.append(candidate["ns_per_iteration"] / baseline["ns_per_iteration"])
            if candidate["counters"] is not None:
                candidate_counters.append(candidate["counters"])
        output.append(
            {
                "case_id": case_id,
                "layer": layer,
                "interval_count": count,
                "oracle_digest": oracle,
                "legacy_work_units": work_units,
                "candidate_counters_by_block": candidate_counters,
                **_ratio_summary(raw),
            }
        )
    return output


def _gate(
    name: str, observed: float, threshold: float, comparison: str
) -> dict[str, Any]:
    passed = observed <= threshold if comparison == "<=" else observed < threshold
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
    }


def _gates(
    rows: list[dict[str, Any]], diagnostic: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {row["case_id"]: row for row in rows}
    gates = [
        _gate(
            "n64_upper95",
            by_id["promotion-restorative-n64-b16"][
                "candidate_over_vector_confidence_95"
            ][1],
            1.10,
            "<=",
        ),
        _gate(
            "n100000_local_median",
            by_id["promotion-local-n100000-b16"]["candidate_over_vector_median"],
            0.50,
            "<=",
        ),
        _gate(
            "n100000_local_upper95",
            by_id["promotion-local-n100000-b16"]["candidate_over_vector_confidence_95"][
                1
            ],
            0.60,
            "<",
        ),
    ]
    for percentage in (1, 10, 100):
        row = by_id[f"promotion-wide-n100000-p{percentage}"]
        gates.append(
            _gate(
                f"wide_{percentage}_upper95",
                row["candidate_over_vector_confidence_95"][1],
                1.15,
                "<=",
            )
        )
    for count in INTERVAL_COUNTS:
        row = by_id[f"promotion-snapshot-n{count}"]
        gates.append(
            _gate(
                f"snapshot_n{count}_upper95",
                row["candidate_over_vector_confidence_95"][1],
                1.15,
                "<=",
            )
        )
    counters = {cell["case_id"]: cell["counters"] for cell in diagnostic["cells"]}
    local_values = [
        counters[f"matrix-n{count}-b16-middle"]["intervals_copied"]
        for count in INTERVAL_COUNTS
    ]
    gates.append(
        _gate(
            "local_intervals_copied_bound", float(max(local_values)), float(8 * K), "<="
        )
    )
    gates.append(
        _gate(
            "local_copy_n_growth",
            float(max(local_values) - min(local_values)),
            float(K),
            "<=",
        )
    )
    retained = diagnostic["retained_memory"]
    gates.append(_gate("retained_bytes_after_1000", retained["ratio"], 1.10, "<="))
    return gates


def run_experiment(
    *,
    baseline_root: Path = DEFAULT_BASELINE,
    candidate_root: Path = _REPOSITORY_ROOT,
    blocks: int = PROMOTION_BLOCKS,
    profile: str = "full",
) -> dict[str, Any]:
    if profile == "full" and blocks < PROMOTION_BLOCKS:
        raise ValueError(
            f"full evidence requires at least {PROMOTION_BLOCKS} fixed blocks"
        )
    if profile == "smoke" and blocks < 2:
        raise ValueError("smoke evidence requires at least two blocks")
    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    baseline_commit = _git(baseline_root, "rev-parse", "HEAD").strip()
    if not baseline_commit.startswith(BASELINE_COMMIT):
        raise ValueError(
            f"baseline must be commit {BASELINE_COMMIT}, got {baseline_commit}"
        )
    diagnostic = _run_worker(
        candidate_root,
        "diagnostic",
        profile,
        comparison_root=baseline_root,
    )
    raw_blocks = []
    for index in range(blocks):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        measured = {
            name: _run_worker(
                baseline_root if name == "baseline" else candidate_root,
                "promotion",
                profile,
                comparison_root=(
                    candidate_root if name == "baseline" else baseline_root
                ),
            )
            for name in order
        }
        raw_blocks.append(
            {
                "block_index": index,
                "process_order": list(order),
                "baseline": measured["baseline"],
                "candidate": measured["candidate"],
            }
        )
    rows = _promotion_rows(raw_blocks)
    gates = _gates(rows, diagnostic) if profile == "full" else []
    return {
        "schema": SCHEMA,
        "profile": profile,
        "decision": "ACCEPTED"
        if gates and all(gate["passed"] for gate in gates)
        else ("REJECTED" if gates else "DIAGNOSTIC"),
        "provenance": _provenance(baseline_root, candidate_root),
        "methodology": {
            "fixed_balanced_blocks": blocks,
            "process_order": "baseline-first on even blocks; candidate-first on odd blocks",
            "worker_scope": "one isolated process measures the complete matrix for one root and block",
            "timed_instance": "restorative calls reuse one instance; last packed result and final snapshot are validated outside timing",
            "excluded": "manager/workload setup except construction cells, canonical scalar replay, packed-result validation, final snapshots, counters, RSS, and artifact writing",
            "bootstrap_unit": "whole balanced block candidate/vector ratio",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "resource_limits": dict(RESOURCE_LIMITS),
        "matrix_identity": {
            "interval_counts": list(INTERVAL_COUNTS if profile == "full" else (64,)),
            "batch_sizes": list(BATCH_SIZES if profile == "full" else (0, 16)),
            "diagnostic_shapes": list(DIAGNOSTIC_SHAPES),
            "block_counts": list(BLOCK_COUNTS),
            "block_shapes": list(BLOCK_SHAPES),
            "promotion_case_ids": [
                item["case_id"] for item in promotion_definitions(profile)
            ],
        },
        "diagnostic_candidate": diagnostic,
        "balanced_blocks": raw_blocks,
        "promotion_rows": rows,
        "gates": gates,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key: {key}")
        output[key] = value
    return output


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _validate_types(value: Any) -> None:
    if type(value) not in (dict, list, str, int, float, bool, type(None)):
        raise ValueError("artifact contains an unsupported JSON type")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("artifact contains a non-finite number")
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("artifact object key is not a string")
            _validate_types(item)
    elif type(value) is list:
        for item in value:
            _validate_types(item)


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} schema mismatch")
    return value


def _json_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _json_exact(actual[key], expected[key]) for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _json_exact(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _archived_rejection(report: dict[str, Any]) -> dict[str, Any]:
    row = next(
        item
        for item in report["promotion_rows"]
        if item["case_id"] == "promotion-restorative-n64-b16"
    )
    upper = row["candidate_over_vector_confidence_95"][1]
    return {
        "decision": "REJECTED" if upper > 1.10 else "ACCEPTED",
        "workload": row["case_id"],
        "balanced_blocks": len(report["balanced_blocks"]),
        "candidate_over_vector_median": row["candidate_over_vector_median"],
        "candidate_over_vector_confidence_95": row[
            "candidate_over_vector_confidence_95"
        ],
        "upper95_gate": 1.10,
    }


def archive_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Bind the durable rejection archive to its reproducible source patch."""
    return {
        "source_artifact": ARCHIVE_SOURCE_ARTIFACT,
        "patch": {
            "path": str(SEGMENTED_PATCH),
            "sha256": SEGMENTED_PATCH_SHA256,
        },
        "baseline": {
            "commit": BASELINE_FULL_COMMIT,
            "exact_batch_source_sha256": BASELINE_SOURCE_SHA256,
            "binary_sha256": BASELINE_BINARY_SHA256,
        },
        "candidate": {
            "result_exact_batch_source_sha256": SEGMENTED_SOURCE_SHA256,
            "binary_sha256": SEGMENTED_BINARY_SHA256,
        },
        "runtime": dict(ARCHIVED_RUNTIME),
        "rejection": _archived_rejection(report),
    }


def _verify_patch_application() -> None:
    patch = (_REPOSITORY_ROOT / SEGMENTED_PATCH).read_bytes()
    if hashlib.sha256(patch).hexdigest() != SEGMENTED_PATCH_SHA256:
        raise ValueError("segmented patch checksum mismatch")
    source = subprocess.run(
        [
            "git",
            "show",
            f"{BASELINE_FULL_COMMIT}:treemendous/cpp/exact_batch_bindings.cpp",
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(source).hexdigest() != BASELINE_SOURCE_SHA256:
        raise ValueError("baseline exact-batch source checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="treemendous-segmented-proof-") as raw:
        root = Path(raw)
        destination = root / "treemendous/cpp/exact_batch_bindings.cpp"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(source)
        completed = subprocess.run(
            ["patch", "-p1", "--batch", "--forward"],
            cwd=root,
            input=patch,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise ValueError("segmented patch does not apply to baseline source")
        if (
            hashlib.sha256(destination.read_bytes()).hexdigest()
            != SEGMENTED_SOURCE_SHA256
        ):
            raise ValueError("segmented patch result checksum mismatch")


def _offline_cell_expected(definition: dict[str, Any]) -> tuple[str, int]:
    """Reconstruct oracle and work declarations without segmented runtime hooks."""
    from treemendous import Span, create_range_set

    manager = create_range_set(
        definition["domain"],
        backend="py_boundary",
        initially_available=definition["initially_available"],
    )

    def apply(operations: tuple[Operation, ...]) -> tuple[Any, ...]:
        output = []
        for opcode, start, end in operations:
            span = Span(start, end)
            if opcode == 0:
                output.append(manager.add(span))
            else:
                output.append(manager.discard(span, require_covered=opcode == 2))
        return tuple(output)

    if definition["setup"]:
        apply(definition["setup"])
    work_units = 0
    results: list[Any] = []
    for operation in definition["rows"]:
        work_units += 1 + len(manager.snapshot().intervals)
        results.extend(apply((operation,)))
    final = _geometry(manager.snapshot())
    return (
        _checksum({"results": _serialize_results(tuple(results)), "final": final}),
        work_units,
    )


def _sha256_string(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_archived_worker(
    report: Any,
    *,
    root_role: str,
    mode: str,
    definitions: tuple[dict[str, Any], ...],
) -> None:
    worker = _exact_keys(
        report,
        {
            "schema",
            "mode",
            "profile",
            "root",
            "imports",
            "binary",
            "cells",
            "retained_memory",
            "process_peak_rss_bytes",
        },
        "archived worker",
    )
    if (
        worker["schema"] != WORKER_SCHEMA
        or worker["mode"] != mode
        or worker["profile"] != "smoke"
        or type(worker["root"]) is not str
        or not Path(worker["root"]).is_absolute()
        or worker["retained_memory"] is not None
        or type(worker["process_peak_rss_bytes"]) is not int
        or worker["process_peak_rss_bytes"] <= 0
    ):
        raise ValueError("archived worker identity/profile mismatch")
    imports = _exact_keys(
        worker["imports"], {"treemendous", "native"}, "archived imports"
    )
    for label in ("treemendous", "native"):
        metadata = _exact_keys(
            imports[label], {"path", "sha256"}, f"archived {label} import"
        )
        if (
            type(metadata["path"]) is not str
            or not Path(metadata["path"]).is_absolute()
            or not Path(metadata["path"]).is_relative_to(Path(worker["root"]))
            or not _sha256_string(metadata["sha256"])
        ):
            raise ValueError("archived import path/hash mismatch")
    if worker["binary"] != imports["native"]:
        raise ValueError("archived binary/import mismatch")
    expected_binary = (
        BASELINE_BINARY_SHA256 if root_role == "baseline" else SEGMENTED_BINARY_SHA256
    )
    expected_root = (
        ARCHIVED_BASELINE_ROOT if root_role == "baseline" else ARCHIVED_CANDIDATE_ROOT
    )
    if (
        worker["root"] != expected_root
        or worker["binary"]["sha256"] != expected_binary
        or imports["treemendous"]["sha256"] != ARCHIVED_PACKAGE_SHA256
    ):
        raise ValueError("archived root/import/binary provenance mismatch")

    cells = worker["cells"]
    if type(cells) is not list or len(cells) != len(definitions):
        raise ValueError("archived worker matrix length mismatch")
    cell_keys = {
        "case_id",
        "layer",
        "interval_count",
        "batch_size",
        "iterations",
        "elapsed_ns",
        "ns_per_iteration",
        "oracle_digest",
        "legacy_work_units",
        "counters",
        "peak_rss_bytes",
    }
    observed_ids = []
    for cell, definition in zip(cells, definitions, strict=True):
        _exact_keys(cell, cell_keys, "archived worker cell")
        if type(cell["case_id"]) is not str or cell["case_id"] != definition["case_id"]:
            raise ValueError("archived worker matrix order/uniqueness mismatch")
        observed_ids.append(cell["case_id"])
        expected_oracle, expected_work = _offline_cell_expected(definition)
        expected_layer = definition.get("layer", "mutate")
        expected_iterations = definition.get("iterations", 1)
        if (
            type(cell["case_id"]) is not str
            or cell["case_id"] != definition["case_id"]
            or type(cell["layer"]) is not str
            or cell["layer"] != expected_layer
            or type(cell["interval_count"]) is not int
            or cell["interval_count"] != definition["interval_count"]
            or type(cell["batch_size"]) is not int
            or cell["batch_size"] != len(definition["rows"])
            or type(cell["iterations"]) is not int
            or cell["iterations"] != expected_iterations
            or type(cell["elapsed_ns"]) is not int
            or cell["elapsed_ns"] <= 0
            or type(cell["ns_per_iteration"]) is not float
            or cell["ns_per_iteration"] != cell["elapsed_ns"] / expected_iterations
            or type(cell["oracle_digest"]) is not str
            or cell["oracle_digest"] != expected_oracle
            or type(cell["legacy_work_units"]) is not int
            or cell["legacy_work_units"] != expected_work
            or type(cell["peak_rss_bytes"]) is not int
            or cell["peak_rss_bytes"] <= 0
        ):
            raise ValueError("archived worker cell derivation mismatch")
        counters = cell["counters"]
        if root_role == "baseline":
            if counters is not None:
                raise ValueError("archived baseline unexpectedly exposes counters")
        else:
            _exact_keys(counters, set(COUNTER_KEYS), "archived candidate counters")
            if any(type(value) is not int for value in counters.values()):
                raise ValueError("archived candidate counter type mismatch")
    expected_ids = [definition["case_id"] for definition in definitions]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(expected_ids):
        raise ValueError("archived worker matrix order/uniqueness mismatch")


def render_markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Segmented ExactBatch storage qualification",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Balanced blocks: `{report['methodology']['fixed_balanced_blocks']}`",
        f"- Baseline: `{report['provenance']['baseline']['commit']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
    ]
    if "archive" in report:
        archive = report["archive"]
        lines.extend(
            (
                f"- Archived segmented decision: **{archive['rejection']['decision']}**",
                f"- Segmented patch SHA-256: `{archive['patch']['sha256']}`",
                f"- Resulting source SHA-256: `{archive['candidate']['result_exact_batch_source_sha256']}`",
                "",
            )
        )
    lines.extend(
        [
            "| Cell | Median candidate/vector | Upper 95% |",
            "|---|---:|---:|",
        ]
    )
    for row in report["promotion_rows"]:
        lines.append(
            f"| {row['case_id']} | {row['candidate_over_vector_median']:.4f} | {row['candidate_over_vector_confidence_95'][1]:.4f} |"
        )
    lines.extend(
        ("", "| Gate | Observed | Threshold | Result |", "|---|---:|---:|---|")
    )
    for gate in report["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['observed']:.6g} | {gate['comparison']} {gate['threshold']:.6g} | {'pass' if gate['passed'] else 'fail'} |"
        )
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    if output.suffix != ".json":
        raise ValueError("output must have .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    for path, content in (
        (output, encoded),
        (markdown, render_markdown(report, digest).encode()),
        (checksum, f"{digest}  {output.name}\n".encode()),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return output, markdown, checksum


def verify_artifacts(output: Path) -> dict[str, Any]:
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    if not all(path.is_file() for path in (output, markdown, checksum)):
        raise ValueError("artifact triplet is incomplete")
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum.read_text() != f"{digest}  {output.name}\n":
        raise ValueError("checksum mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    _validate_types(report)
    if encoded != (json.dumps(report, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("JSON is not canonical")
    _exact_keys(
        report,
        {
            "schema",
            "profile",
            "decision",
            "provenance",
            "methodology",
            "resource_limits",
            "matrix_identity",
            "diagnostic_candidate",
            "balanced_blocks",
            "promotion_rows",
            "gates",
        },
        "report",
    )
    if report["schema"] != SCHEMA or report["profile"] not in ("full", "smoke"):
        raise ValueError("schema/profile mismatch")
    if report["resource_limits"] != RESOURCE_LIMITS:
        raise ValueError("resource limits mismatch")
    profile = report["profile"]
    expected_identity = {
        "interval_counts": list(INTERVAL_COUNTS if profile == "full" else (64,)),
        "batch_sizes": list(BATCH_SIZES if profile == "full" else (0, 16)),
        "diagnostic_shapes": list(DIAGNOSTIC_SHAPES),
        "block_counts": list(BLOCK_COUNTS),
        "block_shapes": list(BLOCK_SHAPES),
        "promotion_case_ids": [
            item["case_id"] for item in promotion_definitions(profile)
        ],
    }
    if report["matrix_identity"] != expected_identity:
        raise ValueError("matrix identity mismatch")
    blocks = report["balanced_blocks"]
    required = PROMOTION_BLOCKS if profile == "full" else 2
    if type(blocks) is not list or len(blocks) < required:
        raise ValueError("balanced block count is insufficient")
    provenance = report["provenance"]
    baseline = Path(provenance["baseline"]["root"])
    candidate = Path(provenance["candidate"]["root"])
    _validate_worker_report(report["diagnostic_candidate"], candidate, baseline)
    for index, block in enumerate(blocks):
        expected_order = (
            ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        )
        if block["block_index"] != index or block["process_order"] != expected_order:
            raise ValueError("balanced process order mismatch")
        _validate_worker_report(block["baseline"], baseline, candidate)
        _validate_worker_report(block["candidate"], candidate, baseline)
    reconstructed_rows = _promotion_rows(blocks)
    if report["promotion_rows"] != reconstructed_rows:
        raise ValueError("promotion derivation mismatch")
    if provenance != _provenance(baseline, candidate):
        raise ValueError("source/runtime/compiler/binary provenance mismatch")
    expected_gates = (
        _gates(reconstructed_rows, report["diagnostic_candidate"])
        if profile == "full"
        else []
    )
    if report["gates"] != expected_gates:
        raise ValueError("gate derivation mismatch")
    expected_decision = (
        "ACCEPTED"
        if expected_gates and all(item["passed"] for item in expected_gates)
        else ("REJECTED" if expected_gates else "DIAGNOSTIC")
    )
    if report["decision"] != expected_decision:
        raise ValueError("decision mismatch")
    if markdown.read_text() != render_markdown(report, digest):
        raise ValueError("Markdown mismatch")
    return report


def verify_archive_artifacts(output: Path) -> dict[str, Any]:
    """Verify the durable historical archive without loading its old binaries."""
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    if output.name != ARCHIVE_NAME:
        raise ValueError("archive filename mismatch")
    if not all(path.is_file() for path in (output, markdown, checksum)):
        raise ValueError("archive triplet is incomplete")
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum.read_text() != f"{digest}  {output.name}\n":
        raise ValueError("archive checksum mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    _validate_types(report)
    if encoded != (json.dumps(report, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("archive JSON is not canonical")
    _exact_keys(
        report,
        {
            "schema",
            "profile",
            "decision",
            "provenance",
            "methodology",
            "resource_limits",
            "matrix_identity",
            "diagnostic_candidate",
            "balanced_blocks",
            "promotion_rows",
            "gates",
            "archive",
        },
        "archive report",
    )
    if (
        report["schema"] != SCHEMA
        or report["profile"] != "smoke"
        or report["decision"] != "DIAGNOSTIC"
        or report["gates"] != []
        or not _json_exact(report["resource_limits"], RESOURCE_LIMITS)
    ):
        raise ValueError("archive report identity mismatch")
    expected_methodology = {
        "fixed_balanced_blocks": PROMOTION_BLOCKS,
        "process_order": "baseline-first on even blocks; candidate-first on odd blocks",
        "worker_scope": "one isolated process measures the complete matrix for one root and block",
        "timed_instance": "restorative calls reuse one instance; last packed result and final snapshot are validated outside timing",
        "excluded": "manager/workload setup except construction cells, canonical scalar replay, packed-result validation, final snapshots, counters, RSS, and artifact writing",
        "bootstrap_unit": "whole balanced block candidate/vector ratio",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if not _json_exact(report["methodology"], expected_methodology):
        raise ValueError("archive methodology mismatch")
    expected_identity = {
        "interval_counts": [64],
        "batch_sizes": [0, 16],
        "diagnostic_shapes": list(DIAGNOSTIC_SHAPES),
        "block_counts": list(BLOCK_COUNTS),
        "block_shapes": list(BLOCK_SHAPES),
        "promotion_case_ids": [
            definition["case_id"] for definition in promotion_definitions("smoke")
        ],
    }
    if not _json_exact(report["matrix_identity"], expected_identity):
        raise ValueError("archive matrix identity mismatch")

    provenance = _exact_keys(
        report["provenance"],
        {"baseline", "candidate", "runtime", "build"},
        "archive provenance",
    )
    source_keys = {
        "root",
        "commit",
        "tree",
        "clean",
        "changed_paths",
        "source_state_sha256",
        "exact_batch_source_sha256",
        "binary",
    }
    for role in ("baseline", "candidate"):
        source = _exact_keys(provenance[role], source_keys, f"archive {role}")
        binary = _exact_keys(source["binary"], {"path", "sha256"}, "archive binary")
        if (
            type(source["root"]) is not str
            or not Path(source["root"]).is_absolute()
            or type(source["commit"]) is not str
            or source["commit"] != BASELINE_FULL_COMMIT
            or type(source["tree"]) is not str
            or source["tree"] != ARCHIVED_TREE
            or type(source["clean"]) is not bool
            or type(source["changed_paths"]) is not list
            or any(type(path) is not str for path in source["changed_paths"])
            or source["changed_paths"] != sorted(set(source["changed_paths"]))
            or not _sha256_string(source["source_state_sha256"])
            or type(binary["path"]) is not str
            or not Path(binary["path"]).is_absolute()
            or not Path(binary["path"]).is_relative_to(Path(source["root"]))
        ):
            raise ValueError("archive source provenance type/path mismatch")
    baseline_source = provenance["baseline"]
    candidate_source = provenance["candidate"]
    baseline_changed_digest = hashlib.sha256(
        json.dumps(baseline_source["changed_paths"], separators=(",", ":")).encode()
    ).hexdigest()
    candidate_changed_digest = hashlib.sha256(
        json.dumps(candidate_source["changed_paths"], separators=(",", ":")).encode()
    ).hexdigest()
    if (
        baseline_source["root"] != ARCHIVED_BASELINE_ROOT
        or candidate_source["root"] != ARCHIVED_CANDIDATE_ROOT
        or baseline_source["clean"] is not True
        or baseline_source["changed_paths"] != []
        or baseline_changed_digest != ARCHIVED_BASELINE_CHANGED_PATHS_SHA256
        or candidate_changed_digest != ARCHIVED_CANDIDATE_CHANGED_PATHS_SHA256
        or baseline_source["source_state_sha256"] != ARCHIVED_BASELINE_STATE_SHA256
        or candidate_source["source_state_sha256"] != ARCHIVED_CANDIDATE_STATE_SHA256
        or baseline_source["exact_batch_source_sha256"] != BASELINE_SOURCE_SHA256
        or baseline_source["binary"]["sha256"] != BASELINE_BINARY_SHA256
        or candidate_source["clean"] is not False
        or "treemendous/cpp/exact_batch_bindings.cpp"
        not in candidate_source["changed_paths"]
        or candidate_source["exact_batch_source_sha256"] != SEGMENTED_SOURCE_SHA256
        or candidate_source["binary"]["sha256"] != SEGMENTED_BINARY_SHA256
        or baseline_source["root"] == candidate_source["root"]
        or baseline_source["binary"]["path"] == candidate_source["binary"]["path"]
        or baseline_source["binary"]["sha256"] == candidate_source["binary"]["sha256"]
    ):
        raise ValueError("archive source/binary provenance mismatch")
    if not _json_exact(provenance["runtime"], ARCHIVED_RUNTIME):
        raise ValueError("archive runtime provenance mismatch")
    build = _exact_keys(
        provenance["build"],
        {"command", "cxx", "cxx_version", "cc", "cflags"},
        "archive build",
    )
    if not _json_exact(build, ARCHIVED_BUILD):
        raise ValueError("archive build provenance mismatch")

    promotion_definitions_smoke = promotion_definitions("smoke")
    diagnostic_definitions_smoke = diagnostic_definitions("smoke")
    diagnostic = report["diagnostic_candidate"]
    _verify_archived_worker(
        diagnostic,
        root_role="candidate",
        mode="diagnostic",
        definitions=diagnostic_definitions_smoke,
    )
    if diagnostic["root"] != candidate_source["root"]:
        raise ValueError("archive diagnostic candidate root mismatch")
    blocks = report["balanced_blocks"]
    if type(blocks) is not list or len(blocks) != PROMOTION_BLOCKS:
        raise ValueError("archive exact balanced block count mismatch")
    for index, block in enumerate(blocks):
        _exact_keys(
            block,
            {"block_index", "process_order", "baseline", "candidate"},
            "archive balanced block",
        )
        expected_order = (
            ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        )
        if (
            type(block["block_index"]) is not int
            or block["block_index"] != index
            or block["process_order"] != expected_order
        ):
            raise ValueError("archive balanced block ordering mismatch")
        _verify_archived_worker(
            block["baseline"],
            root_role="baseline",
            mode="promotion",
            definitions=promotion_definitions_smoke,
        )
        _verify_archived_worker(
            block["candidate"],
            root_role="candidate",
            mode="promotion",
            definitions=promotion_definitions_smoke,
        )
        if (
            block["baseline"]["root"] != baseline_source["root"]
            or block["candidate"]["root"] != candidate_source["root"]
        ):
            raise ValueError("archive worker/root provenance mismatch")

    reconstructed_rows = _promotion_rows(blocks)
    if not _json_exact(report["promotion_rows"], reconstructed_rows):
        raise ValueError("archive promotion raw derivation mismatch")
    expected_row_ids = [item["case_id"] for item in promotion_definitions_smoke]
    observed_row_ids = [item["case_id"] for item in report["promotion_rows"]]
    if observed_row_ids != expected_row_ids or len(set(observed_row_ids)) != len(
        expected_row_ids
    ):
        raise ValueError("archive promotion matrix order/uniqueness mismatch")
    if not _json_exact(report["archive"], archive_metadata(report)):
        raise ValueError("archive patch/rejection binding mismatch")
    _verify_patch_application()
    if markdown.read_text() != render_markdown(report, digest):
        raise ValueError("archive Markdown mismatch")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--blocks", type=int, default=PROMOTION_BLOCKS)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/experiments/exact-batch-storage-matrix.json"),
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="verify the durable historical rejection without old checkout binaries",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("diagnostic", "promotion"), help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        if args.root is None or args.mode is None:
            raise SystemExit("worker requires --root and --mode")
        print(
            json.dumps(
                _worker_report(args.root.resolve(), args.mode, args.profile),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if args.archive:
        if not args.verify:
            raise SystemExit("--archive requires --verify")
        report = verify_archive_artifacts(args.output)
    elif args.verify:
        report = verify_artifacts(args.output)
    else:
        report = run_experiment(
            baseline_root=args.baseline_root,
            candidate_root=args.candidate_root,
            blocks=args.blocks,
            profile=args.profile,
        )
        write_artifacts(report, args.output)
        verify_artifacts(args.output)
    print(
        f"decision={report['decision']} blocks={len(report['balanced_blocks'])} output={args.output}"
    )
    return int(report["decision"] == "REJECTED")


if __name__ == "__main__":
    raise SystemExit(main())
