"""Paired, correctness-attested attribution for native exact-delta mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

SCHEMA = "treemendous-mutation-attribution-v1"
PRIMARY_LAYER = "rangeset_public"
REQUIRED_LAYERS = (
    "binding_no_result",
    "binding_result",
    "adapter",
    PRIMARY_LAYER,
    "observed_publication",
)
FOCUSED_WORKLOAD_NAMES = (
    "no-op-add-discard",
    "allocation-hit",
    "allocation-miss",
    "allocation-bounded-miss",
)
REPRESENTATIVE_MANIFEST_SCHEMA = "treemendous-representative-workload-manifest-v1"
PYTHON_CONTROLS = (
    "py_boundary",
    "py_avl_earliest",
    "py_summary",
    "py_treap",
    "py_boundary_summary",
)


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_statistics(
    baseline_ns: list[int],
    candidate_ns: list[int],
    *,
    ratio_limit: float | None,
    seed: int = 50,
) -> dict[str, Any]:
    """Return paired statistics and an optional policy classification."""
    if len(baseline_ns) != len(candidate_ns):
        raise ValueError("paired timing lists must have equal length")
    if len(baseline_ns) < 20:
        raise ValueError("paired comparisons require at least 20 samples")
    if ratio_limit is not None and ratio_limit <= 0:
        raise ValueError("ratio limit must be positive")
    if any(value <= 0 for value in (*baseline_ns, *candidate_ns)):
        raise ValueError("timing samples must be positive")

    ratios = [
        candidate / baseline
        for baseline, candidate in zip(baseline_ns, candidate_ns, strict=True)
    ]
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10_000)
    ]
    median_ratio = statistics.median(ratios)
    confidence = (
        _percentile(bootstrap, 0.025),
        _percentile(bootstrap, 0.975),
    )
    if ratio_limit is None:
        classification = "not-evaluated"
    elif median_ratio > ratio_limit:
        classification = "fail"
    elif confidence[1] <= ratio_limit:
        classification = "pass"
    else:
        classification = "inconclusive"
    return {
        "sample_count": len(ratios),
        "baseline_samples_ns": baseline_ns,
        "candidate_samples_ns": candidate_ns,
        "paired_ratios": ratios,
        "median_ratio": median_ratio,
        "median_improvement": 1.0 - median_ratio,
        "confidence_95_ratio": confidence,
        "ratio_limit": ratio_limit,
        "classification": classification,
    }


def _runs(points: frozenset[int]) -> tuple[tuple[int, int], ...]:
    if not points:
        return ()
    ordered = sorted(points)
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for point in ordered[1:]:
        if point != previous + 1:
            result.append((start, previous + 1))
            start = point
        previous = point
    result.append((start, previous + 1))
    return tuple(result)


def _points(intervals: Any) -> frozenset[int]:
    result: set[int] = set()
    for item in intervals:
        if isinstance(item, tuple):
            start, end = item
        else:
            start, end = item.start, item.end
        result.update(range(start, end))
    return frozenset(result)


def _evidence(
    kind: str, before: frozenset[int], after: frozenset[int], start: int, end: int
) -> dict[str, Any]:
    target = frozenset(range(start, end))
    changed = after - before if kind == "add" else before - after
    return {
        "kind": kind,
        "changed": _runs(changed),
        "changed_length": len(changed),
        "fully_covered": target <= before,
    }


def _result_evidence(kind: str, result: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "changed": tuple((span.start, span.end) for span in result.changed),
        "changed_length": result.changed_length,
        "fully_covered": result.fully_covered,
    }


def _trace() -> Any:
    from tests.performance.workload import canonical_mutation_workload

    return canonical_mutation_workload(
        interval_count=64, operation_count=1_000, seed=50
    )


def _focused_workloads() -> tuple[Any, ...]:
    """Return independent no-op and allocation regression traces."""
    from tests.performance.harness import BenchmarkWorkload, Operation

    no_op = BenchmarkWorkload(
        "no-op-add-discard",
        ((0, 256),),
        (Operation("add", start=0, end=128),),
        tuple(
            Operation("add", start=0, end=128)
            if index % 2 == 0
            else Operation("discard", start=128, end=256)
            for index in range(1_000)
        ),
        256,
        (("mutation_effect", "all operations are no-ops"),),
    )
    allocation_hit = BenchmarkWorkload(
        "allocation-hit",
        ((0, 4_096),),
        (Operation("add", start=0, end=4_096),),
        tuple(
            Operation("allocate", length=8, not_before=0, job_id=index)
            for index in range(256)
        ),
        4_096,
        (("allocation_result", "all requests succeed"),),
    )
    fragmented_setup = tuple(
        Operation("add", start=index * 4, end=index * 4 + 2) for index in range(64)
    )
    allocation_miss = BenchmarkWorkload(
        "allocation-miss",
        ((0, 256),),
        fragmented_setup,
        tuple(Operation("allocate", length=3, not_before=0) for _ in range(1_000)),
        256,
        (("allocation_result", "unbounded fragmented miss"),),
    )
    allocation_bounded_miss = BenchmarkWorkload(
        "allocation-bounded-miss",
        ((0, 256),),
        (Operation("add", start=0, end=64),),
        tuple(
            Operation("allocate", length=8, not_before=60, not_after=64)
            for _ in range(1_000)
        ),
        256,
        (("allocation_result", "bounded miss"),),
    )
    return no_op, allocation_hit, allocation_miss, allocation_bounded_miss


def _workload_manifest_entry(workload: Any) -> dict[str, Any]:
    return {
        "name": workload.name,
        "domain": workload.domain,
        "setup": [asdict(operation) for operation in workload.setup],
        "operations": [asdict(operation) for operation in workload.operations],
        "coordinate_extent": workload.coordinate_extent,
        "dimensions": list(workload.dimensions),
    }


def representative_workload_manifest() -> dict[str, Any]:
    """Return the exact, versioned standard representative workload manifest."""
    from tests.performance.profiles import benchmark_profile

    workloads = [
        _workload_manifest_entry(workload)
        for workload in benchmark_profile("standard").sampled_workloads
    ]
    try:
        body = json.loads(
            json.dumps(
                {
                    "schema": REPRESENTATIVE_MANIFEST_SCHEMA,
                    "profile": "standard",
                    "workloads": workloads,
                }
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("representative workload manifest is not JSON-safe") from exc
    return {**body, "digest": _checksum(body)}


def _trace_digest(workload: Any) -> str:
    return _checksum(_workload_manifest_entry(workload))


def _expected_evidence(workload: Any) -> tuple[list[dict[str, Any]], Any]:
    free: frozenset[int] = frozenset()
    for operation in workload.setup:
        assert operation.start is not None and operation.end is not None
        free |= frozenset(range(operation.start, operation.end))
    evidence: list[dict[str, Any]] = []
    for operation in workload.operations:
        assert operation.start is not None and operation.end is not None
        before = free
        target = frozenset(range(operation.start, operation.end))
        free = free | target if operation.kind == "add" else free - target
        evidence.append(
            _evidence(
                operation.kind,
                before,
                free,
                operation.start,
                operation.end,
            )
        )
    return evidence, _runs(free)


def _new_raw(workload: Any) -> Any:
    from treemendous.cpp.boundary import IntervalManager  # type: ignore[import-untyped]

    manager = IntervalManager()
    manager.set_managed_domain(workload.domain)
    for operation in workload.setup:
        assert operation.start is not None and operation.end is not None
        manager.release_interval(operation.start, operation.end)
    return manager


def _time_calls(operations: Any, invoke: Callable[[Any], Any]) -> int:
    total = 0
    for operation in operations:
        started = time.perf_counter_ns()
        invoke(operation)
        total += time.perf_counter_ns() - started
    return total


def _raw_no_result_layer(workload: Any) -> tuple[int, list[dict[str, Any]], Any]:
    def prepared_calls(manager: Any) -> tuple[tuple[Any, int, int, str], ...]:
        calls: list[tuple[Any, int, int, str]] = []
        for operation in workload.operations:
            assert operation.start is not None and operation.end is not None
            method = (
                manager.release_interval
                if operation.kind == "add"
                else manager.reserve_interval
            )
            calls.append((method, operation.start, operation.end, operation.kind))
        return tuple(calls)

    def invoke(call: tuple[Any, int, int, str]) -> Any:
        method, start, end, _ = call
        return method(start, end)

    timed = _new_raw(workload)
    execution_ns = _time_calls(prepared_calls(timed), invoke)
    timed_state = tuple(timed.get_intervals())

    validation = _new_raw(workload)
    evidence: list[dict[str, Any]] = []
    for call in prepared_calls(validation):
        _, start, end, kind = call
        before = _points(validation.get_intervals())
        invoke(call)
        after = _points(validation.get_intervals())
        evidence.append(_evidence(kind, before, after, start, end))
    if tuple(validation.get_intervals()) != timed_state:
        raise AssertionError("no-result validation replay diverged")
    return execution_ns, evidence, timed_state


def _binding_result_layer(workload: Any) -> tuple[int, list[dict[str, Any]], Any]:
    def prepared_calls(manager: Any) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        calls: list[tuple[str, Any, tuple[Any, ...]]] = []
        for operation in workload.operations:
            assert operation.start is not None and operation.end is not None
            if operation.kind == "add":
                calls.append(
                    (
                        "add",
                        manager.release_with_delta,
                        (operation.start, operation.end),
                    )
                )
            else:
                calls.append(
                    (
                        "discard",
                        manager.reserve_with_delta,
                        (operation.start, operation.end, False),
                    )
                )
        return tuple(calls)

    def invoke(call: tuple[str, Any, tuple[Any, ...]]) -> Any:
        _, method, arguments = call
        return method(*arguments)

    timed = _new_raw(workload)
    execution_ns = _time_calls(prepared_calls(timed), invoke)
    timed_state = tuple(timed.get_intervals())

    validation = _new_raw(workload)
    evidence = [
        _result_evidence(call[0], invoke(call)) for call in prepared_calls(validation)
    ]
    if tuple(validation.get_intervals()) != timed_state:
        raise AssertionError("binding-result validation replay diverged")
    return execution_ns, evidence, timed_state


def _adapter_layer(workload: Any) -> tuple[int, list[dict[str, Any]], Any]:
    from treemendous.backends.adapters import BackendAdapter

    def new_adapter() -> Any:
        return BackendAdapter(_new_raw(workload))

    def prepared_calls(adapter: Any) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        calls: list[tuple[str, Any, tuple[Any, ...]]] = []
        for operation in workload.operations:
            assert operation.start is not None and operation.end is not None
            if operation.kind == "add":
                calls.append(
                    (
                        "add",
                        adapter.release_with_delta,
                        (operation.start, operation.end),
                    )
                )
            else:
                calls.append(
                    (
                        "discard",
                        adapter.reserve_with_delta,
                        (operation.start, operation.end, False),
                    )
                )
        return tuple(calls)

    def invoke(call: tuple[str, Any, tuple[Any, ...]]) -> Any:
        _, method, arguments = call
        return method(*arguments)

    timed = new_adapter()
    execution_ns = _time_calls(prepared_calls(timed), invoke)
    timed_state = tuple(timed.implementation.get_intervals())

    validation = new_adapter()
    evidence = [
        _result_evidence(call[0], invoke(call)) for call in prepared_calls(validation)
    ]
    validation_state = tuple(validation.implementation.get_intervals())
    if validation_state != timed_state:
        raise AssertionError("adapter validation replay diverged")
    return execution_ns, evidence, timed_state


def _rangeset_layer(
    workload: Any, *, observe: bool
) -> tuple[int, list[dict[str, Any]], Any]:
    from treemendous import Span, create_range_set

    def new_ranges() -> Any:
        ranges = create_range_set(
            workload.domain,
            backend="cpp_boundary",
            initially_available=False,
        )
        for operation in workload.setup:
            assert operation.start is not None and operation.end is not None
            ranges.add(Span(operation.start, operation.end))
        return ranges

    def prepared_replay(
        ranges: Any,
    ) -> tuple[
        tuple[tuple[str, Callable[[Any], Any], Any], ...],
        Callable[[tuple[str, Callable[[Any], Any], Any]], Any],
    ]:
        calls: list[tuple[str, Callable[[Any], Any], Any]] = []
        for operation in workload.operations:
            assert operation.start is not None and operation.end is not None
            method = ranges.add if operation.kind == "add" else ranges.discard
            calls.append((operation.kind, method, Span(operation.start, operation.end)))

        if observe:

            def invoke_observed(call: tuple[str, Callable[[Any], Any], Any]) -> Any:
                _, method, span = call
                result = method(span)
                ranges.snapshot()
                return result

            return tuple(calls), invoke_observed

        def invoke_mutation(call: tuple[str, Callable[[Any], Any], Any]) -> Any:
            _, method, span = call
            return method(span)

        return tuple(calls), invoke_mutation

    timed = new_ranges()
    timed_calls, timed_invoke = prepared_replay(timed)
    execution_ns = _time_calls(timed_calls, timed_invoke)
    timed_state = tuple((item.start, item.end) for item in timed.intervals())

    validation = new_ranges()
    validation_calls, validation_invoke = prepared_replay(validation)
    evidence = [
        _result_evidence(call[0], validation_invoke(call)) for call in validation_calls
    ]
    validation_state = tuple((item.start, item.end) for item in validation.intervals())
    if validation_state != timed_state:
        raise AssertionError("RangeSet validation replay diverged")
    return execution_ns, evidence, timed_state


def _layer_report(workload: Any, runner: Callable[..., Any]) -> dict[str, Any]:
    execution_ns, evidence, state = runner(workload)
    expected, expected_state = _expected_evidence(workload)
    if evidence != expected or state != expected_state:
        raise AssertionError("attribution layer diverged from the finite trace oracle")
    return {
        "execution_ns": execution_ns,
        "evidence_checksum": _checksum(evidence),
        "state_checksum": _checksum(state),
        "operation_count": len(workload.operations),
        "no_op_count": sum(not item["changed"] for item in evidence),
        "touched_interval_count": sum(len(item["changed"]) for item in evidence),
        "touched_length": sum(item["changed_length"] for item in evidence),
    }


def _compiler_version() -> str:
    try:
        completed = subprocess.run(
            ["c++", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.splitlines()[0] if completed.stdout else "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_extension_provenance(source_root: Path) -> dict[str, str]:
    from treemendous.cpp import boundary

    extension = Path(boundary.__file__).resolve()
    try:
        relative = extension.relative_to(source_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"native extension was imported outside source root: {extension}"
        ) from exc
    return {
        "extension_path": relative.as_posix(),
        "extension_sha256": _file_sha256(extension),
    }


def _build_root(source_root: Path) -> dict[str, str]:
    """Force-build one source root and retain normalized compiler evidence."""
    command = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--force",
        "--verbose",
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    build_output = "\n".join((completed.stdout, completed.stderr))
    normalized = build_output.replace(
        str(Path(sys.prefix).resolve()), "<PYTHON_ENV>"
    ).replace(str(source_root.resolve()), "<ROOT>")
    compiler_lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip()
        and (
            "boundary.cpp" in line
            or "boundary_bindings.cpp" in line
            or "boundary.cpython" in line
        )
    ]
    return {
        "build_command": "${PYTHON} setup.py build_ext --inplace --force --verbose",
        "compiler_invocations": "\n".join(compiler_lines) or "unavailable",
    }


def _environment(source_root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            return "unknown"

    affinity = "unavailable"
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity = ",".join(str(cpu) for cpu in sorted(get_affinity(0)))
        except OSError:
            pass
    governor = "unavailable"
    try:
        governor = (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        pass
    flags = {
        name: os.environ.get(name, "0")
        for name in (
            "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
            "TREE_MENDOUS_LOCAL_NATIVE",
            "TREE_MENDOUS_SANITIZERS",
            "TREE_MENDOUS_GLIBCXX_DEBUG",
        )
    }
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": str(git("status", "--porcelain") != "").lower(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
        "cpu_affinity": affinity,
        "cpu_governor": governor or "unknown",
        "cxx": _compiler_version(),
        "cc": str(sysconfig.get_config_var("CC") or "unknown"),
        "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
        "build_flags": json.dumps(flags, sort_keys=True),
        **_native_extension_provenance(source_root),
    }


def _worker(source_root: Path, warmups: int, full: bool) -> dict[str, Any]:
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    workload = _trace()
    layer_runners: dict[str, Callable[..., Any]] = {
        "binding_no_result": _raw_no_result_layer,
        "binding_result": _binding_result_layer,
        "adapter": _adapter_layer,
        "rangeset_public": lambda item: _rangeset_layer(item, observe=False),
        "observed_publication": lambda item: _rangeset_layer(item, observe=True),
    }
    for _ in range(warmups):
        for runner in layer_runners.values():
            _layer_report(workload, runner)
    layers = {
        name: _layer_report(workload, runner) for name, runner in layer_runners.items()
    }

    focused: dict[str, Any] = {}
    representative: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    if full:
        from tests.performance.harness import run_validated_sample
        from tests.performance.profiles import benchmark_profile

        def validated_report(backend_id: str, item: Any) -> dict[str, Any]:
            for _ in range(warmups):
                run_validated_sample(backend_id, item)
            sample = run_validated_sample(backend_id, item)
            summary = sample.summary
            return {
                "execution_ns": sample.execution_ns,
                "state_checksum": summary.state_checksum,
                "query_checksum": summary.query_checksum,
                "operation_count": summary.requested_operations,
                "successful_operation_count": summary.successful_operations,
                "no_op_count": summary.no_op_operations,
                "touched_interval_count": summary.touched_intervals,
                "touched_length": summary.touched_length,
            }

        for focused_workload in _focused_workloads():
            focused_report = validated_report("cpp_boundary", focused_workload)
            focused_report["trace_digest"] = _trace_digest(focused_workload)
            focused[focused_workload.name] = focused_report
        for representative_workload in benchmark_profile("standard").sampled_workloads:
            representative_report = validated_report(
                "cpp_boundary", representative_workload
            )
            representative_report["trace_digest"] = _trace_digest(
                representative_workload
            )
            representative[representative_workload.name] = representative_report
        for backend_id in PYTHON_CONTROLS:
            controls[backend_id] = validated_report(backend_id, workload)

    return {
        "environment": _environment(source_root),
        "trace_digest": _trace_digest(workload),
        "layers": layers,
        "focused": focused,
        "representative": representative,
        "representative_manifest": representative_workload_manifest(),
        "controls": controls,
    }


def _run_worker_process(
    source_root: Path, *, warmups: int, full: bool
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--warmups",
        str(warmups),
    ]
    if full:
        command.append("--full")
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("worker output is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("worker output must be a JSON object")
    return decoded


def _paired_values(
    rounds: list[dict[str, dict[str, Any]]], section: str, name: str
) -> tuple[list[int], list[int]]:
    baseline: list[int] = []
    candidate: list[int] = []
    for round_result in rounds:
        left = round_result["baseline"][section][name]["execution_ns"]
        right = round_result["candidate"][section][name]["execution_ns"]
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
        ):
            raise ValueError("worker execution_ns values must be integers")
        baseline.append(left)
        candidate.append(right)
    return baseline, candidate


def _semantic_match(rounds: list[dict[str, dict[str, Any]]]) -> bool:
    reference = rounds[0]["baseline"]
    for round_result in rounds:
        baseline = round_result["baseline"]
        candidate = round_result["candidate"]
        if (
            baseline["trace_digest"] != candidate["trace_digest"]
            or baseline["trace_digest"] != reference["trace_digest"]
            or baseline["representative_manifest"]
            != candidate["representative_manifest"]
            or baseline["representative_manifest"]
            != reference["representative_manifest"]
        ):
            return False
        for section in ("layers", "focused", "representative", "controls"):
            if set(baseline[section]) != set(candidate[section]):
                return False
            for name in baseline[section]:
                left = baseline[section][name]
                right = candidate[section][name]
                for key in (
                    "evidence_checksum",
                    "trace_digest",
                    "state_checksum",
                    "query_checksum",
                    "operation_count",
                    "successful_operation_count",
                    "no_op_count",
                    "touched_interval_count",
                    "touched_length",
                ):
                    if key in left or key in right:
                        reference_value = reference[section][name].get(key)
                        if (
                            left.get(key) != right.get(key)
                            or left.get(key) != reference_value
                        ):
                            return False
    return True


def _environment_match(rounds: list[dict[str, dict[str, Any]]]) -> bool:
    ignored = {"commit", "dirty", "extension_sha256"}
    stable_per_root = (
        "commit",
        "extension_path",
        "extension_sha256",
        "build_command",
        "compiler_invocations",
        "build_flags",
        "cxx",
        "cc",
        "cflags",
    )
    references = {
        label: rounds[0][label]["environment"] for label in ("baseline", "candidate")
    }
    for round_result in rounds:
        baseline = round_result["baseline"]["environment"]
        candidate = round_result["candidate"]["environment"]
        keys = (set(baseline) | set(candidate)) - ignored
        if any(baseline.get(key) != candidate.get(key) for key in keys):
            return False
        for label in ("baseline", "candidate"):
            environment = round_result[label]["environment"]
            if any(
                environment.get(key) != references[label].get(key)
                for key in stable_per_root
            ):
                return False
    return True


def _provenance_complete(rounds: list[dict[str, dict[str, Any]]]) -> bool:
    required = (
        "build_command",
        "compiler_invocations",
        "build_flags",
        "extension_path",
        "extension_sha256",
    )
    for round_result in rounds:
        for label in ("baseline", "candidate"):
            environment = round_result[label]["environment"]
            if any(
                environment.get(key) in {None, "", "unknown", "unavailable"}
                for key in required
            ):
                return False
            if not _valid_sha256(environment["extension_sha256"]):
                return False
    return True


def compare_roots(
    baseline_root: Path,
    candidate_root: Path,
    *,
    samples: int,
    warmups: int,
    full: bool,
    primary_ratio_limit: float | None = None,
    regression_ratio_limit: float | None = None,
    control_ratio_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run alternating paired rounds with an optional explicit gate policy."""
    if samples < 20:
        raise ValueError("paired comparisons require at least 20 samples")
    if warmups < 1:
        raise ValueError("at least one warmup is required")
    policy_values = (
        primary_ratio_limit,
        regression_ratio_limit,
        control_ratio_bounds,
    )
    if any(value is None for value in policy_values) and any(
        value is not None for value in policy_values
    ):
        raise ValueError("gate policy inputs must be supplied together")
    if control_ratio_bounds is not None and (
        control_ratio_bounds[0] <= 0
        or control_ratio_bounds[1] < control_ratio_bounds[0]
    ):
        raise ValueError("control ratio bounds are invalid")
    policy_supplied = primary_ratio_limit is not None
    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    build_provenance = {"baseline": _build_root(baseline_root)}
    build_provenance["candidate"] = (
        build_provenance["baseline"]
        if candidate_root == baseline_root
        else _build_root(candidate_root)
    )
    rounds: list[dict[str, dict[str, Any]]] = []
    for index in range(samples):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        roots = {"baseline": baseline_root, "candidate": candidate_root}
        result: dict[str, dict[str, Any]] = {}
        for label in order:
            result[label] = _run_worker_process(
                roots[label], warmups=warmups, full=full
            )
            result[label]["environment"].update(build_provenance[label])
        rounds.append(result)

    layer_comparisons: dict[str, Any] = {}
    for name in rounds[0]["baseline"]["layers"]:
        baseline, candidate = _paired_values(rounds, "layers", name)
        limit = primary_ratio_limit if name == PRIMARY_LAYER else regression_ratio_limit
        layer_comparisons[name] = paired_statistics(
            baseline, candidate, ratio_limit=limit
        )

    focused_comparisons: dict[str, Any] = {}
    for name in rounds[0]["baseline"]["focused"]:
        baseline, candidate = _paired_values(rounds, "focused", name)
        focused_comparisons[name] = paired_statistics(
            baseline, candidate, ratio_limit=regression_ratio_limit
        )

    representative_comparisons: dict[str, Any] = {}
    for name in rounds[0]["baseline"]["representative"]:
        baseline, candidate = _paired_values(rounds, "representative", name)
        representative_comparisons[name] = paired_statistics(
            baseline, candidate, ratio_limit=regression_ratio_limit
        )

    representative_manifest = representative_workload_manifest()
    representative_names = {
        workload["name"] for workload in representative_manifest["workloads"]
    }
    manifest_matches = all(
        round_result[label]["representative_manifest"] == representative_manifest
        for round_result in rounds
        for label in ("baseline", "candidate")
    )
    structure_complete = set(layer_comparisons) == set(REQUIRED_LAYERS) and (
        not full
        or (
            set(focused_comparisons) == set(FOCUSED_WORKLOAD_NAMES)
            and set(representative_comparisons) == representative_names
            and set(rounds[0]["baseline"]["controls"]) == set(PYTHON_CONTROLS)
            and manifest_matches
        )
    )
    control_round_ratios: list[float] = []
    if full and structure_complete:
        for round_result in rounds:
            ratios = [
                round_result["candidate"]["controls"][backend]["execution_ns"]
                / round_result["baseline"]["controls"][backend]["execution_ns"]
                for backend in PYTHON_CONTROLS
            ]
            control_round_ratios.append(
                math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
            )
    control_median = (
        statistics.median(control_round_ratios) if control_round_ratios else 1.0
    )
    controls_valid = (
        None
        if control_ratio_bounds is None
        else bool(
            control_round_ratios
            and control_ratio_bounds[0] <= control_median <= control_ratio_bounds[1]
        )
    )

    semantics_match = _semantic_match(rounds)

    def evidence_for(worker_result: dict[str, Any]) -> dict[str, Any]:
        return {
            section: {
                name: {
                    key: value for key, value in entry.items() if key != "execution_ns"
                }
                for name, entry in worker_result[section].items()
            }
            for section in ("layers", "focused", "representative", "controls")
        }

    semantic_evidence = {
        section: {
            name: {
                label: evidence_for(rounds[0][label])[section][name]
                for label in ("baseline", "candidate")
            }
            for name in rounds[0]["baseline"][section]
        }
        for section in ("layers", "focused", "representative", "controls")
    }
    round_evidence: list[dict[str, Any]] = []
    for index, round_result in enumerate(rounds):
        round_entry: dict[str, Any] = {
            "round": index,
            "execution_order": (
                ["baseline", "candidate"]
                if index % 2 == 0
                else ["candidate", "baseline"]
            ),
        }
        for label in ("baseline", "candidate"):
            round_entry[label] = {
                "environment": round_result[label]["environment"],
                "trace_digest": round_result[label]["trace_digest"],
                "representative_manifest_digest": round_result[label][
                    "representative_manifest"
                ]["digest"],
                "timings_ns": {
                    section: {
                        name: entry["execution_ns"]
                        for name, entry in round_result[label][section].items()
                    }
                    for section in (
                        "layers",
                        "focused",
                        "representative",
                        "controls",
                    )
                },
                "semantic": evidence_for(round_result[label]),
            }
        round_evidence.append(round_entry)
    environments_match = _environment_match(rounds)
    provenance_complete = _provenance_complete(rounds)
    clean = all(
        round_result[label]["environment"]["dirty"] == "false"
        for round_result in rounds
        for label in ("baseline", "candidate")
    )
    commits_valid = all(
        _valid_commit(round_result[label]["environment"].get("commit"))
        for round_result in rounds
        for label in ("baseline", "candidate")
    )
    primary = layer_comparisons[PRIMARY_LAYER]["classification"]
    controls_failed = controls_valid is not None and not controls_valid
    controls_passed = controls_valid is not None and controls_valid
    regression_gates = [
        comparison
        for name, comparison in layer_comparisons.items()
        if name != PRIMARY_LAYER
    ]
    regression_gates.extend(focused_comparisons.values())
    regression_gates.extend(representative_comparisons.values())
    if (
        not semantics_match
        or not structure_complete
        or not environments_match
        or not provenance_complete
    ):
        status = "invalid"
    elif (
        not policy_supplied
        or not full
        or not clean
        or not commits_valid
        or not environments_match
    ):
        status = "diagnostic"
    elif (
        primary == "fail"
        or any(item["classification"] == "fail" for item in regression_gates)
        or controls_failed
    ):
        status = "fail"
    elif (
        primary == "pass"
        and all(item["classification"] == "pass" for item in regression_gates)
        and controls_passed
    ):
        status = "pass"
    else:
        status = "inconclusive"

    return {
        "schema": SCHEMA,
        "methodology": {
            "samples": samples,
            "warmups_per_process": warmups,
            "order": "baseline/candidate process order alternates by paired round",
            "trace": "canonical-local-mutation-throughput: 64 intervals, 1000 mutations, seed 50",
            "confidence": "10000-resample paired-ratio bootstrap interval for the median",
            "primary_layer": PRIMARY_LAYER,
            "gate_policy_supplied": policy_supplied,
            "primary_ratio_limit": primary_ratio_limit,
            "representative_regression_limit": regression_ratio_limit,
            "focused_workloads": FOCUSED_WORKLOAD_NAMES,
            "python_control_bounds": control_ratio_bounds,
            "full_representative_suite": full,
        },
        "baseline": rounds[0]["baseline"]["environment"],
        "candidate": rounds[0]["candidate"]["environment"],
        "trace_digest": rounds[0]["baseline"]["trace_digest"],
        "representative_workload_manifest": representative_manifest,
        "semantic_evidence": semantic_evidence,
        "round_evidence": round_evidence,
        "semantic_checksums_match": semantics_match,
        "environments_match": environments_match,
        "binary_provenance_complete": provenance_complete,
        "clean_worktrees": clean,
        "controls": {
            "round_geomean_ratios": control_round_ratios,
            "median_geomean_ratio": control_median,
            "valid": controls_valid,
        },
        "layers": layer_comparisons,
        "focused": focused_comparisons,
        "representative": representative_comparisons,
        "status": status,
        "promotion_eligible": status == "pass",
    }


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write canonical JSON, Markdown, and SHA-256 sidecar."""
    if output.suffix != ".json":
        raise ValueError("attribution output must use a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    lines = [
        "# Native mutation attribution",
        "",
        f"- Status: **{report['status']}**",
        f"- Baseline: `{report['baseline']['commit']}`",
        f"- Candidate: `{report['candidate']['commit']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
        "| Gate | Median candidate/baseline | 95% ratio CI | Result |",
        "|---|---:|---:|---:|",
    ]
    for section in ("layers", "focused", "representative"):
        for name, comparison in report[section].items():
            low, high = comparison["confidence_95_ratio"]
            lines.append(
                f"| {section}/{name} | {comparison['median_ratio']:.4f} | "
                f"{low:.4f}-{high:.4f} | {comparison['classification']} |"
            )
    markdown_text = "\n".join(lines) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    temporary_markdown = markdown.with_name(f".{markdown.name}.tmp")
    temporary_markdown.write_text(markdown_text, encoding="utf-8")
    temporary_markdown.replace(markdown)
    temporary_checksum = checksum.with_name(f".{checksum.name}.tmp")
    temporary_checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    temporary_checksum.replace(checksum)
    return output, markdown, checksum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--primary-ratio-limit", type=float)
    parser.add_argument("--regression-ratio-limit", type=float)
    parser.add_argument("--control-ratio-minimum", type=float)
    parser.add_argument("--control-ratio-maximum", type=float)
    parser.add_argument(
        "--quick", action="store_true", help="skip representative and control workloads"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--full", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        if args.source_root is None:
            raise SystemExit("--worker requires --source-root")
        print(json.dumps(_worker(args.source_root.resolve(), args.warmups, args.full)))
        return 0
    if args.baseline_root is None or args.candidate_root is None or args.output is None:
        raise SystemExit(
            "coordinator requires --baseline-root, --candidate-root, and --output"
        )
    raw_control_bounds = (args.control_ratio_minimum, args.control_ratio_maximum)
    if (raw_control_bounds[0] is None) != (raw_control_bounds[1] is None):
        raise SystemExit(
            "--control-ratio-minimum and --control-ratio-maximum are paired"
        )
    control_bounds = (
        None
        if raw_control_bounds[0] is None
        else (raw_control_bounds[0], raw_control_bounds[1])
    )
    report = compare_roots(
        args.baseline_root.resolve(),
        args.candidate_root.resolve(),
        samples=args.samples,
        warmups=args.warmups,
        full=not args.quick,
        primary_ratio_limit=args.primary_ratio_limit,
        regression_ratio_limit=args.regression_ratio_limit,
        control_ratio_bounds=control_bounds,
    )
    write_artifacts(report, args.output)
    print(
        f"attribution status={report['status']} "
        f"primary_ratio={report['layers'][PRIMARY_LAYER]['median_ratio']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
