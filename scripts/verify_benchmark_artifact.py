#!/usr/bin/env python3
"""Verify one published generic benchmark artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.performance.harness import timing_statistics

SCHEMA = "treemendous-validated-benchmark-suite-v4"
STABLE_BACKENDS = (
    "py_boundary",
    "py_avl_earliest",
    "py_summary",
    "py_treap",
    "py_boundary_summary",
    "cpp_boundary",
)


@dataclass(frozen=True)
class ArtifactVerification:
    """Verified benchmark bundle metadata."""

    json_path: Path
    markdown_path: Path
    checksum_path: Path
    digest: str
    workloads: tuple[str, ...]


def _strict_json(encoded: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not permitted: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is not permitted: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            encoded,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("benchmark JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("benchmark JSON root must be an object")
    return decoded


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _verify_timing_samples(value: Any, description: str, expected_count: int) -> None:
    timing = _require_mapping(value, description)
    samples = timing.get("samples_ns")
    if not isinstance(samples, list):
        raise ValueError(f"{description} samples_ns must be an array")
    if len(samples) != expected_count:
        raise ValueError(f"{description} sample count is inconsistent")
    if any(
        isinstance(sample, bool)
        or not isinstance(sample, int)
        or sample <= 0
        or sample > 2**63 - 1
        for sample in samples
    ):
        raise ValueError(
            f"{description} samples must be positive signed 64-bit integers"
        )
    expected = timing_statistics(samples)
    for key, expected_value in (
        ("independent_runs", expected.independent_runs),
        ("median_ns", expected.median_ns),
        ("median_absolute_deviation_ns", expected.median_absolute_deviation_ns),
        ("confidence_95_ns", expected.confidence_95_ns),
        ("p10_ns", expected.p10_ns),
        ("p90_ns", expected.p90_ns),
    ):
        observed = timing.get(key)
        if key == "confidence_95_ns" and isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected_value:
            raise ValueError(f"{description} has inconsistent {key}")


def verify_artifact(
    json_path: Path,
    *,
    expected_profile: str | None = None,
    expected_section: str | None = None,
    expected_commit: str | None = None,
    required_workloads: tuple[str, ...] = (),
    require_all_stable: bool = False,
) -> ArtifactVerification:
    """Verify checksums, metadata, and required sampled workloads."""
    if json_path.suffix != ".json":
        raise ValueError("benchmark artifact must use a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    checksum_path = Path(f"{json_path}.sha256")
    for path in (json_path, markdown_path, checksum_path):
        if not path.is_file():
            raise ValueError(f"benchmark artifact is missing: {path}")

    temporary = tuple(json_path.parent.glob(f".{json_path.stem}*.tmp"))
    if temporary:
        raise ValueError(f"temporary benchmark artifacts remain: {temporary}")

    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    expected_sidecar = f"{digest}  {json_path.name}\n"
    if checksum_path.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError("benchmark checksum sidecar does not match JSON bytes")

    report = _strict_json(encoded)
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unexpected benchmark schema: {report.get('schema')!r}")
    profile = report.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("benchmark profile metadata is missing")
    if expected_profile is not None and profile.get("name") != expected_profile:
        raise ValueError("benchmark profile does not match the requested profile")
    if expected_section is not None and profile.get("section") != expected_section:
        raise ValueError("benchmark section does not match the requested section")

    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("benchmark environment metadata is missing")
    if expected_commit is not None and environment.get("commit") != expected_commit:
        raise ValueError("benchmark commit does not match the checked-out commit")
    provenance = report.get("ci_provenance", {})
    if expected_commit is not None and isinstance(provenance, dict):
        github_sha = provenance.get("github_sha")
        if github_sha is not None and github_sha != expected_commit:
            raise ValueError(
                "benchmark GitHub SHA does not match the checked-out commit"
            )

    backends = report.get("backends")
    if require_all_stable and tuple(backends or ()) != STABLE_BACKENDS:
        raise ValueError("benchmark does not contain the exact stable backend set")

    sampled_reports = report.get("sampled_reports")
    if not isinstance(sampled_reports, list):
        raise ValueError("sampled benchmark reports are missing")
    workload_names: list[str] = []
    for index, raw_item in enumerate(sampled_reports):
        item = _require_mapping(raw_item, f"sampled report {index}")
        workload = item.get("workload")
        if not isinstance(workload, str) or not workload:
            raise ValueError(f"sampled report {index} workload is invalid")
        workload_names.append(workload)
        methodology = _require_mapping(
            item.get("methodology"), f"sampled report {workload} methodology"
        )
        independent_runs = methodology.get("independent_runs")
        if (
            isinstance(independent_runs, bool)
            or not isinstance(independent_runs, int)
            or independent_runs < 1
        ):
            raise ValueError(f"sampled report {workload} independent_runs is invalid")
        results = _require_mapping(
            item.get("results"), f"sampled report {workload} results"
        )
        if not results:
            raise ValueError(f"sampled report {workload} results are empty")
        for backend_id, raw_result in results.items():
            result = _require_mapping(
                raw_result, f"sampled report {workload}/{backend_id}"
            )
            for timing_name in ("execution", "setup", "validation_overhead"):
                _verify_timing_samples(
                    result.get(timing_name),
                    f"sampled report {workload}/{backend_id}/{timing_name}",
                    independent_runs,
                )
            operation_latency = _require_mapping(
                result.get("operation_latency"),
                f"sampled report {workload}/{backend_id}/operation_latency",
            )
            if not operation_latency:
                raise ValueError(
                    f"sampled report {workload}/{backend_id} operation latency is empty"
                )
            for operation, raw_latency in operation_latency.items():
                latency = _require_mapping(
                    raw_latency,
                    f"sampled report {workload}/{backend_id}/{operation} latency",
                )
                _verify_timing_samples(
                    latency.get("per_run_median"),
                    (
                        f"sampled report {workload}/{backend_id}/{operation} "
                        "per-run median"
                    ),
                    independent_runs,
                )
    workloads = tuple(workload_names)
    missing = tuple(name for name in required_workloads if name not in workloads)
    if missing:
        raise ValueError(f"required sampled workloads are missing: {missing}")

    markdown = markdown_path.read_text(encoding="utf-8")
    heading = (
        f"# Tree-Mendous benchmark: {profile.get('name')} / {profile.get('section')}"
    )
    if not markdown.startswith(heading) or f"JSON SHA-256: `{digest}`" not in markdown:
        raise ValueError("benchmark Markdown does not identify the JSON artifact")

    return ArtifactVerification(
        json_path=json_path,
        markdown_path=markdown_path,
        checksum_path=checksum_path,
        digest=digest,
        workloads=workloads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-section")
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-workload", action="append", default=[])
    parser.add_argument("--require-all-stable", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_artifact(
        args.json_path,
        expected_profile=args.expected_profile,
        expected_section=args.expected_section,
        expected_commit=args.expected_commit,
        required_workloads=tuple(args.require_workload),
        require_all_stable=args.require_all_stable,
    )
    print(
        f"verified {result.json_path} sha256={result.digest} "
        f"sampled_workloads={len(result.workloads)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
