"""Run reproducible benchmark profiles and write durable validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.performance.harness import (
    benchmark_backends,
    environment_metadata,
    qualify_backends,
)
from tests.performance.payload_benchmark import qualify_payload_backends
from tests.performance.profiles import BenchmarkProfile, benchmark_profile
from treemendous import BackendRegistry
from treemendous.backends.types import Available, Maturity

SCHEMA = "treemendous-validated-benchmark-suite-v2"
PROFILE_NAMES = ("smoke", "standard", "large")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="smoke")
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-all-stable",
        action="store_true",
        help="fail instead of omitting an unavailable stable CPU backend",
    )
    return parser


def _stable_backends(
    requested: list[str] | None, *, require_all: bool
) -> tuple[str, ...]:
    registry = BackendRegistry.discover()
    stable = tuple(spec for spec in registry.specs if spec.maturity is Maturity.STABLE)
    available = tuple(
        spec.id for spec in stable if isinstance(registry.states[spec.id], Available)
    )
    unavailable = tuple(spec.id for spec in stable if spec.id not in available)
    if require_all and unavailable:
        reasons = ", ".join(
            f"{backend_id}: {registry.states[backend_id]}" for backend_id in unavailable
        )
        raise SystemExit(f"stable backends unavailable: {reasons}")
    if requested is None:
        selected = available
    else:
        unknown = tuple(
            backend_id for backend_id in requested if backend_id not in available
        )
        if unknown:
            raise SystemExit(
                "requested backends are unavailable or not stable: "
                + ", ".join(unknown)
            )
        selected = tuple(dict.fromkeys(requested))
    if not selected:
        raise SystemExit("no semantically validated stable backends are available")
    return selected


def _ci_provenance() -> dict[str, str]:
    names = (
        "GITHUB_ACTIONS",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_REF",
        "GITHUB_SHA",
    )
    return {name.lower(): os.environ[name] for name in names if name in os.environ}


def run_profile(
    profile: BenchmarkProfile, backend_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Run sampled measurements and separate large-scale qualifications."""
    sampled: list[dict[str, Any]] = []
    for workload in profile.sampled_workloads:
        print(
            f"sampled: {workload.name} "
            f"({len(workload.operations):,} operations, {len(backend_ids)} backends)",
            flush=True,
        )
        sampled.append(
            benchmark_backends(
                backend_ids,
                workload,
                samples=profile.samples,
                warmups=profile.warmups,
                processes=profile.processes,
            )
        )

    qualifications: list[dict[str, Any]] = []
    for workload in profile.qualification_workloads:
        print(
            f"qualify: {workload.name} "
            f"({len(workload.operations):,} operations, {len(backend_ids)} backends)",
            flush=True,
        )
        qualifications.append(qualify_backends(backend_ids, workload))

    print(
        f"qualify: payload policies ({profile.payload_operations:,} operations, "
        f"{len(backend_ids)} backends)",
        flush=True,
    )
    payload_reports = qualify_payload_backends(
        backend_ids,
        scale=profile.payload_scale,
        operations=profile.payload_operations,
        seed=71,
    )

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": {
            "name": profile.name,
            "description": profile.description,
            "samples": profile.samples,
            "warmups": profile.warmups,
            "processes": profile.processes,
            "payload_scale": profile.payload_scale,
            "payload_operations": profile.payload_operations,
        },
        "backends": list(backend_ids),
        "environment": environment_metadata(),
        "ci_provenance": _ci_provenance(),
        "interpretation": (
            "sampled reports are local directional measurements with complete "
            "oracle validation; qualification reports are single-run scale gates "
            "and must not be used to rank backends"
        ),
        "sampled_reports": sampled,
        "qualification_reports": qualifications,
        "payload_reports": payload_reports,
    }


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        f"# Tree-Mendous benchmark: {report['profile']['name']}",
        "",
        f"- Commit: `{report['environment']['commit']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Backends: {', '.join(report['backends'])}",
        f"- JSON SHA-256: `{digest}`",
        "",
        "> These are correctness-checked local measurements, not universal speed claims.",
        "",
        "## Sampled measurements",
        "",
        "| Workload | Backend | Initial intervals | Operations | Median | 95% median CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["sampled_reports"]:
        for backend_id, result in item["results"].items():
            timing = result["execution"]
            low, high = timing["confidence_95_ns"]
            lines.append(
                f"| {item['workload']} | {backend_id} | "
                f"{item['dataset']['actual_interval_count']:,} | "
                f"{item['dataset']['timed_operations']:,} | "
                f"{timing['median_ns'] / 1e6:.3f} ms | "
                f"{low / 1e6:.3f}–{high / 1e6:.3f} ms |"
            )
    lines.extend(
        [
            "",
            "## Large-scale qualification",
            "",
            "| Workload | Backend | Initial intervals | Operations | Elapsed | Ops/s | State |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["qualification_reports"]:
        state = item["validation"]["state_checksum"][:12]
        for backend_id, result in item["results"].items():
            rate = result["operations_per_second"]
            lines.append(
                f"| {item['workload']} | {backend_id} | "
                f"{item['dataset']['actual_interval_count']:,} | "
                f"{item['dataset']['timed_operations']:,} | "
                f"{result['execution_ns'] / 1e9:.3f} s | "
                f"{rate:,.0f} | `{state}` |"
            )
    lines.extend(
        [
            "",
            "## Payload-policy qualification",
            "",
            "| Workload | Backend | Setup operations | Operations | Elapsed | Ops/s | State |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["payload_reports"]:
        state = item["validation"]["state_checksum"][:12]
        for backend_id, result in item["results"].items():
            lines.append(
                f"| {item['workload']} | {backend_id} | "
                f"{item['dataset']['setup_operations']:,} | "
                f"{item['dataset']['timed_operations']:,} | "
                f"{result['execution_ns'] / 1e9:.3f} s | "
                f"{result['operations_per_second']:,.0f} | `{state}` |"
            )
    lines.extend(
        [
            "",
            "Every geometry row was accepted only after complete state, query, mutation-accounting,",
            "snapshot, statistics, and overlap observations matched the independent oracle.",
            "Payload rows matched across every stable geometry backend; policy laws are",
            "independently exercised by the property-based payload suite.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write JSON, Markdown, and a checksum sidecar."""
    if output.suffix != ".json":
        raise ValueError("benchmark output must use a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    files = (
        (output, encoded),
        (markdown, _markdown(report, digest).encode()),
        (checksum, f"{digest}  {output.name}\n".encode()),
    )
    for path, content in files:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return output, markdown, checksum


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = benchmark_profile(args.profile)
    backend_ids = _stable_backends(args.backends, require_all=args.require_all_stable)
    report = run_profile(profile, backend_ids)
    output = args.output or Path("build/benchmarks") / f"{profile.name}.json"
    artifacts = write_artifacts(report, output)
    print("wrote:")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
