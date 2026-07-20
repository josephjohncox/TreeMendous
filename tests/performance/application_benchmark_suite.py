"""Durable correctness-checked benchmarks for the 50 concrete applications."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.performance.applications.harness import ApplicationSample
from tests.performance.harness import environment_metadata
from treemendous.applications import (
    SCENARIO_SPECS,
    ScenarioSpec,
    ScenarioStatus,
    validate_catalog_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "benchmarks" / "applications-smoke.json"


@dataclass(frozen=True)
class ApplicationBenchmarkProfile:
    """Bounded execution scale for every concrete scenario."""

    name: str
    operations: int
    samples: int
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must be nonempty")
        if not 0 < self.operations <= 10_000:
            raise ValueError("profile operations must be in [1, 10000]")
        if not 0 < self.samples <= 20:
            raise ValueError("profile samples must be in [1, 20]")


PROFILES = {
    "smoke": ApplicationBenchmarkProfile("smoke", operations=8, samples=1),
    "standard": ApplicationBenchmarkProfile("standard", operations=64, samples=5),
}


def _benchmark_module_name(spec: ScenarioSpec) -> str:
    if spec.benchmark is None:
        raise ValueError(f"scenario has no benchmark evidence: {spec.id}")
    path = Path(spec.benchmark)
    if path.suffix != ".py":
        raise ValueError(f"benchmark evidence is not a Python module: {spec.id}")
    return ".".join(path.with_suffix("").parts)


def _run_scenario(
    spec: ScenarioSpec,
    profile: ApplicationBenchmarkProfile,
) -> dict[str, Any]:
    module = importlib.import_module(_benchmark_module_name(spec))
    run_benchmark = getattr(module, "run_benchmark", None)
    if not callable(run_benchmark):
        raise ValueError(f"benchmark lacks run_benchmark(): {spec.id}")

    samples: list[ApplicationSample] = []
    for _ in range(profile.samples):
        sample = run_benchmark(operations=profile.operations, seed=profile.seed)
        if not isinstance(sample, ApplicationSample):
            raise TypeError(f"benchmark returned invalid sample: {spec.id}")
        if sample.scenario_id != spec.id:
            raise ValueError(f"benchmark reported wrong scenario id: {spec.id}")
        if sample.operations != profile.operations:
            raise ValueError(f"benchmark reported wrong operation count: {spec.id}")
        if not sample.validated:
            raise ValueError(f"benchmark sample is not validated: {spec.id}")
        samples.append(sample)

    checksums = {
        (
            sample.result_checksum,
            sample.state_checksum,
            sample.counters_checksum,
            sample.evidence_checksum,
        )
        for sample in samples
    }
    if len(checksums) != 1:
        raise AssertionError(f"benchmark evidence is not reproducible: {spec.id}")
    result_checksum, state_checksum, counters_checksum, evidence_checksum = next(
        iter(checksums)
    )
    execution_ns = [sample.execution_ns for sample in samples]
    return {
        "scenario_id": spec.id,
        "family": spec.family,
        "engine": spec.engine,
        "operations": profile.operations,
        "sample_count": profile.samples,
        "seed": profile.seed,
        "execution_ns": execution_ns,
        "median_execution_ns": statistics.median(execution_ns),
        "min_execution_ns": min(execution_ns),
        "max_execution_ns": max(execution_ns),
        "result_checksum": result_checksum,
        "state_checksum": state_checksum,
        "counters_checksum": counters_checksum,
        "evidence_checksum": evidence_checksum,
        "validated": True,
    }


def run_profile(
    profile: ApplicationBenchmarkProfile,
    *,
    scenario_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run selected real engines and return bounded reproducibility evidence."""
    validate_catalog_evidence(SCENARIO_SPECS, root=PROJECT_ROOT)
    selected = set(scenario_ids) if scenario_ids is not None else None
    unknown = selected - {spec.id for spec in SCENARIO_SPECS} if selected else set()
    if unknown:
        raise ValueError(f"unknown scenario ids: {sorted(unknown)!r}")
    specs = [
        spec
        for spec in SCENARIO_SPECS
        if spec.status is ScenarioStatus.COMPLETE
        and (selected is None or spec.id in selected)
    ]
    if not specs:
        raise ValueError("application benchmark selection is empty")
    return {
        "schema_version": 1,
        "suite": "concrete-applications",
        "profile": {
            "name": profile.name,
            "operations": profile.operations,
            "samples": profile.samples,
            "seed": profile.seed,
        },
        "environment": environment_metadata(),
        "scenario_count": len(specs),
        "results": [_run_scenario(spec, profile) for spec in specs],
    }


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Concrete application benchmark evidence",
        "",
        f"- Profile: `{report['profile']['name']}`",
        f"- Scenarios: {report['scenario_count']}",
        f"- Commit: `{report['environment']['commit']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
        "All timings cover application execution only. Setup, observation,",
        "canonicalization, checksums, and independent oracle work are excluded.",
        "Every row attests the exact instance timed by that sample.",
        "",
        "| Scenario | Family | Operations | Samples | Median ns | Evidence SHA-256 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"]:
        lines.append(
            f"| `{result['scenario_id']}` | `{result['family']}` | "
            f"{result['operations']} | {result['sample_count']} | "
            f"{result['median_execution_ns']} | `{result['evidence_checksum']}` |"
        )
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write canonical JSON, Markdown, and SHA-256 evidence."""
    if output.suffix != ".json":
        raise ValueError("application benchmark output must use .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    for path, content in (
        (output, encoded),
        (markdown, _markdown(report, digest).encode()),
        (checksum, f"{digest}  {output.name}\n".encode()),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return output, markdown, checksum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a bounded profile and persist its attestation artifacts."""
    args = _parser().parse_args(argv)
    report = run_profile(PROFILES[args.profile], scenario_ids=args.scenarios)
    output, markdown, checksum = write_artifacts(report, args.output)
    print(output)
    print(markdown)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
