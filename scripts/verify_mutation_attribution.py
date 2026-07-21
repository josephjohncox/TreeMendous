#!/usr/bin/env python3
"""Strictly verify a paired native-mutation attribution artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tests.performance.mutation_attribution import (
    FOCUSED_WORKLOAD_NAMES,
    PRIMARY_LAYER,
    PYTHON_CONTROLS,
    REPRESENTATIVE_MANIFEST_SCHEMA,
    REQUIRED_LAYERS,
    SCHEMA,
)


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
        raise ValueError("attribution JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("attribution JSON root must be an object")
    return decoded


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _integer_samples(value: Any, description: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        or item > 2**63 - 1
        for item in value
    ):
        raise ValueError(
            f"{description} must contain positive signed 64-bit integer samples"
        )
    return value


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


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_representative_manifest() -> dict[str, Any]:
    """Reconstruct the exact standard workload manifest independently."""
    from tests.performance.profiles import benchmark_profile

    workloads = []
    for workload in benchmark_profile("standard").sampled_workloads:
        workloads.append(
            {
                "name": workload.name,
                "domain": workload.domain,
                "setup": [asdict(operation) for operation in workload.setup],
                "operations": [asdict(operation) for operation in workload.operations],
                "coordinate_extent": workload.coordinate_extent,
                "dimensions": list(workload.dimensions),
            }
        )
    body = json.loads(
        json.dumps(
            {
                "schema": REPRESENTATIVE_MANIFEST_SCHEMA,
                "profile": "standard",
                "workloads": workloads,
            }
        )
    )
    return {**body, "digest": _checksum(body)}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _recalculate_paired_statistics(
    baseline: list[int], candidate: list[int], ratio_limit: float | None
) -> dict[str, Any]:
    ratios = [right / left for left, right in zip(baseline, candidate, strict=True)]
    rng = random.Random(50)
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
        "baseline_samples_ns": baseline,
        "candidate_samples_ns": candidate,
        "paired_ratios": ratios,
        "median_ratio": median_ratio,
        "median_improvement": 1.0 - median_ratio,
        "confidence_95_ratio": confidence,
        "ratio_limit": ratio_limit,
        "classification": classification,
    }


def _verify_comparison(
    value: Any,
    description: str,
    ratio_limit: float | None,
    expected_baseline: list[int],
    expected_candidate: list[int],
) -> str:
    comparison = _require_mapping(value, description)
    baseline = _integer_samples(
        comparison.get("baseline_samples_ns"), f"{description} baseline"
    )
    candidate = _integer_samples(
        comparison.get("candidate_samples_ns"), f"{description} candidate"
    )
    if baseline != expected_baseline or candidate != expected_candidate:
        raise ValueError(f"{description} samples differ from per-round evidence")
    recalculated = _recalculate_paired_statistics(baseline, candidate, ratio_limit)
    for key in (
        "sample_count",
        "paired_ratios",
        "median_ratio",
        "median_improvement",
        "confidence_95_ratio",
        "ratio_limit",
        "classification",
    ):
        observed = comparison.get(key)
        expected = recalculated[key]
        if key == "confidence_95_ratio" and isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"{description} has inconsistent {key}")
    return recalculated["classification"]


def _environment_match(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ignored = {"commit", "dirty", "extension_sha256"}
    keys = (set(baseline) | set(candidate)) - ignored
    return all(baseline.get(key) == candidate.get(key) for key in keys)


_STABLE_ROOT_PROVENANCE_FIELDS = (
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

_SEMANTIC_CHECKSUM_FIELDS = {
    "layers": ("evidence_checksum", "state_checksum"),
    "focused": ("trace_digest", "state_checksum", "query_checksum"),
    "representative": ("trace_digest", "state_checksum", "query_checksum"),
    "controls": ("state_checksum", "query_checksum"),
}
_SEMANTIC_COUNTER_FIELDS = {
    "layers": (
        "operation_count",
        "no_op_count",
        "touched_interval_count",
        "touched_length",
    ),
    "focused": (
        "operation_count",
        "successful_operation_count",
        "no_op_count",
        "touched_interval_count",
        "touched_length",
    ),
    "representative": (
        "operation_count",
        "successful_operation_count",
        "no_op_count",
        "touched_interval_count",
        "touched_length",
    ),
    "controls": (
        "operation_count",
        "successful_operation_count",
        "no_op_count",
        "touched_interval_count",
        "touched_length",
    ),
}


def _semantic_entry(value: Any, section: str, description: str) -> dict[str, Any]:
    entry = _require_mapping(value, description)
    checksum_fields = _SEMANTIC_CHECKSUM_FIELDS[section]
    counter_fields = _SEMANTIC_COUNTER_FIELDS[section]
    if set(entry) != set((*checksum_fields, *counter_fields)):
        raise ValueError(f"{description} fields are incomplete or unexpected")
    if any(not _valid_sha256(entry[field]) for field in checksum_fields):
        raise ValueError(f"{description} contains an invalid checksum")
    for field in counter_fields:
        counter = entry[field]
        minimum = 1 if field == "operation_count" else 0
        if (
            isinstance(counter, bool)
            or not isinstance(counter, int)
            or counter < minimum
            or counter > 2**63 - 1
        ):
            raise ValueError(f"{description} contains an invalid {field}")
    if section != "layers" and (
        entry["successful_operation_count"] + entry["no_op_count"]
        > entry["operation_count"]
    ):
        raise ValueError(f"{description} counters are inconsistent")
    if entry["no_op_count"] > entry["operation_count"]:
        raise ValueError(f"{description} counters are inconsistent")
    return entry


def _provenance_complete(environment: dict[str, Any]) -> bool:
    required = (
        "build_command",
        "compiler_invocations",
        "build_flags",
        "extension_path",
    )
    return all(
        isinstance(environment.get(key), str)
        and environment[key] not in {"", "unknown", "unavailable"}
        for key in required
    ) and _valid_sha256(environment.get("extension_sha256"))


def verify_report(
    report: dict[str, Any],
    *,
    expected_baseline: str | None = None,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
    gate: bool = False,
    expected_primary_ratio_limit: float | None = None,
    expected_regression_ratio_limit: float | None = None,
    expected_control_ratio_bounds: tuple[float, float] | None = None,
) -> None:
    """Verify all derived fields; optionally enforce promotion eligibility."""
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected attribution schema")
    methodology = _require_mapping(report.get("methodology"), "methodology")
    samples = methodology.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 20:
        raise ValueError("attribution sample count is invalid")
    if require_samples is not None and samples != require_samples:
        raise ValueError("attribution sample count does not match the requirement")

    primary_limit = methodology.get("primary_ratio_limit")
    regression_limit = methodology.get("representative_regression_limit")
    raw_control_bounds = methodology.get("python_control_bounds")
    control_bounds: tuple[float, float] | None
    if raw_control_bounds is None:
        control_bounds = None
    elif (
        isinstance(raw_control_bounds, list | tuple)
        and len(raw_control_bounds) == 2
        and all(
            not isinstance(item, bool) and isinstance(item, int | float)
            for item in raw_control_bounds
        )
    ):
        control_bounds = (float(raw_control_bounds[0]), float(raw_control_bounds[1]))
    else:
        raise ValueError("control ratio bounds are invalid")
    policy_supplied = all(
        value is not None for value in (primary_limit, regression_limit, control_bounds)
    )
    if (
        any(
            value is not None
            for value in (primary_limit, regression_limit, control_bounds)
        )
        != policy_supplied
    ):
        raise ValueError("gate policy inputs are incomplete")
    if methodology.get("gate_policy_supplied") is not policy_supplied:
        raise ValueError("gate policy declaration is inconsistent")
    if policy_supplied and (
        isinstance(primary_limit, bool)
        or not isinstance(primary_limit, int | float)
        or primary_limit <= 0
        or isinstance(regression_limit, bool)
        or not isinstance(regression_limit, int | float)
        or regression_limit <= 0
        or control_bounds is None
        or control_bounds[0] <= 0
        or control_bounds[1] < control_bounds[0]
    ):
        raise ValueError("gate policy limits are invalid")
    expected_policy = (
        expected_primary_ratio_limit,
        expected_regression_ratio_limit,
        expected_control_ratio_bounds,
    )
    if any(item is not None for item in expected_policy):
        if not all(item is not None for item in expected_policy):
            raise ValueError("expected gate policy inputs are incomplete")
        if (
            primary_limit != expected_primary_ratio_limit
            or regression_limit != expected_regression_ratio_limit
            or control_bounds != expected_control_ratio_bounds
        ):
            raise ValueError("artifact gate policy does not match requested policy")

    declared_focused = methodology.get("focused_workloads")
    if (
        not isinstance(declared_focused, list | tuple)
        or tuple(declared_focused) != FOCUSED_WORKLOAD_NAMES
    ):
        raise ValueError("focused workload declaration is incomplete")
    full = methodology.get("full_representative_suite")
    if not isinstance(full, bool):
        raise ValueError("full representative suite declaration must be Boolean")

    expected_manifest = _expected_representative_manifest()
    manifest = report.get("representative_workload_manifest")
    if manifest != expected_manifest:
        raise ValueError("representative workload manifest is missing or inconsistent")
    expected_names = tuple(
        workload["name"] for workload in expected_manifest["workloads"]
    )
    canonical_entry = next(
        workload
        for workload in expected_manifest["workloads"]
        if workload["name"] == "canonical-local-mutation-throughput"
    )
    if report.get("trace_digest") != _checksum(canonical_entry):
        raise ValueError("trace digest is inconsistent with the workload manifest")

    rounds = report.get("round_evidence")
    if not isinstance(rounds, list) or len(rounds) != samples:
        raise ValueError("per-round evidence is incomplete")
    expected_sections = {"layers", "focused", "representative", "controls"}
    expected_section_names = {
        "layers": tuple(REQUIRED_LAYERS),
        "focused": tuple(FOCUSED_WORKLOAD_NAMES) if full else (),
        "representative": expected_names if full else (),
        "controls": tuple(PYTHON_CONTROLS) if full else (),
    }
    timings: dict[str, dict[str, dict[str, list[int]]]] = {
        section: {name: {"baseline": [], "candidate": []} for name in names}
        for section, names in expected_section_names.items()
    }
    reference_semantics: dict[str, Any] | None = None
    reference_environments: dict[str, dict[str, Any]] = {}
    environments_match = True
    provenance_complete = True
    clean = True
    commits_valid = True
    for index, raw_round in enumerate(rounds):
        round_entry = _require_mapping(raw_round, f"round {index}")
        expected_order = (
            ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        )
        if (
            round_entry.get("round") != index
            or round_entry.get("execution_order") != expected_order
        ):
            raise ValueError(f"round {index} ordering evidence is inconsistent")
        pair_semantics: dict[str, Any] = {}
        pair_environments: dict[str, dict[str, Any]] = {}
        for label in ("baseline", "candidate"):
            evidence = _require_mapping(
                round_entry.get(label), f"round {index} {label} evidence"
            )
            environment = _require_mapping(
                evidence.get("environment"), f"round {index} {label} environment"
            )
            pair_environments[label] = environment
            reference_environment = reference_environments.setdefault(
                label, environment
            )
            for field in _STABLE_ROOT_PROVENANCE_FIELDS:
                if environment.get(field) != reference_environment.get(field):
                    raise ValueError(
                        f"round {index} {label} {field} changed between rounds"
                    )
            if evidence.get("trace_digest") != report.get("trace_digest"):
                raise ValueError(f"round {index} {label} trace digest is inconsistent")
            if (
                evidence.get("representative_manifest_digest")
                != expected_manifest["digest"]
            ):
                raise ValueError(
                    f"round {index} {label} manifest digest is inconsistent"
                )
            semantic = _require_mapping(
                evidence.get("semantic"), f"round {index} {label} semantics"
            )
            timing = _require_mapping(
                evidence.get("timings_ns"), f"round {index} {label} timings"
            )
            if set(semantic) != expected_sections or set(timing) != expected_sections:
                raise ValueError(f"round {index} {label} sections are incomplete")
            for section, names in expected_section_names.items():
                section_semantic = _require_mapping(
                    semantic.get(section), f"round {index} {label}/{section} semantics"
                )
                section_timing = _require_mapping(
                    timing.get(section), f"round {index} {label}/{section} timings"
                )
                if set(section_semantic) != set(names) or set(section_timing) != set(
                    names
                ):
                    raise ValueError(
                        f"round {index} {label}/{section} evidence is incomplete"
                    )
                for name in names:
                    _semantic_entry(
                        section_semantic[name],
                        section,
                        f"round {index} {label}/{section}/{name} semantics",
                    )
                    value = section_timing[name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        or value > 2**63 - 1
                    ):
                        raise ValueError(
                            f"round {index} {label}/{section}/{name} timing is invalid"
                        )
                    timings[section][name][label].append(value)
            pair_semantics[label] = semantic
            provenance_complete &= _provenance_complete(environment)
            clean &= environment.get("dirty") == "false"
            commits_valid &= _valid_commit(environment.get("commit"))
        if pair_semantics["baseline"] != pair_semantics["candidate"]:
            raise ValueError(f"round {index} semantic evidence differs")
        if reference_semantics is None:
            reference_semantics = pair_semantics["baseline"]
        elif pair_semantics["baseline"] != reference_semantics:
            raise ValueError(f"round {index} semantic evidence changed between rounds")
        environments_match &= _environment_match(
            pair_environments["baseline"], pair_environments["candidate"]
        )

    assert reference_semantics is not None
    semantic_evidence = _require_mapping(
        report.get("semantic_evidence"), "semantic evidence"
    )
    if set(semantic_evidence) != expected_sections:
        raise ValueError("semantic evidence sections are incomplete")
    for section, names in expected_section_names.items():
        entries = _require_mapping(
            semantic_evidence.get(section), f"{section} semantic evidence"
        )
        if set(entries) != set(names):
            raise ValueError(f"{section} semantic evidence is incomplete")
        for name in names:
            pair = _require_mapping(entries[name], f"{section}/{name} evidence pair")
            if set(pair) != {"baseline", "candidate"}:
                raise ValueError(f"{section}/{name} evidence pair is incomplete")
            for label in ("baseline", "candidate"):
                _semantic_entry(
                    pair[label], section, f"{section}/{name} {label} evidence"
                )
            expected_value = reference_semantics[section][name]
            if pair != {"baseline": expected_value, "candidate": expected_value}:
                raise ValueError(f"semantic evidence differs for {section}/{name}")

    baseline = _require_mapping(report.get("baseline"), "baseline environment")
    candidate = _require_mapping(report.get("candidate"), "candidate environment")
    first_round = _require_mapping(rounds[0], "round 0")
    first_baseline = _require_mapping(first_round.get("baseline"), "round 0 baseline")
    first_candidate = _require_mapping(
        first_round.get("candidate"), "round 0 candidate"
    )
    if baseline != first_baseline.get(
        "environment"
    ) or candidate != first_candidate.get("environment"):
        raise ValueError("top-level environments differ from round evidence")
    if expected_baseline is not None and baseline.get("commit") != expected_baseline:
        raise ValueError("baseline commit does not match")
    if expected_candidate is not None and candidate.get("commit") != expected_candidate:
        raise ValueError("candidate commit does not match")

    reported_semantics = report.get("semantic_checksums_match")
    if not isinstance(reported_semantics, bool) or not reported_semantics:
        raise ValueError("semantic checksums do not match")
    for field, recalculated in (
        ("environments_match", environments_match),
        ("binary_provenance_complete", provenance_complete),
        ("clean_worktrees", clean),
    ):
        observed = report.get(field)
        if not isinstance(observed, bool) or observed != recalculated:
            raise ValueError(f"{field} is inconsistent with per-round evidence")

    classifications: list[str] = []
    for section in ("layers", "focused", "representative"):
        comparisons = _require_mapping(report.get(section), f"{section} comparisons")
        names = expected_section_names[section]
        if set(comparisons) != set(names):
            raise ValueError(f"required {section} comparisons are incomplete")
        for name in names:
            ratio_limit = (
                primary_limit
                if section == "layers" and name == PRIMARY_LAYER
                else regression_limit
            )
            values = timings[section][name]
            classifications.append(
                _verify_comparison(
                    comparisons[name],
                    f"{section} {name}",
                    ratio_limit,
                    values["baseline"],
                    values["candidate"],
                )
            )

    controls = _require_mapping(report.get("controls"), "control comparison")
    expected_control_ratios: list[float] = []
    if full:
        for index in range(samples):
            ratios = [
                timings["controls"][backend]["candidate"][index]
                / timings["controls"][backend]["baseline"][index]
                for backend in PYTHON_CONTROLS
            ]
            expected_control_ratios.append(
                math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
            )
    if controls.get("round_geomean_ratios") != expected_control_ratios:
        raise ValueError("Python control ratios are inconsistent with round evidence")
    expected_control_median = (
        statistics.median(expected_control_ratios) if expected_control_ratios else 1.0
    )
    if controls.get("median_geomean_ratio") != expected_control_median:
        raise ValueError("Python control median is inconsistent")
    expected_controls_valid = (
        None
        if control_bounds is None
        else bool(
            expected_control_ratios
            and control_bounds[0] <= expected_control_median <= control_bounds[1]
        )
    )
    if controls.get("valid") is not expected_controls_valid:
        raise ValueError("Python control validity is inconsistent")

    if not provenance_complete or not environments_match:
        expected_status = "invalid"
    elif not policy_supplied or not full or not clean or not commits_valid:
        expected_status = "diagnostic"
    elif "fail" in classifications or expected_controls_valid is False:
        expected_status = "fail"
    elif all(item == "pass" for item in classifications) and expected_controls_valid:
        expected_status = "pass"
    else:
        expected_status = "inconclusive"
    if report.get("status") != expected_status:
        raise ValueError("attribution status is inconsistent")
    promotion_eligible = report.get("promotion_eligible")
    if not isinstance(promotion_eligible, bool) or promotion_eligible != (
        expected_status == "pass"
    ):
        raise ValueError("promotion eligibility is inconsistent")
    if gate and not promotion_eligible:
        raise ValueError("attribution artifact is not promotion-eligible")
    if gate and not all(item is not None for item in expected_policy):
        raise ValueError("gate verification requires an explicit expected policy")


def verify_artifact(
    json_path: Path,
    **requirements: Any,
) -> dict[str, Any]:
    """Verify the artifact triplet and its strict report contents."""
    if json_path.suffix != ".json":
        raise ValueError("attribution artifact must use a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    checksum_path = Path(f"{json_path}.sha256")
    for path in (json_path, markdown_path, checksum_path):
        if not path.is_file():
            raise ValueError(f"attribution artifact is missing: {path}")

    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum_path.read_text(encoding="utf-8") != f"{digest}  {json_path.name}\n":
        raise ValueError("attribution checksum sidecar does not match JSON bytes")
    markdown = markdown_path.read_text(encoding="utf-8")
    if not markdown.startswith("# Native mutation attribution\n"):
        raise ValueError("attribution Markdown heading is invalid")
    if f"JSON SHA-256: `{digest}`" not in markdown:
        raise ValueError("attribution Markdown does not identify the JSON digest")

    report = _strict_json(encoded)
    verify_report(report, **requirements)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-baseline")
    parser.add_argument("--expected-candidate")
    parser.add_argument("--require-samples", type=int)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--expected-primary-ratio-limit", type=float)
    parser.add_argument("--expected-regression-ratio-limit", type=float)
    parser.add_argument("--expected-control-ratio-minimum", type=float)
    parser.add_argument("--expected-control-ratio-maximum", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_bounds = (
        args.expected_control_ratio_minimum,
        args.expected_control_ratio_maximum,
    )
    if (raw_bounds[0] is None) != (raw_bounds[1] is None):
        raise SystemExit("expected control ratio bounds must be supplied together")
    bounds = None if raw_bounds[0] is None else (raw_bounds[0], raw_bounds[1])
    report = verify_artifact(
        args.json_path,
        expected_baseline=args.expected_baseline,
        expected_candidate=args.expected_candidate,
        require_samples=args.require_samples,
        gate=args.gate,
        expected_primary_ratio_limit=args.expected_primary_ratio_limit,
        expected_regression_ratio_limit=args.expected_regression_ratio_limit,
        expected_control_ratio_bounds=bounds,
    )
    primary = report["layers"][PRIMARY_LAYER]
    print(
        f"verified attribution status={report['status']} "
        f"primary_ratio={primary['median_ratio']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
