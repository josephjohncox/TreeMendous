"""Contracts for the durable concrete-application benchmark suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.performance import application_benchmark_suite as suite
from treemendous.applications import SCENARIO_SPECS


def test_profiles_are_bounded() -> None:
    assert set(suite.PROFILES) == {"smoke", "standard"}
    with pytest.raises(ValueError, match="operations"):
        suite.ApplicationBenchmarkProfile("too-large", operations=10_001, samples=1)
    with pytest.raises(ValueError, match="samples"):
        suite.ApplicationBenchmarkProfile("too-many", operations=1, samples=21)


def test_every_complete_scenario_has_an_importable_benchmark_module_name() -> None:
    names = [suite._benchmark_module_name(spec) for spec in SCENARIO_SPECS]
    assert len(names) == len(set(names)) == 50
    assert all(name.startswith("tests.performance.applications.") for name in names)


def test_artifacts_are_canonical_checksummed_and_atomic(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "suite": "concrete-applications",
        "profile": {"name": "test", "operations": 1, "samples": 1, "seed": 42},
        "environment": {"commit": "abc"},
        "scenario_count": 1,
        "results": [
            {
                "scenario_id": "synthetic",
                "family": "partition",
                "operations": 1,
                "sample_count": 1,
                "median_execution_ns": 10,
                "evidence_checksum": "0" * 64,
            }
        ],
    }
    output, markdown, checksum = suite.write_artifacts(report, tmp_path / "run.json")
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()

    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        pytest.fail(f"artifact was not valid JSON: {exc}")
    assert decoded == report
    assert f"{digest}  run.json\n" == checksum.read_text(encoding="utf-8")
    assert digest in markdown.read_text(encoding="utf-8")
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_output_requires_json_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".json"):
        suite.write_artifacts({}, tmp_path / "run.txt")


def test_unknown_scenario_selection_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="unknown scenario ids"):
        suite.run_profile(suite.PROFILES["smoke"], scenario_ids=["not-real"])
