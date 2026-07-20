"""Executable checks for the bounded multidimensional benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.performance.bounded_box_index_benchmark import run_benchmark


@pytest.mark.benchmark
def test_each_benchmark_trial_is_correctness_checked() -> None:
    report = run_benchmark(3)
    assert report["trials"] == 3
    assert not report["universal_ranking_claim"]
    assert len(report["duration_ns"]) == 3
    assert "independent finite point-set oracle" in report["correctness"]


@pytest.mark.benchmark
def test_benchmark_cli_writes_machine_readable_artifact(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "bounded.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.bounded_box_index_benchmark",
            "--trials",
            "2",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    try:
        report = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"benchmark artifact could not be read: {exc}")
    assert report["algorithm"] == "sparse_grid"
    assert not report["universal_ranking_claim"]
