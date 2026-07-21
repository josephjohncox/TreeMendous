"""Clean-install smoke for the freshly built release source distribution."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from scripts.verify_artifact_contents import verify_sdist

pytestmark = pytest.mark.packaging


def _sdist_from_environment() -> Path:
    configured = os.environ.get("TREEMENDOUS_SDIST")
    if not configured:
        pytest.skip("set TREEMENDOUS_SDIST to run clean-install sdist smoke")
    sdist = Path(configured).resolve()
    if not sdist.is_file():
        pytest.fail(f"TREEMENDOUS_SDIST does not exist: {sdist}")
    return sdist


def _venv_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts/python.exe"
    return environment / "bin/python"


def test_sdist_clean_install_and_arbitrary_cwd(tmp_path: Path) -> None:
    sdist = _sdist_from_environment()
    verify_sdist(sdist)
    environment = tmp_path / "venv"
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    clean_environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    subprocess.run(
        [str(python), "-m", "pip", "install", str(sdist)],
        cwd=unrelated,
        env=clean_environment,
        check=True,
    )
    code = """
from treemendous import Span, create_range_set
from treemendous.exact_batch import BatchMutation, ExactBatchRangeSet, MutationOpcode

ranges = create_range_set((0, 32), backend='cpp_boundary')
allocated = ranges.allocate(4, not_before=2)
assert allocated is not None and allocated.span == Span(2, 6)

exact = ExactBatchRangeSet((0, 32), initially_available=False)
before = exact.snapshot()
results = exact.mutate([
    BatchMutation(MutationOpcode.ADD, 8, 16),
    BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 8, 16),
])
assert [result.changed_length for result in results] == [8, 8]
assert exact.snapshot() == before
"""
    subprocess.run(
        [str(python), "-c", code],
        cwd=unrelated,
        env=clean_environment,
        check=True,
    )
