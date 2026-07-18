"""Executable documentation and maintained-link contracts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
MAINTAINED_DOCS = (
    ROOT / "docs/getting-started.md",
    ROOT / "docs/api.md",
    ROOT / "docs/backends.md",
    ROOT / "docs/building.md",
    ROOT / "docs/benchmarking.md",
    ROOT / "docs/contributing.md",
    ROOT / "docs/releasing.md",
)


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL)


def test_readme_python_blocks_execute_from_unrelated_cwd(tmp_path: Path) -> None:
    blocks = _python_blocks(README.read_text())
    assert blocks, "README must contain an executable Python quickstart"
    for index, block in enumerate(blocks):
        completed = subprocess.run(
            [sys.executable, "-c", block],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": ""},
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"README Python block {index} failed:\n{completed.stderr}"
        )


def test_tracked_example_executes_from_unrelated_cwd(tmp_path: Path) -> None:
    example = ROOT / "examples/basic_rangeset.py"
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "allocated [9, 11)"


def test_maintained_relative_links_resolve() -> None:
    for document in (README, *MAINTAINED_DOCS):
        text = document.read_text()
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            path_text = target.split("#", 1)[0]
            linked = (document.parent / path_text).resolve()
            assert linked.exists(), f"{document.relative_to(ROOT)} -> {target}"


def test_installed_version_matches_project_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert version("treemendous") == project["project"]["version"]


def test_version_resolution_uses_metadata_and_source_fallback(monkeypatch) -> None:
    import treemendous

    monkeypatch.setattr(treemendous, "_metadata_version", lambda name: "9.8.7")
    assert treemendous._resolve_version() == "9.8.7"

    def missing_metadata(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(treemendous, "_metadata_version", missing_metadata)
    assert treemendous._resolve_version() == "0.0.0+dev"


def test_implicit_uniform_payload_policy_matches_documented_behavior() -> None:
    from treemendous import Span, create_range_set

    ranges = create_range_set(
        domain=(0, 8),
        backend="py_boundary",
        initially_available=False,
    )
    ranges.add(Span(0, 4), payload="cpu")
    ranges.add(Span(4, 8), payload="cpu")
    intervals = ranges.intervals()
    result = ranges.first_fit(8, not_before=0)
    assert result is not None
    assert len(intervals) == 1
    assert intervals[0] == result
    assert intervals[0].data == "cpu"
