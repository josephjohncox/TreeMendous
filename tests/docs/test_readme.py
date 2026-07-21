"""Executable documentation and maintained-link contracts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from tests.docs.inventory import tracked_markdown

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
MAINTAINED_DOCS = tracked_markdown(ROOT)
RUNNABLE_EXAMPLES = (
    (ROOT / "examples/basic_rangeset.py", "allocated [9, 11)"),
    (
        ROOT / "examples/exact_batch.py",
        "changed=12,4,4,12 restored=True max_operations=4",
    ),
    (
        ROOT / "examples/multidimensional/core/linear_box_index.py",
        ("matches=2 handles=1,2 updated=primary-updated removed=secondary remaining=1"),
    ),
    (
        ROOT / "examples/multidimensional/core/fixed_box_indexes.py",
        "\n".join(
            (
                "BoxIndex2D: matches=2 handles=1,2 algorithm=axis_projection",
                "BoxIndex3D: matches=2 handles=1,2 algorithm=axis_projection",
                "BoxIndex4D: matches=2 handles=1,2 algorithm=axis_projection",
            )
        ),
    ),
    (
        ROOT / "examples/multidimensional/core/bounded_box_index.py",
        "matches=2 handles=1,2 grid=(4, 4, 4) postings=9",
    ),
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


@pytest.mark.parametrize("example,expected_output", RUNNABLE_EXAMPLES)
def test_tracked_examples_execute_from_unrelated_cwd(
    tmp_path: Path,
    example: Path,
    expected_output: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_output


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
    assert project["project"]["version"] == "1.1.0"
    assert version("treemendous") == project["project"]["version"]


def test_release_tag_contract_tracks_the_releasable_major_version() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    assert 'expected="v${version}"' in workflow
    assert 'GITHUB_REF_NAME" != "$expected' in workflow
    # v1.13.0 is an annotated tag. Pin the dereferenced commit whose matching
    # GHCR image exists, not the tag-object SHA that yields `manifest unknown`.
    assert "ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e" in workflow
    assert "106e0b0b7c337fa67ed433972f777c6357f78598" not in workflow
    assert (
        "uv run --frozen --no-sync pytest \\\n"
        "            tests/packaging/test_wheel_install.py -q"
    ) in workflow
    assert 'version = "1.1.0"' in (ROOT / "pyproject.toml").read_text()


def test_version_resolution_uses_metadata_and_source_fallback(monkeypatch) -> None:
    import treemendous

    monkeypatch.setattr(treemendous, "_metadata_version", lambda name: "9.8.7")
    assert treemendous._resolve_version() == "9.8.7"

    def missing_metadata(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(treemendous, "_metadata_version", missing_metadata)
    assert treemendous._resolve_version() == "0.0.0+dev"


def test_payloads_require_an_explicit_policy() -> None:
    from treemendous import Span, create_range_set

    ranges = create_range_set(
        domain=(0, 8),
        backend="py_boundary",
        initially_available=False,
    )
    with pytest.raises(ValueError, match="payload policy"):
        ranges.add(Span(0, 4), payload="cpu")
