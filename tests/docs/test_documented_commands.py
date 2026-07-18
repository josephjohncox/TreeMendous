"""Audit maintained commands and backend identifiers against code."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from treemendous.backends import CATALOG_BY_ID

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs/getting-started.md",
    ROOT / "docs/api.md",
    ROOT / "docs/backends.md",
    ROOT / "docs/building.md",
    ROOT / "docs/benchmarking.md",
    ROOT / "docs/contributing.md",
    ROOT / "docs/releasing.md",
)


def _just_recipes() -> set[str]:
    if shutil.which("just") is None:
        return set(
            re.findall(
                r"^([a-zA-Z0-9_-]+)(?:\s+[^:\n]+)?:",
                (ROOT / "Justfile").read_text(),
                re.MULTILINE,
            )
        )
    completed = subprocess.run(
        ["just", "--list", "--unsorted"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.match(r"\s{4}([a-zA-Z0-9_-]+)(?:\s|$)", line))
    }


def test_every_documented_just_command_exists() -> None:
    recipes = _just_recipes()
    documented: set[str] = set()
    for document in DOCUMENTS:
        documented.update(
            re.findall(r"\bjust\s+([a-zA-Z0-9_-]+)", document.read_text())
        )
    assert documented
    assert documented - recipes == set()


def test_documented_backend_tables_match_catalog() -> None:
    expected = set(CATALOG_BY_ID)
    for document in (ROOT / "README.md", ROOT / "docs/backends.md"):
        documented = set(
            re.findall(r"^\| `([^`]+)` \|", document.read_text(), re.MULTILINE)
        )
        assert documented == expected, (
            f"backend table drift in {document.relative_to(ROOT)}: "
            f"missing={expected - documented}, unknown={documented - expected}"
        )
