"""Audit maintained commands and backend identifiers against code."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from tests.docs.inventory import tracked_markdown
from treemendous.applications import SCENARIO_SPECS
from treemendous.backends import CATALOG_BY_ID

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (ROOT / "README.md", *tracked_markdown(ROOT))


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


def test_benchmark_commands_use_package_module_entry_points() -> None:
    workflows = tuple(sorted((ROOT / ".github/workflows").glob("*.y*ml")))
    tracked_markdown_result = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_documents = tuple(
        ROOT / relative_path
        for relative_path in tracked_markdown_result.stdout.splitlines()
    )
    assert tracked_documents
    command_sources = (ROOT / "Justfile", *workflows, *tracked_documents)
    for source in command_sources:
        content = re.sub(r"\\\s*\n\s*", " ", source.read_text(encoding="utf-8"))
        assert "python tests/performance/" not in content, source
        assert "python scripts/verify_" not in content, source


def test_documented_application_matrix_matches_scenario_manifest() -> None:
    documented = set(
        re.findall(
            r"^\| `([^`]+)` \|",
            (ROOT / "docs/use-cases.md").read_text(),
            re.MULTILINE,
        )
    )
    expected = {spec.id for spec in SCENARIO_SPECS}
    assert documented == expected, (
        f"use-case matrix drift: missing={expected - documented}, "
        f"unknown={documented - expected}"
    )


def test_generated_scenario_status_is_current() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_scenario_catalog.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_completed_catalog_has_no_stale_zero_of_fifty_claim() -> None:
    assert len(SCENARIO_SPECS) == 50
    assert all(spec.status.value == "complete" for spec in SCENARIO_SPECS)
    text = (ROOT / "docs/use-cases.md").read_text(encoding="utf-8")
    assert re.search(r"(?<!\d)0\s*/\s*50(?!\d)", text) is None


def test_generated_status_links_every_registered_evidence_file() -> None:
    document = ROOT / "docs/use-cases.md"
    text = document.read_text(encoding="utf-8")
    _, remainder = text.split("<!-- BEGIN GENERATED SCENARIO STATUS -->", 1)
    block, _ = remainder.split("<!-- END GENERATED SCENARIO STATUS -->", 1)
    actual_targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", block)

    expected_targets: list[str] = []
    for spec in SCENARIO_SPECS:
        assert spec.engine is not None
        module_name, separator, _ = spec.engine.partition(":")
        assert separator == ":"
        expected_targets.append(f"../{module_name.replace('.', '/')}.py")
        for field_name in ("example", "oracle", "benchmark", "docs"):
            reference = getattr(spec, field_name)
            assert reference is not None
            expected_targets.append(
                reference.removeprefix("docs/")
                if reference.startswith("docs/")
                else f"../{reference}"
            )

    assert sorted(actual_targets) == sorted(expected_targets)
    for target in actual_targets:
        assert (document.parent / target).resolve().is_file(), target


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
