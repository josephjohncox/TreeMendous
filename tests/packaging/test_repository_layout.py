"""Monorepo ownership contracts for authored documentation and examples."""

from __future__ import annotations

import subprocess
from pathlib import Path

from setuptools._distutils.filelist import FileList

ROOT = Path(__file__).resolve().parents[2]
AUTHORING_ROOTS = (ROOT / "docs", ROOT / "examples")
LOCAL_METADATA = {ROOT / "docs/.gitrepo", ROOT / "examples/.gitrepo"}


def _check_ignore(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_nested_authored_paths_are_not_ignored() -> None:
    probes = (
        ROOT / "docs/guides/new.md",
        ROOT / "examples/tutorials/new.py",
    )
    for probe in probes:
        assert _check_ignore(probe).returncode == 1


def test_existing_authored_files_are_visible_to_git() -> None:
    for authoring_root in AUTHORING_ROOTS:
        for path in authoring_root.rglob("*"):
            if not path.is_file() or path in LOCAL_METADATA:
                continue
            if "_build" in path.parts or "__pycache__" in path.parts:
                continue
            assert _check_ignore(path).returncode == 1, path.relative_to(ROOT)


def test_legacy_subrepo_metadata_remains_local_only() -> None:
    for metadata in LOCAL_METADATA:
        if metadata.exists():
            assert _check_ignore(metadata).returncode == 0


def test_manifest_includes_authored_files_and_excludes_local_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    authored = (
        tmp_path / "docs/guides/guide.md",
        tmp_path / "examples/tutorials/example.py",
    )
    forbidden = (
        tmp_path / "docs/.context.md",
        tmp_path / "examples/.gitrepo",
        tmp_path / "examples/review-probe.so",
        tmp_path / "examples/__pycache__/helper.pyc",
    )
    for path in (*authored, *forbidden):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("probe", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    files = FileList()
    files.findall()
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        if line and not line.startswith("#"):
            files.process_template_line(line)

    included = set(files.files)
    assert {path.relative_to(tmp_path).as_posix() for path in authored} <= included
    assert not {path.relative_to(tmp_path).as_posix() for path in forbidden} & included


def test_bidirectional_subrepo_sync_is_retired() -> None:
    assert not (ROOT / "scripts/sync-repos.sh").exists()
