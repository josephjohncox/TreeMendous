"""Hermetic Git-owned documentation inventory."""

from __future__ import annotations

import subprocess
from pathlib import Path


def tracked_markdown(root: Path) -> tuple[Path, ...]:
    """Return maintained worktree Markdown under docs and examples."""
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "docs",
            "examples",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = (
        root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item
    )
    return tuple(
        sorted(path for path in paths if path.suffix == ".md" and path.is_file())
    )
