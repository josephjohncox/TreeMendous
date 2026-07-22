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
DESCRIPTION = (
    "Exact integer range sets with Python/C++ backends, atomic native batches, "
    "experimental multidimensional indexes, and 50 application engines."
)
REPOSITORY_URL = "https://github.com/josephjohncox/TreeMendous"
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
    (
        ROOT / "examples/patterns/atomic_port_pool_reconciliation.py",
        "free=10000-10001,10003-10005,10006-10008 changed=2,1",
    ),
    (
        ROOT / "examples/patterns/atomic_memory_map_updates.py",
        "mapped=-4096:-2048,0:4096 changed=2048,4096,1024,1024",
    ),
    (
        ROOT / "examples/patterns/atomic_partition_availability_updates.py",
        "available=0:2,8:10,12:16 changed=2,4,2",
    ),
    (
        ROOT / "examples/patterns/genomic_mask_batch_updates.py",
        "masks=100:180 changed=40,40,20,20",
    ),
    (
        ROOT / "examples/patterns/spatiotemporal_geofences.py",
        "matches=alpha,beta handles=1,2,3 process_local=True",
    ),
    (
        ROOT / "examples/patterns/warehouse_space_time_reservations.py",
        "conflicts=forklift-a,forklift-b handles=1,2,3 remaining=1,3",
    ),
    (
        ROOT / "examples/patterns/video_region_timeline_overlap.py",
        "matches=title,title-copy order=1,2 touching=later remaining=2,3",
    ),
    (
        ROOT / "examples/patterns/robot_volume_time_conflicts.py",
        "conflicts=arm-a,arm-b handles=1,2,3 moved=2 remaining=2,3",
    ),
)


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL)


def test_readme_python_blocks_execute_from_unrelated_cwd(tmp_path: Path) -> None:
    blocks = _python_blocks(README.read_text())
    assert len(blocks) == 2, "README must contain two independent API quickstarts"
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
    assert project["project"]["version"] == "1.1.1"
    assert version("treemendous") == project["project"]["version"]


def test_project_metadata_describes_and_links_the_user_surface() -> None:
    import treemendous

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["description"] == DESCRIPTION
    assert treemendous.__description__ == DESCRIPTION
    assert project["readme"] == "README.md"
    assert {
        "intervals",
        "range-set",
        "allocation",
        "scheduling",
        "spatial-index",
        "pybind11",
    } <= set(project["keywords"])
    assert project["urls"] == {
        "Homepage": "https://pypi.org/project/treemendous/",
        "Documentation": f"{REPOSITORY_URL}/blob/main/docs/README.md",
        "Repository": REPOSITORY_URL,
        "Issues": f"{REPOSITORY_URL}/issues",
        "Changelog": f"{REPOSITORY_URL}/releases",
    }


def test_readme_links_are_absolute_and_pypi_safe() -> None:
    targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", README.read_text())
    assert targets
    assert all(target.startswith("https://") for target in targets)
    repository_targets = [
        target for target in targets if target.startswith("https://github.com/")
    ]
    assert repository_targets
    assert all(target.startswith(f"{REPOSITORY_URL}/") for target in repository_targets)


def test_performance_guide_binds_durable_release_evidence() -> None:
    performance = (ROOT / "docs/performance.md").read_text()
    assert "ec91793d0dbd5152b3bb2baf4231297df92692f0" in performance
    assert (
        "af70331fad467ba3cf6eeb3f24b2733b15e05dee4ee4e1acc2f7053cdfdb78e1"
        in performance
    )
    assert "treemendous-rangeset-standard-ec91793.json" in performance
    assert "treemendous-exact-batch-1.1.0.json" in performance
    assert "treemendous-exact-batch-scaling-1.1.0.json" in performance
    assert "treemendous-scalar-attribution-1.1.0.json" in performance
    assert (
        "7b63e752b0f12c765c0e099e7229d4f3492da7fd364cc370dda0d4b9860732d9"
        in performance
    )


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
    assert workflow.count("persist-credentials: false") == 5
    assert workflow.count("enable-cache: false") == 4
    assert "baseline-ref: fdb4efd5f407717c8e18b94e6f4c21cbfb8e5daa" in workflow
    assert 'version = "1.1.1"' in (ROOT / "pyproject.toml").read_text()


def test_release_artifact_jobs_run_unsplit_strict_twine_checks_before_upload() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    artifact_check = (
        "        run: >-\n"
        "          uv run --frozen --no-sync python -m twine check --strict\n"
        "          artifacts/*"
    )
    aggregate_check = (
        "        run: >-\n"
        "          uv run --frozen --no-sync python -m twine check --strict\n"
        "          release-artifacts/*"
    )
    assert workflow.count(artifact_check) == 2
    assert workflow.count(aggregate_check) == 1
    assert workflow.count("python -m twine check --strict") == 3

    sdist = workflow.split("  sdist:\n", 1)[1].split("  wheels:\n", 1)[0]
    wheels = workflow.split("  wheels:\n", 1)[1].split("  verify-release:\n", 1)[0]
    aggregate = workflow.split("  verify-release:\n", 1)[1].split("  publish:\n", 1)[0]
    assert sdist.index(artifact_check) < sdist.index("actions/upload-artifact")
    assert wheels.index(artifact_check) < wheels.index("actions/upload-artifact")
    assert aggregate.index(aggregate_check) < aggregate.index("actions/upload-artifact")


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
