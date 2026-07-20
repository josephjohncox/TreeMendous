#!/usr/bin/env python3
"""Deterministically enforce Tree-Mendous sdist and wheel content policy."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

NATIVE_SUFFIXES = {".so", ".pyd", ".dll", ".dylib"}
GENERATED_SUFFIXES = {
    *NATIVE_SUFFIXES,
    ".pyc",
    ".pyo",
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".air",
    ".metallib",
}
WHEEL_FORBIDDEN_GENERATED_SUFFIXES = {".o", ".obj", ".a", ".lib", ".air"}
SOURCE_SUFFIXES = {".cpp", ".h", ".cu", ".mm", ".metal"}
LOCAL_METADATA_NAMES = {".context.md", ".gitrepo"}
REQUIRED_SDIST = {
    "MANIFEST.in",
    "pyproject.toml",
    "setup.py",
    "setup_gpu.py",
    "setup_metal.py",
    "docs/README.md",
    "docs/theory/box_index_denotation.md",
    "examples/README.md",
    "examples/basic_rangeset.py",
    "examples/multidimensional/core/linear_box_index.py",
    "treemendous/cpp/interval_types.h",
    "treemendous/cpp/boundary_bindings.cpp",
    "treemendous/cpp/gpu/boundary_summary_gpu.cu",
    "treemendous/cpp/gpu/boundary_summary_gpu_bindings.cpp",
    "treemendous/cpp/metal/boundary_summary_metal.h",
    "treemendous/cpp/metal/boundary_summary_metal.mm",
    "treemendous/cpp/metal/boundary_summary_metal_bindings.cpp",
    "treemendous/cpp/metal/boundary_summary_metal.metal",
}
CPU_EXTENSIONS = {
    "treemendous/cpp/boundary",
    "treemendous/cpp/treap",
    "treemendous/cpp/boundary_summary",
    "treemendous/cpp/boundary_summary_optimized",
}
METAL_EXTENSION = "treemendous/cpp/metal/boundary_summary_metal"
METAL_RESOURCE = "treemendous/cpp/metal/resources/boundary_summary_metal.metallib"


class ArtifactPolicyError(RuntimeError):
    """Raised when an artifact violates the release content contract."""


def _normalized_sdist_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:*") as archive:
        names: set[str] = set()
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) > 1:
                names.add(PurePosixPath(*parts[1:]).as_posix())
        return names


def _wheel_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {PurePosixPath(name).as_posix() for name in archive.namelist()}


def _suffix(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def _is_artifact(path: Path) -> bool:
    return path.suffix == ".whl" or path.name.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    )


def artifact_paths(inputs: list[Path]) -> list[Path]:
    """Expand directories using artifact-only filtering, ignoring hidden metadata."""
    artifacts: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            artifacts.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and _is_artifact(candidate)
            )
        elif _is_artifact(path):
            artifacts.add(path)
        else:
            raise ArtifactPolicyError(f"unsupported artifact type: {path}")
    if not artifacts:
        raise ArtifactPolicyError("no sdist or wheel artifacts found")
    return sorted(artifacts)


def verify_sdist(path: Path) -> dict[str, object]:
    names = _normalized_sdist_names(path)
    generated = sorted(
        name
        for name in names
        if _suffix(name) in GENERATED_SUFFIXES
        or PurePosixPath(name).name in LOCAL_METADATA_NAMES
        or "__pycache__" in PurePosixPath(name).parts
    )
    missing = sorted(REQUIRED_SDIST - names)
    if generated or missing:
        raise ArtifactPolicyError(
            f"sdist policy failure: generated={generated!r}; missing={missing!r}"
        )
    source_inventory = sorted(
        name for name in names if _suffix(name) in SOURCE_SUFFIXES
    )
    return {
        "artifact": str(path),
        "kind": "sdist",
        "file_count": len(names),
        "generated_host_files": generated,
        "required_sources": source_inventory,
    }


def _contains_extension(names: set[str], stem: str) -> bool:
    return any(
        name.startswith(stem + ".") and _suffix(name) in {".so", ".pyd"}
        for name in names
    )


def verify_wheel(path: Path, *, require_manylinux: bool = False) -> dict[str, object]:
    names = _wheel_names(path)
    leaked_sources = sorted(name for name in names if _suffix(name) in SOURCE_SUFFIXES)
    generated = sorted(
        name for name in names if _suffix(name) in WHEEL_FORBIDDEN_GENERATED_SUFFIXES
    )
    missing_cpu = sorted(
        stem for stem in CPU_EXTENSIONS if not _contains_extension(names, stem)
    )
    metal_present = _contains_extension(names, METAL_EXTENSION)
    resource_present = METAL_RESOURCE in names
    metal_mismatch = metal_present != resource_present
    generic_linux = "-linux_" in path.name
    missing_manylinux = require_manylinux and "manylinux" not in path.name
    if (
        leaked_sources
        or generated
        or missing_cpu
        or metal_mismatch
        or (require_manylinux and generic_linux)
        or missing_manylinux
    ):
        raise ArtifactPolicyError(
            "wheel policy failure: "
            f"sources={leaked_sources!r}; generated={generated!r}; "
            f"cpu_extensions={missing_cpu!r}; metal_extension={metal_present!r}; "
            f"metal_resource={resource_present!r}; generic_linux={generic_linux!r}; "
            f"manylinux_required={require_manylinux!r}"
        )
    native = sorted(name for name in names if _suffix(name) in NATIVE_SUFFIXES)
    return {
        "artifact": str(path),
        "kind": "wheel",
        "file_count": len(names),
        "native_files": native,
        "generated_host_files": generated,
        "metal_resource": METAL_RESOURCE if metal_present else None,
        "source_files": leaked_sources,
    }


def verify(path: Path, *, require_manylinux: bool = False) -> dict[str, object]:
    if path.suffix == ".whl":
        return verify_wheel(path, require_manylinux=require_manylinux)
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        if require_manylinux:
            raise ArtifactPolicyError("--require-manylinux accepts wheels only")
        return verify_sdist(path)
    raise ArtifactPolicyError(f"unsupported artifact type: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-manylinux",
        action="store_true",
        help="require every supplied wheel to carry a repaired manylinux tag",
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="artifact files or directories (directories are artifact-filtered recursively)",
    )
    args = parser.parse_args()
    reports = [
        verify(path, require_manylinux=args.require_manylinux)
        for path in artifact_paths(args.artifacts)
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
