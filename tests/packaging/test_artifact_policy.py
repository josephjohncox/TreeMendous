from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_artifact_contents import (
    CPU_EXTENSIONS,
    METAL_EXTENSION,
    METAL_RESOURCE,
    REQUIRED_SDIST,
    ArtifactPolicyError,
    artifact_paths,
    verify_sdist,
    verify_wheel,
)

pytestmark = pytest.mark.packaging


def _sdist(path: Path, names: set[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(names):
            data = b"input"
            info = tarfile.TarInfo(f"treemendous-0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _wheel(path: Path, names: set[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(names):
            archive.writestr(name, b"input")


def _cpu_extension_names() -> set[str]:
    return {f"{stem}.fake.so" for stem in CPU_EXTENSIONS}


@pytest.mark.parametrize(
    "suffix",
    [
        ".so",
        ".pyd",
        ".dll",
        ".dylib",
        ".o",
        ".obj",
        ".a",
        ".lib",
        ".air",
        ".metallib",
    ],
)
def test_sdist_policy_rejects_generated_host_artifacts(
    tmp_path: Path, suffix: str
) -> None:
    artifact = tmp_path / "treemendous-0.tar.gz"
    leaked = f"treemendous/cpp/leaked{suffix}"
    _sdist(artifact, REQUIRED_SDIST | {leaked})
    with pytest.raises(ArtifactPolicyError, match=rf"leaked{suffix}"):
        verify_sdist(artifact)


def test_sdist_policy_requires_every_backend_rebuild_input(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0.tar.gz"
    missing = "treemendous/cpp/metal/boundary_summary_metal.metal"
    _sdist(artifact, REQUIRED_SDIST - {missing})
    with pytest.raises(ArtifactPolicyError, match="boundary_summary_metal.metal"):
        verify_sdist(artifact)


@pytest.mark.parametrize("suffix", [".o", ".obj", ".a", ".lib", ".air"])
def test_wheel_policy_rejects_generated_host_artifacts(
    tmp_path: Path, suffix: str
) -> None:
    artifact = tmp_path / "treemendous-0-cp312-macosx.whl"
    leaked = f"treemendous/cpp/leaked{suffix}"
    _wheel(artifact, _cpu_extension_names() | {leaked})
    with pytest.raises(ArtifactPolicyError, match=rf"leaked{suffix}"):
        verify_wheel(artifact)


def test_wheel_policy_requires_every_cpu_extension(tmp_path: Path) -> None:
    for index, missing in enumerate(sorted(CPU_EXTENSIONS)):
        artifact = tmp_path / f"treemendous-0-cp312-macosx-{index}.whl"
        names = _cpu_extension_names() - {f"{missing}.fake.so"}
        _wheel(artifact, names)
        with pytest.raises(ArtifactPolicyError, match=missing.rsplit("/", 1)[-1]):
            verify_wheel(artifact)


def test_metal_wheel_requires_installed_metallib(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0-cp312-macosx.whl"
    _wheel(
        artifact,
        _cpu_extension_names() | {f"{METAL_EXTENSION}.fake.so"},
    )
    with pytest.raises(ArtifactPolicyError, match="metal_resource=False"):
        verify_wheel(artifact)


def test_cpu_wheel_rejects_orphan_metallib(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0-cp312-macosx.whl"
    _wheel(artifact, _cpu_extension_names() | {METAL_RESOURCE})
    with pytest.raises(ArtifactPolicyError, match="metal_extension=False"):
        verify_wheel(artifact)


def test_metal_wheel_accepts_matched_extension_and_resource(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0-cp312-macosx.whl"
    _wheel(
        artifact,
        _cpu_extension_names() | {f"{METAL_EXTENSION}.fake.so", METAL_RESOURCE},
    )
    assert verify_wheel(artifact)["metal_resource"] == METAL_RESOURCE


def test_manylinux_gate_rejects_generic_linux_wheel(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0-cp312-cp312-linux_x86_64.whl"
    _wheel(artifact, _cpu_extension_names())
    with pytest.raises(ArtifactPolicyError, match="generic_linux=True"):
        verify_wheel(artifact, require_manylinux=True)


def test_manylinux_gate_accepts_repaired_wheel(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0-cp312-cp312-manylinux_2_17_x86_64.whl"
    _wheel(artifact, _cpu_extension_names())
    assert verify_wheel(artifact, require_manylinux=True)["kind"] == "wheel"


def test_directory_inputs_filter_hidden_non_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "treemendous-0.tar.gz"
    _sdist(artifact, REQUIRED_SDIST)
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    assert artifact_paths([tmp_path]) == [artifact]


def _run_cuda_setup(cuda_home: Path) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "setup_gpu.py", "--name"],
        cwd=repository,
        env={**os.environ, "WITH_CUDA": "1", "CUDA_HOME": str(cuda_home)},
        check=False,
        capture_output=True,
        text=True,
    )


def test_cuda_setup_reports_missing_include_directory(tmp_path: Path) -> None:
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.touch()
    result = _run_cuda_setup(tmp_path)
    assert result.returncode != 0
    assert (
        f"CUDA toolkit include directory not found: expected {tmp_path / 'include'}"
        in (result.stderr)
    )


def test_cuda_setup_reports_missing_library_directories(tmp_path: Path) -> None:
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.touch()
    (tmp_path / "include").mkdir()
    result = _run_cuda_setup(tmp_path)
    assert result.returncode != 0
    assert "CUDA toolkit library directory not found: expected one of" in result.stderr
    assert str(tmp_path / "lib") in result.stderr
