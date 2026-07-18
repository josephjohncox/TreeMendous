#!/usr/bin/env python
"""Build portable CPU extensions plus the experimental CUDA backend.

CUDA build/import does not imply runtime support. The production catalog keeps
this backend experimental and capability-empty until a hardware parity and
compute-sanitizer lane passes.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.errors import CompileError

from setup import PortableBuildExt, make_cpu_extensions

if os.environ.get("WITH_CUDA") != "1":
    raise SystemExit("CUDA build not requested; set WITH_CUDA=1")

cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
nvcc = cuda_home / "bin/nvcc"
if not nvcc.is_file():
    raise SystemExit(f"CUDA toolkit not found: expected nvcc at {nvcc}")

cuda_include = cuda_home / "include"
if not cuda_include.is_dir():
    raise SystemExit(
        f"CUDA toolkit include directory not found: expected {cuda_include}"
    )

primary_cuda_lib = cuda_home / (
    "lib/x64" if platform.system() == "Windows" else "lib64"
)
fallback_cuda_lib = cuda_home / "lib"
cuda_lib = primary_cuda_lib if primary_cuda_lib.is_dir() else fallback_cuda_lib
if not cuda_lib.is_dir():
    raise SystemExit(
        "CUDA toolkit library directory not found: expected one of "
        f"{primary_cuda_lib}, {fallback_cuda_lib}"
    )

cuda_arch = os.environ.get("CUDA_ARCH", "sm_75")


class CUDAExtension(Extension):
    """Marker extension compiled by NVCC before the platform linker runs."""


class BuildExtension(PortableBuildExt):
    """Compile exactly one CUDA/pybind translation unit, then link it once."""

    def build_extensions(self) -> None:
        for extension in self.extensions:
            if isinstance(extension, CUDAExtension):
                self._compile_cuda(extension)
        super().build_extensions()

    def _compile_cuda(self, extension: CUDAExtension) -> None:
        if len(extension.sources) != 1 or not extension.sources[0].endswith(".cu"):
            raise CompileError("CUDA extension must contain exactly one .cu source")
        source = extension.sources[0]
        output_dir = Path(self.build_temp) / "cuda"
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".obj" if platform.system() == "Windows" else ".o"
        output = output_dir / f"{Path(source).stem}{suffix}"
        host_flags = "/MD" if platform.system() == "Windows" else "-fPIC"
        command = [
            str(nvcc),
            "-c",
            source,
            "-o",
            str(output),
            f"-arch={cuda_arch}",
            "--compiler-options",
            host_flags,
            "-std=c++17",
            "-O3",
            "-DWITH_CUDA",
            f"-I{pybind11.get_include()}",
            f"-I{cuda_include}",
        ]
        import sysconfig

        command.append(f"-I{sysconfig.get_path('include')}")
        for include_dir in extension.include_dirs:
            command.append(f"-I{include_dir}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise CompileError(f"NVCC failed with exit code {exc.returncode}") from exc
        extension.sources = []
        extension.extra_objects = [
            *getattr(extension, "extra_objects", []),
            str(output),
        ]


cuda_extension = CUDAExtension(
    name="treemendous.cpp.gpu.boundary_summary_gpu",
    sources=["treemendous/cpp/gpu/boundary_summary_gpu.cu"],
    include_dirs=[str(cuda_include), pybind11.get_include()],
    library_dirs=[str(cuda_lib)],
    libraries=["cudart"],
    define_macros=[("WITH_CUDA", None)],
    language="c++",
)


if __name__ == "__main__":
    setup(
        ext_modules=[*make_cpu_extensions(), cuda_extension],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
