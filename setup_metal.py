#!/usr/bin/env python
"""Build the portable CPU extensions plus the experimental macOS Metal backend."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from pybind11 import get_include
from setuptools import Extension, setup

from setup import PortableBuildExt, make_cpu_extensions

if platform.system() != "Darwin":
    raise SystemExit("Metal extensions can only be built on macOS")

try:
    subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True,
        capture_output=True,
        text=True,
    )
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    raise SystemExit(
        "Metal build requires Xcode Command Line Tools and a macOS SDK"
    ) from exc


class MetalExtension(Extension):
    """Extension carrying shader sources compiled by ``BuildMetalExtension``."""

    def __init__(self, *args, metal_sources: list[str], **kwargs):
        self.metal_sources = metal_sources
        self.metallib_files: list[Path] = []
        super().__init__(*args, **kwargs)


class BuildMetalExtension(PortableBuildExt):
    """Compile shaders and install metallibs beside the installed extension."""

    def build_extensions(self) -> None:
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        if hasattr(self.compiler, "language_map"):
            self.compiler.language_map[".mm"] = "c++"

        for extension in self.extensions:
            if isinstance(extension, MetalExtension):
                extension.metallib_files = self._compile_metallibs(extension)

        super().build_extensions()

        for extension in self.extensions:
            if isinstance(extension, MetalExtension):
                self._install_metallibs(extension)

    def _compile_metallibs(self, extension: MetalExtension) -> list[Path]:
        output_dir = Path(self.build_temp) / "metal-resources"
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for source_name in extension.metal_sources:
            source = Path(source_name)
            air = output_dir / f"{source.stem}.air"
            metallib = output_dir / f"{source.stem}.metallib"
            self.spawn(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-c",
                    str(source),
                    "-o",
                    str(air),
                    "-O3",
                ]
            )
            self.spawn(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metallib",
                    str(air),
                    "-o",
                    str(metallib),
                ]
            )
            outputs.append(metallib)
        return outputs

    def _install_metallibs(self, extension: MetalExtension) -> None:
        # ``build_lib`` is the wheel staging tree. ``get_ext_fullpath`` points at
        # the source package for --inplace builds and at build_lib otherwise.
        destinations = {
            Path(self.build_lib) / "treemendous/cpp/metal/resources",
            Path(self.get_ext_fullpath(extension.name)).parent / "resources",
        }
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            for metallib in extension.metallib_files:
                shutil.copy2(metallib, destination / metallib.name)


metal_extension = MetalExtension(
    name="treemendous.cpp.metal.boundary_summary_metal",
    sources=[
        "treemendous/cpp/metal/boundary_summary_metal.mm",
        "treemendous/cpp/metal/boundary_summary_metal_bindings.cpp",
    ],
    metal_sources=["treemendous/cpp/metal/boundary_summary_metal.metal"],
    include_dirs=[get_include()],
    extra_compile_args=[
        "-std=c++17",
        "-DMETAL_AVAILABLE",
        "-fobjc-arc",
        "-Wno-unused-command-line-argument",
    ],
    extra_link_args=[
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "Foundation",
    ],
    language="c++",
)


if __name__ == "__main__":
    setup(
        ext_modules=[*make_cpu_extensions(), metal_extension],
        cmdclass={"build_ext": BuildMetalExtension},
        package_data={"treemendous.cpp.metal": ["resources/*.metallib"]},
        zip_safe=False,
    )
