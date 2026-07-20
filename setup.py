#!/usr/bin/env python
"""Portable build configuration for Tree-Mendous CPU extensions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup  # type: ignore[import-untyped]
from setuptools.command.build_py import build_py  # type: ignore[import-untyped]


def _boost_paths() -> tuple[list[str], list[str]]:
    """Return explicitly configured or common Homebrew Boost paths."""
    configured = os.environ.get("BOOST_ROOT")
    candidates = [
        configured,
        "/opt/homebrew/opt/boost",
        "/usr/local/opt/boost",
    ]
    for candidate in candidates:
        if candidate and Path(candidate, "include").is_dir():
            library_dirs = (
                [str(Path(candidate, "lib"))] if Path(candidate, "lib").is_dir() else []
            )
            return [str(Path(candidate, "include"))], library_dirs
    return [], []


class PortableBuildPy(build_py):
    """Remove native rebuild sources from wheels on every host platform."""

    _source_suffixes = frozenset({".cpp", ".h", ".cu", ".mm", ".metal"})

    @classmethod
    def remove_native_sources(cls, package_root: Path) -> None:
        """Delete source-only native files copied into a wheel build tree."""
        if not package_root.is_dir():
            return
        for path in package_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in cls._source_suffixes:
                path.unlink()

    def run(self) -> None:
        super().run()
        self.remove_native_sources(Path(self.build_lib) / "treemendous")


class PortableBuildExt(build_ext):
    """Apply compiler-specific portable defaults at compiler discovery time."""

    def build_extensions(self) -> None:
        compiler_type = self.compiler.compiler_type
        disabled = os.environ.get("TREE_MENDOUS_DISABLE_OPTIMIZATIONS") == "1"
        local_native = os.environ.get("TREE_MENDOUS_LOCAL_NATIVE") == "1"
        sanitizers = os.environ.get("TREE_MENDOUS_SANITIZERS") == "1"

        if compiler_type == "msvc":
            compile_flags = ["/Od"] if disabled else ["/O2"]
            if local_native:
                compile_flags.append("/arch:AVX2")
            link_flags: list[str] = []
        else:
            compile_flags = ["-O0"] if disabled else ["-O3"]
            # Native instruction selection is intentionally an explicit local-only opt in.
            if local_native:
                compile_flags.append("-march=native")
            link_flags = []
            if sanitizers:
                compile_flags.extend(
                    ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                )
                link_flags.append("-fsanitize=address,undefined")

        for extension in self.extensions:
            extension.extra_compile_args = [
                *getattr(extension, "extra_compile_args", []),
                *compile_flags,
            ]
            extension.extra_link_args = [
                *getattr(extension, "extra_link_args", []),
                *link_flags,
            ]
        super().build_extensions()


def make_cpu_extensions() -> list[Pybind11Extension]:
    """Create the cataloged CPU extension definitions."""
    with_icl = os.environ.get("TREE_MENDOUS_WITH_ICL") == "1"
    include_dirs, library_dirs = _boost_paths()
    define_macros: list[tuple[str, str | None]] = []
    libraries: list[str] = []
    if with_icl:
        define_macros.append(("WITH_IC_MANAGER", None))
        libraries.append("boost_system")
    if os.environ.get("TREE_MENDOUS_GLIBCXX_DEBUG") == "1":
        define_macros.append(("_GLIBCXX_DEBUG", None))

    common: dict[str, Any] = {
        "cxx_std": 20,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "define_macros": define_macros,
    }
    return [
        Pybind11Extension(
            "treemendous.cpp.boundary",
            ["treemendous/cpp/boundary_bindings.cpp"],
            **common,
        ),
        Pybind11Extension(
            "treemendous.cpp.treap",
            ["treemendous/cpp/treap_bindings.cpp"],
            **common,
        ),
        Pybind11Extension(
            "treemendous.cpp.boundary_summary",
            ["treemendous/cpp/boundary_summary_bindings.cpp"],
            **common,
        ),
        Pybind11Extension(
            "treemendous.cpp.boundary_summary_optimized",
            ["treemendous/cpp/boundary_summary_optimized_v2_bindings.cpp"],
            **common,
        ),
    ]


def run_setup() -> None:
    setup(
        ext_modules=make_cpu_extensions(),
        cmdclass={
            "build_ext": PortableBuildExt,
            "build_py": PortableBuildPy,
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    run_setup()
