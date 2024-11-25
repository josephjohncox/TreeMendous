from pathlib import Path
import subprocess
from setuptools import setup
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

class CustomBuildExt(build_ext):
    def run(self) -> None:
        # Build C++ extensions
        super().run()
        
        # Build Rust extensions
        subprocess.check_call(
            ["maturin", "develop", "--release", "--strip"],
            cwd=str(Path(__file__).parent)
        )

def build(setup_kwargs):
    # C++ extension
    ext_modules = [
        Pybind11Extension(
            "treemendous.cpp.boundary",
            ["treemendous/cpp/boundary_bindings.cpp"],
            language='c++',
            cxx_std=20
        )

    ]

    # Update build_kwargs for both C++ and Rust
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": CustomBuildExt},
    })

if __name__ == "__main__":
    setup()