from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "treemendous.cpp.boundary",
        ["treemendous/cpp/boundary_bindings.cpp"],
        cxx_std=20,  # Use C++20
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)