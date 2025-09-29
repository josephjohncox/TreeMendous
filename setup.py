#!/usr/bin/env python
"""
Tree-Mendous Setup with pybind11 Extensions

Following canonical pybind11 setuptools integration pattern:
https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-setuptools
"""

import os
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Check for ICL support
with_icl = os.environ.get('TREE_MENDOUS_WITH_ICL', '0') == '1'

# Compilation settings
compile_args = ["-O3"]
include_dirs = []
libraries = []
extra_link_args = []

if with_icl:
    compile_args.append("-DWITH_IC_MANAGER")
    include_dirs.append("/opt/homebrew/Cellar/boost/1.86.0_2/include")
    libraries.append("boost_system")
    extra_link_args.append("-L/opt/homebrew/Cellar/boost/1.86.0_2/lib")

# Define C++ extensions using pybind11 helpers
ext_modules = [
    # Core boundary manager
    Pybind11Extension(
        "treemendous.cpp.boundary",
        ["treemendous/cpp/boundary_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
    
    # Treap implementation
    Pybind11Extension(
        "treemendous.cpp.treap",
        ["treemendous/cpp/treap_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
    
    # Boundary summary
    Pybind11Extension(
        "treemendous.cpp.boundary_summary",
        ["treemendous/cpp/boundary_summary_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
    
    # Summary enhanced implementations
    Pybind11Extension(
        "treemendous.cpp.summary",
        ["treemendous/cpp/summary_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args + ["-DWITH_SUMMARY_STATS"],
        include_dirs=include_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    )
]

# Setup with pybind11 build_ext for automatic C++ standard detection
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,  # Required for C++ extensions
)