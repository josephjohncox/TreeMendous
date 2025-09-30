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

# Check for ICL support and optimizations
with_icl = os.environ.get('TREE_MENDOUS_WITH_ICL', '0') == '1'
with_optimizations = os.environ.get('TREE_MENDOUS_DISABLE_OPTIMIZATIONS', '0') != '1'

# Detect boost installation paths
boost_paths = [
    "/opt/homebrew/Cellar/boost/1.86.0_2",  # Current
    "/opt/homebrew/opt/boost",  # Symlink
    "/usr/local/opt/boost",  # Intel Mac
]
boost_include = None
boost_lib = None

for path in boost_paths:
    if os.path.exists(path):
        boost_include = f"{path}/include"
        boost_lib = f"{path}/lib"
        break

# Compilation settings
compile_args = ["-O3", "-march=native", "-flto"]  # Enable native CPU optimizations
include_dirs = []
libraries = []
extra_link_args = ["-fuse-linker-plugin"]

# Add boost paths (needed for flat_map)
if boost_include and boost_lib:
    include_dirs.append(boost_include)
    extra_link_args.append(f"-L{boost_lib}")

if with_icl:
    compile_args.append("-DWITH_IC_MANAGER")
    libraries.append("boost_system")

# Optimization flags (can be disabled via TREE_MENDOUS_DISABLE_OPTIMIZATIONS=1)
if with_optimizations:
    # Detect architecture and add appropriate SIMD flags
    import platform
    machine = platform.machine().lower()
    
    if 'arm' in machine or 'aarch64' in machine:
        # ARM/Apple Silicon: use NEON (enabled by default with -O3)
        # No need to explicitly enable NEON on ARM64
        pass
    elif 'x86' in machine or 'amd64' in machine:
        # x86/x64: use AVX2 if available
        compile_args.extend(["-mavx2", "-mfma"])
    
    # Additional optimization hints (arch-independent)
    compile_args.extend(["-ffast-math", "-funroll-loops"])

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
    ),
    
    # OPTIMIZED IMPLEMENTATIONS
    # Optimized boundary manager (flat_map, SIMD, small vector)
    Pybind11Extension(
        "treemendous.cpp.boundary_optimized",
        ["treemendous/cpp/boundary_optimized_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
    
    # Optimized boundary summary (flat_map, SIMD, small vector)
    Pybind11Extension(
        "treemendous.cpp.boundary_summary_optimized",
        ["treemendous/cpp/boundary_summary_optimized_v2_bindings.cpp"],
        cxx_std=20,
        extra_compile_args=compile_args,
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