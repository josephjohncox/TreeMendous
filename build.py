import os
import shutil
from typing import List, Dict, Any
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

def build(setup_kwargs: Dict[str, Any], with_ic: bool = False) -> None:
    compile_args = ["-O3"]
    include_dirs = []
    libraries = []
    extra_link_args = []
    
    if with_ic:
        compile_args.append("-DWITH_IC_MANAGER")
        include_dirs.append("/opt/homebrew/Cellar/boost/1.86.0_2/include")
        libraries.append("boost_system")
        extra_link_args.append("-L/opt/homebrew/Cellar/boost/1.86.0_2/lib")
        
    # Determine which extensions to build
    extensions_to_build = []
    
    # Always build the original boundary implementation (known to work)
    extensions_to_build.append(
        Pybind11Extension(
            "treemendous.cpp.boundary",
            ["treemendous/cpp/boundary_bindings.cpp"],
            cxx_std=20,
            extra_compile_args=compile_args,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        )
    )
    
    # Build treap if requested
    import os
    if os.environ.get('BUILD_TREAP', '1') == '1':
        extensions_to_build.append(
            Pybind11Extension(
                "treemendous.cpp.treap",
                ["treemendous/cpp/treap_bindings.cpp"],
                cxx_std=20,
                extra_compile_args=compile_args,
                include_dirs=include_dirs,
                libraries=libraries,
                extra_link_args=extra_link_args,
            )
        )
    
    # Build boundary summary if requested
    if os.environ.get('BUILD_BOUNDARY_SUMMARY', '1') == '1':
        print("🔧 Adding boundary summary extension to build...")
        extensions_to_build.append(
            Pybind11Extension(
                "treemendous.cpp.boundary_summary",
                ["treemendous/cpp/boundary_summary_bindings.cpp"],
                cxx_std=20,
                extra_compile_args=compile_args,
                include_dirs=include_dirs,
                libraries=libraries,
                extra_link_args=extra_link_args,
            )
        )
        print("✅ Boundary summary extension added")
    
    # Build summary enhanced implementations if requested
    if os.environ.get('BUILD_SUMMARY', '0') == '1':
        extensions_to_build.append(
            Pybind11Extension(
                "treemendous.cpp.summary",
                ["treemendous/cpp/summary_bindings.cpp"],
                cxx_std=20,
                extra_compile_args=compile_args + ["-DWITH_SUMMARY_STATS"],
                include_dirs=include_dirs,
                libraries=libraries,
                extra_link_args=extra_link_args,
            )
        )
    
    ext_modules: List[Pybind11Extension] = extensions_to_build

    distribution = Distribution({
        "name": "treemendous",
        "ext_modules": ext_modules
    })

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)
    # setup_kwargs.update({
    #     "ext_modules": ext_modules,
    #     "cmdclass": {"build_ext": build_ext},
    # })