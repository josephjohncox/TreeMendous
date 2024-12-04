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
        
    ext_modules: List[Pybind11Extension] = [
        Pybind11Extension(
            "treemendous.cpp.boundary",
            ["treemendous/cpp/boundary_bindings.cpp"],
            cxx_std=20,
            extra_compile_args=compile_args,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ]

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