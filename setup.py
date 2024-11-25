from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'treemendous.cpp.boundary',
        ['treemendous/cpp/boundary_bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++20']  # Add C++20 support
    ),
]

setup(
    name='boundary',
    ext_modules=ext_modules,
)