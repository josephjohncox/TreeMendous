#!/usr/bin/env python
"""
Tree-Mendous GPU Build Configuration

This setup file adds CUDA/GPU support to the Tree-Mendous package.
It extends the base setup.py with GPU-specific build configuration.

Environment Variables:
  WITH_CUDA=1              Enable CUDA support (requires CUDA toolkit)
  CUDA_HOME=/path/to/cuda  Specify CUDA installation directory
  CUDA_ARCH=sm_75          Specify GPU compute capability (default: auto-detect)
"""

import os
import sys
import subprocess
from pathlib import Path
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Check if CUDA is requested
with_cuda = os.environ.get('WITH_CUDA', '0') == '1'

if not with_cuda:
    print("❌ CUDA support not requested. Set WITH_CUDA=1 to enable GPU acceleration.")
    print("   Example: WITH_CUDA=1 python setup_gpu.py build_ext --inplace")
    sys.exit(1)

# Detect CUDA installation
cuda_home = os.environ.get('CUDA_HOME')
if not cuda_home:
    # Try common locations
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/lib/cuda',
        '/opt/cuda',
        '/usr/local/cuda-12.0',
        '/usr/local/cuda-11.0',
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_home = path
            break

if not cuda_home or not os.path.exists(cuda_home):
    print("❌ CUDA toolkit not found!")
    print("   Please install CUDA toolkit and set CUDA_HOME environment variable.")
    print("   Download from: https://developer.nvidia.com/cuda-downloads")
    sys.exit(1)

print(f"✅ Found CUDA installation: {cuda_home}")

# CUDA paths
cuda_include = os.path.join(cuda_home, 'include')
cuda_lib = os.path.join(cuda_home, 'lib64')
if not os.path.exists(cuda_lib):
    cuda_lib = os.path.join(cuda_home, 'lib')

nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
if not os.path.exists(nvcc_path):
    print(f"❌ nvcc not found at {nvcc_path}")
    sys.exit(1)

print(f"✅ Found nvcc: {nvcc_path}")

# Detect GPU compute capability
cuda_arch = os.environ.get('CUDA_ARCH')
if not cuda_arch:
    try:
        # Try to detect GPU capability
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        compute_cap = result.stdout.strip().split('\n')[0]
        cuda_arch = f"sm_{compute_cap.replace('.', '')}"
        print(f"✅ Auto-detected GPU compute capability: {cuda_arch}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Default to common capability
        cuda_arch = 'sm_75'  # Tesla T4, RTX 2060/2070/2080
        print(f"⚠️  Could not detect GPU, using default: {cuda_arch}")
        print("   Set CUDA_ARCH environment variable to override (e.g., sm_86 for RTX 3090)")

# Check for pybind11
try:
    import pybind11
    pybind11_include = pybind11.get_include()
    print(f"✅ Found pybind11: {pybind11_include}")
except ImportError:
    print("❌ pybind11 not found. Install with: pip install pybind11")
    sys.exit(1)


class CUDAExtension(Extension):
    """Custom extension class for CUDA modules"""
    pass


class BuildExtension(build_ext):
    """Custom build extension that handles CUDA compilation"""
    
    def build_extensions(self):
        # Compile CUDA files to object files first
        for ext in self.extensions:
            if isinstance(ext, CUDAExtension):
                self.build_cuda_extension(ext)
        
        # Then build normally
        build_ext.build_extensions(self)
    
    def build_cuda_extension(self, ext):
        """Compile CUDA sources to object files"""
        print(f"\n🔧 Building CUDA extension: {ext.name}")
        
        # Find .cu files
        cu_files = [s for s in ext.sources if s.endswith('.cu')]
        cpp_files = [s for s in ext.sources if s.endswith('.cpp')]
        
        if not cu_files:
            return
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Compile each .cu file
        object_files = []
        for cu_file in cu_files:
            obj_file = build_temp / (Path(cu_file).stem + '.o')
            
            nvcc_cmd = [
                nvcc_path,
                '-c',
                cu_file,
                '-o', str(obj_file),
                f'-arch={cuda_arch}',
                '--compiler-options', '-fPIC',
                '-std=c++17',
                '-O3',
                '--use_fast_math',
                '-DWITH_CUDA',
                f'-I{pybind11_include}',
                f'-I{cuda_include}',
            ]
            
            # Add Python includes
            import sysconfig
            python_include = sysconfig.get_path('include')
            nvcc_cmd.append(f'-I{python_include}')
            
            # Add extra includes from extension
            for inc in ext.include_dirs:
                nvcc_cmd.append(f'-I{inc}')
            
            print(f"   Compiling: {cu_file}")
            print(f"   Command: {' '.join(nvcc_cmd)}")
            
            result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ NVCC compilation failed!")
                print(result.stderr)
                sys.exit(1)
            
            object_files.append(str(obj_file))
            print(f"   ✅ Created: {obj_file}")
        
        # Replace .cu files with compiled .o files in sources
        ext.sources = cpp_files + object_files
        
        # Add CUDA libraries
        if cuda_lib not in ext.library_dirs:
            ext.library_dirs.append(cuda_lib)
        
        for lib in ['cudart', 'cudart_static']:
            if lib not in ext.libraries:
                ext.libraries.append(lib)
        
        print(f"   ✅ CUDA extension {ext.name} ready for linking")


# Define GPU extension
gpu_extension = CUDAExtension(
    name='treemendous.cpp.gpu.boundary_summary_gpu',
    sources=[
        'treemendous/cpp/gpu/boundary_summary_gpu.cu',
        'treemendous/cpp/gpu/boundary_summary_gpu_bindings.cpp',
    ],
    include_dirs=[
        pybind11_include,
        cuda_include,
    ],
    library_dirs=[
        cuda_lib,
    ],
    libraries=[
        'cudart',
    ],
    extra_compile_args=[
        '-O3',
        '-std=c++17',
        '-DWITH_CUDA',
    ],
    extra_link_args=[
        f'-L{cuda_lib}',
        '-lcudart',
    ],
    language='c++',
)

print("\n📦 GPU Extension Configuration:")
print(f"   Name: {gpu_extension.name}")
print(f"   Sources: {gpu_extension.sources}")
print(f"   CUDA Home: {cuda_home}")
print(f"   CUDA Arch: {cuda_arch}")
print(f"   Include dirs: {gpu_extension.include_dirs}")
print(f"   Library dirs: {gpu_extension.library_dirs}")
print("")

# Setup
setup(
    name='treemendous-gpu',
    ext_modules=[gpu_extension],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

print("\n✅ GPU build configuration complete!")
print("\nTo build:")
print("  WITH_CUDA=1 python setup_gpu.py build_ext --inplace")
print("\nTo test:")
print("  python -c 'from treemendous.cpp.gpu import boundary_summary_gpu; print(boundary_summary_gpu.get_cuda_device_info())'")

