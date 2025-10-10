#!/usr/bin/env python
"""
Tree-Mendous Metal Build Configuration

This setup file adds Metal/MPS support for GPU acceleration on macOS.
Works with Apple Silicon (M1/M2/M3/M4) and Intel Macs with AMD GPUs.

Requirements:
  - macOS 10.15+ (Catalina or later)
  - Xcode Command Line Tools
  - Python 3.11+
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Check if running on macOS
if platform.system() != 'Darwin':
    print("❌ Metal is only available on macOS")
    print("   For Linux/Windows, use: WITH_CUDA=1 python setup_gpu.py build_ext --inplace")
    sys.exit(1)

print("✅ Building Metal-accelerated extensions for macOS")

# Check for pybind11
try:
    import pybind11
    pybind11_include = pybind11.get_include()
    print(f"✅ Found pybind11: {pybind11_include}")
except ImportError:
    print("❌ pybind11 not found. Install with: pip install pybind11")
    sys.exit(1)

# Check for Metal availability
try:
    result = subprocess.run(
        ['xcrun', '--sdk', 'macosx', '--show-sdk-path'],
        capture_output=True,
        text=True,
        check=True
    )
    sdk_path = result.stdout.strip()
    print(f"✅ Found macOS SDK: {sdk_path}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("❌ Xcode Command Line Tools not found")
    print("   Install with: xcode-select --install")
    sys.exit(1)


class BuildMetalExtension(build_ext):
    """Custom build extension that handles Metal shader compilation"""
    
    def build_extensions(self):
        # Register .mm as a valid C++ source file extension
        self.compiler.src_extensions.append('.mm')
        
        # Make sure .mm files are compiled with appropriate flags
        if hasattr(self.compiler, 'language_map'):
            self.compiler.language_map['.mm'] = 'c++'
        
        # Compile Metal shaders first
        for ext in self.extensions:
            if hasattr(ext, 'metal_sources'):
                self.build_metal_shaders(ext)
        
        # Then build normally
        build_ext.build_extensions(self)
    
    def build_metal_shaders(self, ext):
        """Compile Metal shaders to metallib"""
        print(f"\n🔧 Compiling Metal shaders for {ext.name}")
        
        metal_files = getattr(ext, 'metal_sources', [])
        if not metal_files:
            return
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        metallib_files = []
        
        for metal_file in metal_files:
            metal_path = Path(metal_file)
            air_file = build_temp / (metal_path.stem + '.air')
            metallib_file = build_temp / (metal_path.stem + '.metallib')
            
            # Compile .metal to .air
            print(f"   Compiling: {metal_file} -> {air_file}")
            metal_cmd = [
                'xcrun', '-sdk', 'macosx', 'metal',
                '-c',
                str(metal_file),
                '-o', str(air_file),
                '-std=metal3.2',  # Metal 3.2 for macOS
                '-O3',  # Optimize
            ]
            
            # Add architecture flags
            arch = platform.machine()
            if arch == 'arm64':
                metal_cmd.extend(['-arch', 'air64'])
            else:
                metal_cmd.extend(['-arch', 'air64'])
            
            result = subprocess.run(metal_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Metal compilation failed:")
                print(result.stderr)
                sys.exit(1)
            
            # Create metallib
            print(f"   Creating metallib: {metallib_file}")
            metallib_cmd = [
                'xcrun', '-sdk', 'macosx', 'metallib',
                str(air_file),
                '-o', str(metallib_file)
            ]
            
            result = subprocess.run(metallib_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Metallib creation failed:")
                print(result.stderr)
                sys.exit(1)
            
            metallib_files.append(str(metallib_file))
            print(f"   ✅ Created: {metallib_file}")
        
        # Store metallib paths for installation
        ext.metallib_files = metallib_files


# Define Metal extension
metal_sources = [
    'treemendous/cpp/metal/boundary_summary_metal.mm',       # Objective-C++ implementation
    'treemendous/cpp/metal/boundary_summary_metal_bindings.cpp',  # Python bindings
]

# Metal shaders will be compiled separately

class MetalExtension(Extension):
    """Extension with Metal shader sources"""
    def __init__(self, *args, **kwargs):
        self.metal_sources = kwargs.pop('metal_sources', [])
        super().__init__(*args, **kwargs)


metal_extension = MetalExtension(
    name='treemendous.cpp.metal.boundary_summary_metal',
    sources=metal_sources,
    metal_sources=[
        'treemendous/cpp/metal/boundary_summary_metal.metal',
    ],
    include_dirs=[
        pybind11_include,
    ],
    extra_compile_args=[
        '-std=c++17',
        '-O3',
        '-DMETAL_AVAILABLE',
        '-fobjc-arc',  # Enable ARC for Objective-C++ files
        '-Wno-unused-command-line-argument',
    ],
    extra_link_args=[
        '-framework', 'Metal',
        '-framework', 'MetalPerformanceShaders',
        '-framework', 'Foundation',
    ],
    language='c++',
)

print("\n📦 Metal Extension Configuration:")
print(f"   Name: {metal_extension.name}")
print(f"   Sources: {metal_extension.sources}")
print(f"   Metal Shaders: {metal_extension.metal_sources}")
print(f"   Frameworks: Metal, MetalPerformanceShaders, Foundation")
print("")

# Setup
setup(
    name='treemendous-metal',
    ext_modules=[metal_extension],
    cmdclass={'build_ext': BuildMetalExtension},
    zip_safe=False,
)

print("\n✅ Metal build configuration complete!")
print("\nTo build:")
print("  python setup_metal.py build_ext --inplace")
print("\nTo test:")
print("  python -c 'from treemendous.cpp.metal import boundary_summary_metal; print(boundary_summary_metal.get_metal_device_info())'")
print("\nTo benchmark:")
print("  just test-metal")

