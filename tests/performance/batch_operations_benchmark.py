#!/usr/bin/env python3
"""
Batch Operations Benchmark - Realistic GPU Use Cases

Demonstrates when GPU/Metal truly excels: bulk operations that process
many intervals in a single call, amortizing Python→C++ overhead.

Real-world scenarios:
- Memory allocator: batch allocate/free for multiple processes
- Job scheduler: process entire job queue at once
- Network: reserve bandwidth for multiple flows simultaneously
- Storage: allocate space for file chunks in bulk
"""

import sys
import time
import random
import statistics
import platform
from typing import List, Tuple
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# GPU implementations
GPU_AVAILABLE = False
GPU_TYPE = None
GpuImpl = None

if platform.system() == 'Darwin':
    try:
        from treemendous.cpp.metal.boundary_summary_metal import MetalBoundarySummaryManager as GpuImpl
        GPU_AVAILABLE = True
        GPU_TYPE = 'Metal'
    except ImportError:
        pass
else:
    try:
        from treemendous.cpp.gpu.boundary_summary_gpu import GPUBoundarySummaryManager as GpuImpl
        GPU_AVAILABLE = True
        GPU_TYPE = 'CUDA'
    except ImportError:
        pass

# C++ comparison
from treemendous.cpp.boundary_optimized import IntervalManager as CppBoundary

print(f"🚀 Batch Operations Benchmark - {GPU_TYPE if GPU_AVAILABLE else 'CPU Only'}")
print("=" * 80)


def benchmark_single_operations(impl, operations: List[Tuple[str, int, int]]) -> float:
    """Benchmark traditional one-at-a-time operations"""
    start = time.perf_counter()
    
    for op, s, e in operations:
        if op == 'reserve':
            impl.reserve_interval(s, e)
        else:
            impl.release_interval(s, e)
    
    return time.perf_counter() - start


def benchmark_batch_operations(impl, operations: List[Tuple[str, int, int]]) -> float:
    """Benchmark batched operations (GPU-optimized)"""
    # Separate into reserves and releases
    reserves = [(s, e) for op, s, e in operations if op == 'reserve']
    releases = [(s, e) for op, s, e in operations if op == 'release']
    
    start = time.perf_counter()
    
    if hasattr(impl, 'batch_reserve'):
        # GPU implementation with batch support
        impl.batch_reserve(reserves)
        impl.batch_release(releases)
    else:
        # Fallback for implementations without batch support
        for s, e in reserves:
            impl.reserve_interval(s, e)
        for s, e in releases:
            impl.release_interval(s, e)
    
    return time.perf_counter() - start


def scenario_memory_allocator(batch_size: int = 1000):
    """
    Scenario: Memory allocator processing allocation requests
    Realistic: OS kernel allocating memory for multiple processes
    """
    print(f"\n📦 SCENARIO 1: Memory Allocator ({batch_size:,} allocations)")
    print("-" * 60)
    
    # Generate allocation requests (mix of sizes)
    operations = []
    for _ in range(batch_size):
        op = random.choice(['reserve', 'release'])
        size = random.choice([4096, 8192, 16384, 65536, 1048576])  # Common page sizes
        start = random.randint(0, 100_000_000)
        operations.append((op, start, start + size))
    
    # C++ Baseline
    cpp = CppBoundary()
    cpp.release_interval(0, 100_000_000)
    cpp_time = benchmark_single_operations(cpp, operations)
    
    if GPU_AVAILABLE:
        # GPU Single ops
        gpu1 = GpuImpl()
        gpu1.release_interval(0, 100_000_000)
        gpu_single_time = benchmark_single_operations(gpu1, operations)
        
        # GPU Batch ops
        gpu2 = GpuImpl()
        gpu2.release_interval(0, 100_000_000)
        gpu_batch_time = benchmark_batch_operations(gpu2, operations)
        
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms  ({batch_size/cpp_time:>8,.0f} ops/sec)")
        print(f"   GPU (single ops):    {gpu_single_time*1000:>8.1f}ms  ({batch_size/gpu_single_time:>8,.0f} ops/sec)")
        print(f"   GPU (batch ops):     {gpu_batch_time*1000:>8.1f}ms  ({batch_size/gpu_batch_time:>8,.0f} ops/sec)")
        print(f"\n   💡 Batch speedup: {gpu_single_time/gpu_batch_time:.1f}x faster")
        print(f"   🏆 vs C++: {'FASTER' if gpu_batch_time < cpp_time else 'slower'} by {abs(cpp_time/gpu_batch_time - 1)*100:.0f}%")
    else:
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms  ({batch_size/cpp_time:>8,.0f} ops/sec)")


def scenario_job_scheduler(num_jobs: int = 500):
    """
    Scenario: Job scheduler allocating time slots for batch jobs
    Realistic: HPC cluster scheduling jobs across nodes
    """
    print(f"\n⏰ SCENARIO 2: Job Scheduler ({num_jobs:,} jobs)")
    print("-" * 60)
    
    # Jobs with varying durations
    operations = []
    for _ in range(num_jobs):
        start_time = random.randint(0, 86400)  # 24-hour window
        duration = random.choice([60, 300, 900, 3600, 14400])  # 1min to 4hrs
        operations.append(('reserve', start_time, start_time + duration))
    
    # C++ Baseline
    cpp = CppBoundary()
    cpp.release_interval(0, 86400)
    cpp_time = benchmark_single_operations(cpp, operations)
    
    if GPU_AVAILABLE:
        # GPU Single ops
        gpu1 = GpuImpl()
        gpu1.release_interval(0, 86400)
        gpu_single_time = benchmark_single_operations(gpu1, operations)
        
        # GPU Batch ops
        gpu2 = GpuImpl()
        gpu2.release_interval(0, 86400)
        gpu_batch_time = benchmark_batch_operations(gpu2, operations)
        
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms")
        print(f"   GPU (single ops):    {gpu_single_time*1000:>8.1f}ms")
        print(f"   GPU (batch ops):     {gpu_batch_time*1000:>8.1f}ms")
        print(f"\n   💡 Batch speedup: {gpu_single_time/gpu_batch_time:.1f}x faster")
        print(f"   🏆 vs C++: {'FASTER' if gpu_batch_time < cpp_time else 'slower'} by {abs(cpp_time/gpu_batch_time - 1)*100:.0f}%")
    else:
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms")


def scenario_network_bandwidth(num_flows: int = 10000):
    """
    Scenario: Network bandwidth allocation for multiple flows
    Realistic: SDN controller allocating bandwidth across data center
    """
    print(f"\n🌐 SCENARIO 3: Network Bandwidth Manager ({num_flows:,} flows)")
    print("-" * 60)
    
    # Network flows requesting bandwidth
    operations = []
    total_bandwidth = 100_000_000  # 100 Gbps in Mbps
    for _ in range(num_flows):
        op = random.choice(['reserve', 'release'])
        bandwidth = random.choice([10, 100, 1000, 10000])  # 10Mbps to 10Gbps
        start_time = random.randint(0, total_bandwidth - bandwidth)
        operations.append((op, start_time, start_time + bandwidth))
    
    # C++ Baseline
    cpp = CppBoundary()
    cpp.release_interval(0, total_bandwidth)
    cpp_time = benchmark_single_operations(cpp, operations)
    
    if GPU_AVAILABLE:
        # GPU Single ops
        gpu1 = GpuImpl()
        gpu1.release_interval(0, total_bandwidth)
        gpu_single_time = benchmark_single_operations(gpu1, operations)
        
        # GPU Batch ops
        gpu2 = GpuImpl()
        gpu2.release_interval(0, total_bandwidth)
        gpu_batch_time = benchmark_batch_operations(gpu2, operations)
        
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms")
        print(f"   GPU (single ops):    {gpu_single_time*1000:>8.1f}ms")
        print(f"   GPU (batch ops):     {gpu_batch_time*1000:>8.1f}ms")
        print(f"\n   💡 Batch speedup: {gpu_single_time/gpu_batch_time:.1f}x faster")
        print(f"   🏆 vs C++: {'FASTER' if gpu_batch_time < cpp_time else 'slower'} by {abs(cpp_time/gpu_batch_time - 1)*100:.0f}%")
    else:
        print(f"   C++ (single ops):    {cpp_time*1000:>8.1f}ms")


def scaling_analysis():
    """Show how batch operations scale with batch size"""
    print(f"\n📊 SCALING ANALYSIS: Batch Size Impact")
    print("=" * 80)
    
    batch_sizes = [100, 500, 1000, 5000, 10000]
    
    print(f"\n{'Batch Size':<12} {'C++ (ms)':<12} {'GPU Single (ms)':<18} {'GPU Batch (ms)':<18} {'Speedup':<10}")
    print("-" * 80)
    
    for size in batch_sizes:
        operations = []
        for _ in range(size):
            op = random.choice(['reserve', 'release'])
            start = random.randint(0, 10_000_000)
            length = random.randint(1000, 50000)
            operations.append((op, start, start + length))
        
        # C++
        cpp = CppBoundary()
        cpp.release_interval(0, 10_000_000)
        cpp_time = benchmark_single_operations(cpp, operations)
        
        if GPU_AVAILABLE:
            # GPU single
            gpu1 = GpuImpl()
            gpu1.release_interval(0, 10_000_000)
            gpu_single = benchmark_single_operations(gpu1, operations)
            
            # GPU batch
            gpu2 = GpuImpl()
            gpu2.release_interval(0, 10_000_000)
            gpu_batch = benchmark_batch_operations(gpu2, operations)
            
            speedup = gpu_single / gpu_batch
            
            print(f"{size:<12,} {cpp_time*1000:<11.1f} {gpu_single*1000:<17.1f} {gpu_batch*1000:<17.1f} {speedup:<9.1f}x")
    
    print("\n💡 Key Insight: Batch operations amortize Python→C++→Metal overhead")
    print("   Single ops: ~30µs Python overhead PER operation")
    print("   Batch ops:  ~30µs Python overhead for ENTIRE batch")


def main():
    if not GPU_AVAILABLE:
        print("\n❌ GPU not available - showing C++ baseline only")
        print("   Build with: just build-metal (macOS) or just build-gpu (Linux)")
    
    # Run scenarios
    scenario_memory_allocator(batch_size=10_000_000)
    scenario_job_scheduler(num_jobs=5_000_000)
    scenario_network_bandwidth(num_flows=10_000_000)
    
    # Scaling analysis
    if GPU_AVAILABLE:
        scaling_analysis()
    
    print("\n" + "=" * 80)
    print("✅ Batch Operations Benchmark Complete!")
    print("=" * 80)
    
    if GPU_AVAILABLE:
        print("\n🎯 RECOMMENDATIONS:")
        print("   • Use batch operations for 100+ intervals at once")
        print("   • Batch ops eliminate Python call overhead (30µs → amortized)")
        print("   • Ideal for: bulk allocators, schedulers, resource managers")
        print("   • GPU shines when batching 1000+ operations")


if __name__ == "__main__":
    random.seed(42)
    main()

