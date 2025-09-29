# 🔥 C++ Profiling Guide for Tree-Mendous

## Overview

Profiling C++ extensions requires different tools than Python profiling. This guide covers multiple approaches for profiling Tree-Mendous C++ implementations and generating flame graphs.

## 🛠️ Profiling Tools

### Tool Comparison

| Tool | Platform | Setup | Output | Best For |
|------|----------|-------|--------|----------|
| **py-spy** | All | Easy | SVG flame graph | Mixed Python/C++ |
| **perf** | Linux | Medium | Flame graph | Linux systems |
| **Instruments** | macOS | Easy | Interactive | macOS development |
| **Valgrind** | Linux/macOS | Easy | Call graph | Detailed analysis |
| **gprof** | All | Build flags | Text report | Traditional profiling |

## 🚀 Quick Start

### Method 1: py-spy (Recommended)

**Advantages**: 
- Shows both Python and C++ frames
- No recompilation needed
- Easy to use
- Generates flame graphs

**Installation**:
```bash
pip install py-spy

# On macOS/Linux, may need elevated privileges:
sudo pip install py-spy
```

**Basic Usage**:
```bash
# Profile C++ benchmark
py-spy record --native -o cpp_profile.svg -- python tests/performance/cpp_profiler.py

# Or use our script
uv run python tests/performance/cpp_profiler.py
```

**Advanced Options**:
```bash
# Higher sampling rate for more detail
py-spy record --native --rate 1000 -o profile.svg -- python script.py

# Different output formats
py-spy record --native --format speedscope -o profile.json -- python script.py

# Live top-like view
py-spy top --native --pid <PID>
```

### Method 2: macOS Instruments

**Installation**: Comes with Xcode Command Line Tools

**Usage**:
```bash
# 1. Start your benchmark
python tests/performance/cpp_profiler.py &
PID=$!

# 2. Profile with sample
sample $PID 10 -file cpp_profile.txt

# Or use Instruments GUI
# Xcode -> Open Developer Tool -> Instruments
# Choose "Time Profiler"
# Attach to Python process
```

### Method 3: Linux perf

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install linux-tools-common linux-tools-generic

# Fedora/RHEL
sudo dnf install perf
```

**Usage**:
```bash
# Record performance data
perf record -g python tests/performance/cpp_profiler.py

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Or use flamegraph directly
perf script report flamegraph
```

## 🔧 Building C++ with Profiling Support

### Option 1: Debug Build with Symbols

Edit `setup.py`:

```python
compile_args = ["-O2", "-g"]  # -g adds debug symbols
```

Rebuild:
```bash
just clean-cpp
just build-cpp
```

### Option 2: Profiling Build with gprof

```python
# In setup.py, add profiling flags
compile_args = ["-O2", "-pg"]  # -pg enables gprof profiling
extra_link_args = ["-pg"]

# After rebuild, running creates gmon.out
# Analyze with: gprof yourprogram gmon.out
```

### Option 3: Add Custom Instrumentation

Add timing instrumentation to C++ code:

```cpp
// In boundary.cpp
#include <chrono>
#include <iostream>

class Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::string name;
public:
    Timer(const std::string& n) : name(n) {
        start = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << name << ": " << duration.count() << "µs\n";
    }
};

// Use in functions
void reserve_interval(int start, int end) {
    Timer t("reserve_interval");  // Automatically times
    // ... implementation
}
```

## 📊 Reading C++ Flame Graphs

### Understanding the Output

```
Flame Graph with C++ Frames:

┌────────────────────────────────────────────────────────┐
│ Python: benchmark_interval_manager()                   │ Python frame
├────────────────────────────────────────────────────────┤
│ CPython: PyObject_Call                                 │ Python/C bridge
├────────────────────────────────────────────────────────┤
│ C++: IntervalManager::reserve_interval()               │ C++ implementation
├─────────────────┬──────────────────────────────────────┤
│ std::map::insert│ std::map::find                      │ STL operations
└─────────────────┴──────────────────────────────────────┘
```

**What to Look For**:

1. **Python overhead**: Time spent in Python interpreter vs C++ code
2. **Binding overhead**: Time in pybind11 conversion layer
3. **C++ hotspots**: STL operations, allocations, comparisons
4. **Lock contention**: If using threads (not currently)

### Example Analysis

```
# Good C++ performance profile
Python overhead:     5%  ← Minimal
Pybind11 binding:    5%  ← Acceptable
C++ implementation: 90%  ← Most time in actual logic

# Poor C++ performance profile  
Python overhead:    40%  ← Too high
Pybind11 binding:   30%  ← Conversion bottleneck
C++ implementation: 30%  ← Not utilizing C++ efficiently
```

## 🎯 Optimization Strategies

### 1. Reduce Python/C++ Boundary Crossings

**Problem**: Frequent calls across Python/C++ boundary

```python
# ❌ Inefficient: Many small C++ calls
for i in range(10000):
    cpp_manager.reserve_interval(i, i+1)  # 10k boundary crossings

# ✅ Efficient: Batch operations in C++
cpp_manager.reserve_intervals_batch([(i, i+1) for i in range(10000)])
```

**Solution**: Add batch operations to C++ interface

```cpp
// In boundary.cpp
void reserve_intervals_batch(const std::vector<std::pair<int, int>>& intervals) {
    for (const auto& [start, end] : intervals) {
        reserve_interval(start, end);
    }
}
```

### 2. Optimize STL Usage

**Profile showing `std::map` overhead**:

```cpp
// Current: std::map with many small operations
std::map<int, int> intervals;  // O(log n) per operation

// Optimization options:
// 1. Use flat_map for small sizes
boost::container::flat_map<int, int> intervals;  // Better cache locality

// 2. Reserve capacity for vectors
std::vector<Interval> intervals;
intervals.reserve(1000);  // Avoid reallocations

// 3. Use custom allocator
std::map<int, int, std::less<int>, PoolAllocator<std::pair<const int, int>>> intervals;
```

### 3. Enable Link-Time Optimization (LTO)

```python
# In setup.py
compile_args = ["-O3", "-flto"]  # Enable LTO
extra_link_args = ["-flto"]

# Potential 10-20% speedup with better inlining
```

## 🔬 Advanced Profiling Techniques

### CPU Cache Analysis

```bash
# On Linux with perf
perf stat -e cache-references,cache-misses,cycles,instructions python script.py

# Look for:
# - Cache miss rate (should be <5%)
# - Instructions per cycle (should be >1.0)
```

### Memory Access Patterns

```bash
# Valgrind cachegrind
valgrind --tool=cachegrind --cache-sim=yes python script.py

# Analyze output
cg_annotate cachegrind.out.*
```

### Branch Prediction Analysis

```bash
# perf with branch events
perf stat -e branches,branch-misses python script.py

# Should see <1% branch miss rate for optimal performance
```

## 📈 Performance Optimization Workflow

### Step 1: Establish Baseline

```bash
# Run benchmark and save results
just test-perf > baseline.txt

# Profile C++ code
uv run python tests/performance/cpp_profiler.py
```

### Step 2: Identify Bottleneck

```bash
# Generate flame graph with py-spy
py-spy record --native -o before.svg -- python benchmark.py

# Look for:
# - Widest bars (most time)
# - Unexpected functions
# - Excessive allocations
```

### Step 3: Optimize

```cpp
// Example: Optimize interval merging
// Before:
for (auto it = intervals.begin(); it != intervals.end(); ++it) {
    if (should_merge(*it)) {
        merge(*it);  // Inefficient: multiple map lookups
    }
}

// After:
auto it = intervals.begin();
while (it != intervals.end()) {
    if (should_merge(*it)) {
        it = merge_and_advance(it);  // Single pass, fewer lookups
    } else {
        ++it;
    }
}
```

### Step 4: Measure Impact

```bash
# Rebuild with optimization
just clean-cpp && just build-cpp

# Re-profile
py-spy record --native -o after.svg -- python benchmark.py

# Compare
# - Check if hotspot reduced
# - Verify overall time improved
# - Ensure correctness maintained (just test)
```

### Step 5: Validate

```bash
# Full test suite
just test

# Performance regression check
just test-perf

# Compare before/after
# - Should see improvement in target operation
# - No regression in other operations
# - P95/P99 latencies should improve
```

## 🎓 Real-World Example

### Profiling Boundary Manager C++

**Initial Profile**:
```
Function                                Time    %
std::map::insert                       45ms   45%  ← Hotspot!
IntervalManager::reserve_interval      35ms   35%
std::map::find                         15ms   15%
merge_adjacent_intervals                5ms    5%
```

**Analysis**: Most time in `std::map::insert`

**Optimization**: Batch inserts where possible

```cpp
// Before: Individual inserts
void release_intervals(const std::vector<Interval>& intervals) {
    for (const auto& interval : intervals) {
        intervals_[interval.start] = interval.end;  // Separate insert each time
    }
}

// After: Batch with hint
void release_intervals(const std::vector<Interval>& intervals) {
    auto hint = intervals_.begin();
    for (const auto& interval : intervals) {
        hint = intervals_.insert(hint, {interval.start, interval.end});
    }
}
```

**Results**:
```
Before: 100ms for 10k operations
After:   75ms for 10k operations
Speedup: 1.33x
```

## 📋 Profiling Checklist

### Before Profiling
- [ ] Ensure C++ extensions are compiled with debug symbols (`-g`)
- [ ] Install profiling tools (py-spy recommended)
- [ ] Create realistic workload for profiling
- [ ] Run baseline benchmark for comparison

### During Profiling
- [ ] Use `--native` flag with py-spy to see C++ frames
- [ ] Profile for sufficient duration (10+ seconds)
- [ ] Check both wall-clock time and CPU time
- [ ] Verify profiler overhead is acceptable (<10%)

### After Profiling
- [ ] Generate flame graph for visualization
- [ ] Identify top 3 hotspots
- [ ] Verify hotspots make sense (not profiler artifacts)
- [ ] Document findings

### After Optimization
- [ ] Re-profile to verify improvement
- [ ] Run full test suite to ensure correctness
- [ ] Compare before/after flame graphs
- [ ] Update performance documentation

## 🔍 Troubleshooting

### Issue: Can't see C++ frames in flame graph

**Solution 1**: Use `--native` flag with py-spy
```bash
py-spy record --native -o profile.svg -- python script.py
```

**Solution 2**: Ensure debug symbols in build
```python
# In setup.py
compile_args = ["-O2", "-g"]  # Add -g
```

**Solution 3**: Disable stripping
```python
# In setup.py, remove strip commands or:
# In Pybind11Extension, avoid strip in release builds
```

### Issue: py-spy requires sudo

**Why**: py-spy needs ptrace permissions to inspect running processes

**Solutions**:
```bash
# Option 1: Install with sudo
sudo pip install py-spy

# Option 2: Give py-spy ptrace capability (Linux)
sudo setcap cap_sys_ptrace=+ep $(which py-spy)

# Option 3: Run py-spy as yourself (macOS)
# macOS usually allows this after first sudo run
```

### Issue: Profile shows mostly "unknown" frames

**Cause**: Missing debug symbols or stripped binaries

**Solution**: Rebuild with symbols
```bash
just clean-cpp
# Edit setup.py to ensure -g flag
just build-cpp
```

## 🎯 Performance Goals

### Target Metrics for C++ Implementations

```
Operation              Target       Excellent
reserve_interval       < 1µs        < 0.5µs
release_interval       < 1µs        < 0.5µs
find_interval          < 1µs        < 0.5µs
get_intervals          < 10µs       < 5µs
```

### Speedup Targets vs Python

```
Implementation         Minimum      Target      Excellent
Boundary Manager         3x           5x           10x
Summary Tree             3x           5x           8x
Treap                    5x          10x           20x
Boundary Summary         3x           5x          10x
```

## 📚 Additional Resources

- **Brendan Gregg's Flame Graphs**: http://www.brendangregg.com/flamegraphs.html
- **py-spy Documentation**: https://github.com/benfred/py-spy
- **Linux perf Tutorial**: https://perf.wiki.kernel.org/index.php/Tutorial
- **C++ Performance Optimization**: https://www.agner.org/optimize/

---

**Profile early, profile often. Data-driven optimization beats intuition.** 🎯
