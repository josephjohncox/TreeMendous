# Tree-Mendous

Interval tree implementations in Python and C++ for resource scheduling and space management.

## Implementations

### Python (`treemendous.basic`)
- **AVL Tree**: Self-balancing binary search tree with interval merging
- **Boundary Manager**: SortedDict-based interval tracking  
- **Segment Tree**: Traditional segment tree with lazy propagation
- **Treap**: Randomized tree-heap with probabilistic balancing
- **Summary Tree**: Enhanced tree with aggregate statistics (utilization, fragmentation, largest available)

### C++ (`treemendous.cpp`)
- **Boundary Manager**: std::map-based implementation (compiled ✅)
- **Treap**: High-performance randomized tree-heap (source available)
- **IC Manager**: Boost Interval Container Library implementation (source available)
- **Summary Boundary**: Enhanced versions with comprehensive statistics (source available)

**C++ Module Status:**
- ✅ **Boundary Manager**: Compiled and available (2-30x performance boost)
- 🔧 **Treap, Summary, IC**: Source available, requires compilation (`just build-cpp`)
- 📋 **Check Status**: `just list-backends` shows all available implementations

## Basic Usage

```python
# Summary-enhanced tree with comprehensive analytics
from treemendous.basic.summary import SummaryIntervalTree
tree = SummaryIntervalTree()
tree.release_interval(0, 1000)
stats = tree.get_availability_stats()
best_fit = tree.find_best_fit(50)

# Randomized treap with probabilistic balancing  
from treemendous.basic.treap import IntervalTreap
treap = IntervalTreap()
treap.release_interval(0, 1000)
sample = treap.sample_random_interval()
overlaps = treap.find_overlapping_intervals(100, 200)

# Backend-agnostic usage (auto-selects best available)
from examples.common.backend_config import create_interval_tree
tree = create_interval_tree("auto")        # Auto-select best backend
tree = create_interval_tree("py_treap")    # Force Python treap  
tree = create_interval_tree("cpp_boundary") # Force C++ boundary
```

## Installation

```bash
# Basic installation
uv sync

# With profiling tools (flameprof, snakeviz, memory-profiler)
uv sync --extra profiling

# With visualization tools (matplotlib, numpy, jupyter)
uv sync --extra visualization

# Full development environment
uv sync --all-extras
```

### Performance Profiling

```bash
# Complete profiling - Python + C++ implementations
just profile

# Quick Python vs C++ performance comparison  
just benchmark
# Output includes:
#   Python Boundary:         21.90 ms (baseline)
#   C++ Boundary:             3.74 ms (5.9x faster)
#   C++ Treap:                5.65 ms (81.7x faster than Python Treap)

# Full performance test suite
just test-perf

# Generate flame graphs from Python profiles
just flamegraph

# Visual profiler with ASCII charts
uv run python tests/performance/visual_profiler.py compare
```

**C++ Flame Graphs (Native Frame Capture):**
```bash
# Install py-spy (shows both Python and C++ frames)
uv sync --extra profiling

# Generate C++ flame graph with native code profiling
py-spy record --native -o cpp_flame.svg -- uv run python tests/performance/cpp_profiler.py

# View the flame graph
open cpp_flame.svg  # or upload to https://www.speedscope.app/
```

## Testing & Validation

Tree-Mendous includes a **unified testing framework** that automatically discovers and validates all implementations:

```bash
# Unified cross-implementation testing  
just test               # Complete test suite (197 tests, 8 implementations)
just test-unified       # Cross-implementation validation (64 passed, 34 skipped)

# Implementation-specific tests
just test-treap         # Treap-specific tests  
just test-boundary-summary  # Boundary summary tests
just test-hypothesis    # Property-based testing
```

The unified testing system discovers **8 implementations** automatically:
- **Python** (5): AVL, Boundary, Summary, Treap, Boundary Summary
- **C++** (3): Boundary, Treap, Boundary Summary
- **Specialized**: ICL variants (when compiled with `just build-cpp-full`)

## Performance Testing

```bash
# Comprehensive backend comparison
just test-perf             # All available backends
just test-perf-treap       # Treap-specific benchmarks
just benchmark-backends    # Compare all backends

# Scale testing
just test-perf-large       # Large-scale tests (100s of MB, ~10+ min)
just test-perf-large-quick # Quick large-scale tests (1-50MB)

# Simple testing (no dependencies)
just test-perf-simple      # Summary tree + treap tests
just validate              # Full validation pipeline

# Backend switching demos
just list-backends         # Show available backends
just demo-backends-treap   # Demo with Python treap
just demo-backends-cpp     # Demo with C++ boundary
```



## Development Commands

```bash
just install              # Install dependencies
just test                 # Run tests
just build                # Build package
just clean                # Clean build artifacts
just status               # Show project status
```

## Requirements

- Python 3.9+
- sortedcontainers, sortedcollections
- Optional: Boost ICL for C++ implementations
