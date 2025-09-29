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

## 📚 Documentation & Visualizations

### Core Documentation
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - One-page guide to get started instantly
- **[Architecture Guide](docs/ARCHITECTURE_GUIDE.md)** - Deep dive into design patterns and mathematical foundations
- **[Interval Tree Visualizations](docs/INTERVAL_TREE_VISUALIZATION.md)** - Visual explanations of data structures and algorithms
- **[Interactive Notebook](docs/interactive_visualization.ipynb)** - Jupyter notebook with matplotlib visualizations

### Practical Examples
- **[Tree Structure Demo](examples/visualizations/tree_structure_demo.py)** - Interactive demo of all implementations
- **[Algorithm Analysis](examples/visualizations/algorithm_analysis.py)** - Performance and mathematical analysis
- **[Backend Comparison](examples/backend_comparison_demo.py)** - Cross-implementation validation

### Run Visualizations
```bash
# Interactive demos
uv run python examples/visualizations/tree_structure_demo.py
uv run python examples/visualizations/algorithm_analysis.py

# Jupyter notebook
jupyter notebook docs/interactive_visualization.ipynb
```

### Performance Profiling
- **[Performance Profiling Guide](docs/PERFORMANCE_PROFILING.md)** - Python profiling and optimization
- **[C++ Profiling Guide](docs/CPP_PROFILING_GUIDE.md)** - C++ profiling with native frame capture

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

## Examples

Comprehensive examples in [`examples/`](examples/) demonstrate practical applications:
- **[Randomized Algorithms](examples/randomized_algorithms/)** - Treaps, Monte Carlo optimization, probabilistic scheduling
- **[CP-SAT Applications](examples/cp_sat_applications/)** - Job shop scheduling, constraint programming integration  
- **[Deadline Scheduling](examples/deadline_scheduling/)** - Real-time systems, EDF, response time analysis
- **[Bellman Iteration](examples/bellman_iteration/)** - Queue networks, dynamic programming, RL integration
- **[Practical Applications](examples/practical_applications/)** - Manufacturing, cloud computing, supply chain

```bash
# Run all examples (auto backend selection)
just run-examples

# Run examples with specific backend
just run-examples-with-backend py_summary    # Python summary trees
just run-examples-with-backend cpp_summary   # C++ summary trees (fastest)
just run-examples-with-backend py_treap      # Python randomized treaps

# Backend management
just list-backends          # Show available implementations
just benchmark-backends     # Performance comparison
just demo-backends          # Backend switching demonstration

# Run specific categories  
just run-examples-randomized
just run-examples-deadline
just run-examples-bellman
```

## Development Commands

```bash
just install              # Install dependencies
just test                 # Run tests
just build                # Build package
just clean                # Clean build artifacts
just status               # Show project status
```

## Mathematical Documentation

Comprehensive theoretical analysis available in [`docs/`](docs/):
- **[Mathematical Analysis](docs/MATHEMATICAL_ANALYSIS.md)** - Category theory, complexity analysis, algorithmic foundations
- **[Temporal Algebras](docs/TEMPORAL_ALGEBRAS_SCHEDULING.md)** - Process calculi, scheduling theory, temporal logic
- **[Real-Time Systems](docs/REALTIME_SYSTEMS_THEORY.md)** - Timing analysis, schedulability, fault tolerance
- **[Optimization Theory](docs/OPTIMIZATION_CP_SAT.md)** - Convex optimization, constraint programming, SAT solving
- **[Randomized Algorithms](docs/RANDOMIZED_ALGORITHMS.md)** - Probabilistic methods, stochastic optimization, online algorithms
- **[Queuing Theory](docs/QUEUING_THEORY_OPTIMIZATION.md)** - Stochastic queues, Bellman optimization, RL for queue control

## Requirements

- Python 3.9+
- sortedcontainers, sortedcollections
- Optional: Boost ICL for C++ implementations