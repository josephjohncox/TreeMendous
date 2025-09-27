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
- **Boundary Manager**: std::map-based implementation
- **IC Manager**: Boost Interval Container Library implementation
- Summary-enhanced versions of both with comprehensive statistics

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
```

## Installation

```bash
uv sync
```

## Performance Testing

```bash
# Using Just commands (recommended)
just test-perf-simple      # No dependencies required
just validate              # Quick validation
just test-perf             # Full benchmark (requires uv sync)
just test-perf-large       # Large-scale tests (100s of MB, ~10+ min)
just test-perf-large-quick # Quick large-scale tests (1-50MB)

# Direct commands
python tests/test_summary_simple.py
python tests/performance/simple_benchmark.py
python tests/performance/comprehensive_benchmark.py  # requires uv sync
python tests/performance/large_scale_benchmark.py    # full large-scale tests
python tests/performance/large_scale_benchmark.py --quick  # quick mode
```

## Examples

Comprehensive examples in [`examples/`](examples/) demonstrate practical applications:
- **[Randomized Algorithms](examples/randomized_algorithms/)** - Treaps, Monte Carlo optimization, probabilistic scheduling
- **[CP-SAT Applications](examples/cp_sat_applications/)** - Job shop scheduling, constraint programming integration  
- **[Deadline Scheduling](examples/deadline_scheduling/)** - Real-time systems, EDF, response time analysis
- **[Bellman Iteration](examples/bellman_iteration/)** - Queue networks, dynamic programming, RL integration
- **[Practical Applications](examples/practical_applications/)** - Manufacturing, cloud computing, supply chain

```bash
# Run all examples
just run-examples

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
- **[Mathematical Analysis](MATHEMATICAL_ANALYSIS.md)** - Category theory, complexity analysis, algorithmic foundations
- **[Temporal Algebras](docs/TEMPORAL_ALGEBRAS_SCHEDULING.md)** - Process calculi, scheduling theory, temporal logic
- **[Real-Time Systems](docs/REALTIME_SYSTEMS_THEORY.md)** - Timing analysis, schedulability, fault tolerance
- **[Optimization Theory](docs/OPTIMIZATION_CP_SAT.md)** - Convex optimization, constraint programming, SAT solving
- **[Randomized Algorithms](docs/RANDOMIZED_ALGORITHMS.md)** - Probabilistic methods, stochastic optimization, online algorithms
- **[Queuing Theory](docs/QUEUING_THEORY_OPTIMIZATION.md)** - Stochastic queues, Bellman optimization, RL for queue control

## Requirements

- Python 3.9+
- sortedcontainers, sortedcollections
- Optional: Boost ICL for C++ implementations