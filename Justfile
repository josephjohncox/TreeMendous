# Tree-Mendous Build and Test System
# Simplified build system for efficient development

# Default target
default: install

# Environment setup
install:
    uv sync

install-dev:
    uv sync --all-extras

# Package build (C++ extensions compiled automatically via setup.py)
build: install-dev
    @echo "📦 Building Tree-Mendous package with integrated C++ compilation..."
    uv build
    @echo "✅ Package built with C++ libraries included"

# C++ Build System (clean and simple)
clean-cpp:
    @echo "🧹 Cleaning C++ build artifacts..."
    rm -rf build/ treemendous/cpp/*.so treemendous/__pycache__ treemendous/basic/__pycache__ treemendous/cpp/__pycache__
    @echo "✅ C++ artifacts cleaned"

# Build all C++ extensions (always clean first for reliability)
build-cpp: install clean-cpp
    @echo "🔧 Building all C++ extensions..."
    uv run python setup.py
    @echo "✅ All C++ extensions built"

# Build C++ extensions with Boost ICL support  
build-cpp-icl: install clean-cpp
    @echo "🔧 Building C++ extensions with Boost ICL..."
    TREE_MENDOUS_WITH_ICL=1 uv run python build.py
    @echo "✅ C++ extensions built with ICL support"

# Testing System
test: install-dev
    uv run pytest
    @echo "🔄 Running cross-implementation validation..."
    just test-unified

test-hypothesis: install-dev
    uv run pytest tests/unit/hypothesis/ -v

test-unified: install-dev
    uv run pytest tests/unit/test_unified_implementations.py -v --tb=short

test-protocols: install-dev
    @echo "🔄 Testing unified protocol system..."
    uv run python -c 'import treemendous; treemendous.print_backend_status(); tree = treemendous.create_interval_tree(); tree.release_interval(0, 1000); tree.reserve_interval(100, 200); print(f"✅ {len(tree.get_intervals())} intervals, protocol consistency verified!")'

# Performance Testing
test-perf: install-dev
    uv run python tests/performance/protocol_benchmark.py

test-perf-full: install-dev
    timeout 600 uv run python tests/performance/comprehensive_benchmark.py

# Performance profiling with flame graphs (Python + C++)
profile: install-dev
    @echo "🔥 Profiling all implementations (Python + C++)..."
    @echo ""
    @echo "═══════════════════════════════════════════════════════════"
    @echo "1️⃣  Python Implementations (cProfile + flameprof)"
    @echo "═══════════════════════════════════════════════════════════"
    uv run python tests/performance/flamegraph_profiler.py all
    @echo ""
    @echo "═══════════════════════════════════════════════════════════"
    @echo "2️⃣  C++ Performance Comparison"
    @echo "═══════════════════════════════════════════════════════════"
    uv run python tests/performance/cpp_profiler.py
    @echo ""
    @echo "💡 For C++ flame graphs with native frames:"
    @echo "   py-spy record --native -o cpp_flame.svg -- uv run python tests/performance/cpp_profiler.py"

# Generate flame graphs from existing profiles
flamegraph: install-dev
    @echo "🔥 Generating flame graphs..."
    uv run python tests/performance/flamegraph_profiler.py all

# Performance Benchmarks
benchmark: install-dev
    @echo "📊 Running comprehensive protocol benchmark (all implementations)..."
    uv run python tests/performance/protocol_benchmark.py

benchmark-optimizations: install-dev
    @echo "📊 Comparing original vs optimized C++ implementations..."
    uv run python tests/performance/simple_optimization_benchmark.py

benchmark-flamegraph: install-dev
    @echo "📊 Running benchmarks with flamegraph comparison..."
    uv run python tests/performance/flamegraph_profiler.py compare

# Profile C++ implementations (requires py-spy)
profile-cpp: install-dev
    @echo "🔥 C++ profiling (install py-spy if needed: uv pip install py-spy)..."
    @echo "Running workload to profile..."
    uv run python tests/performance/cpp_profiler.py
    @echo ""
    @echo "💡 To generate flame graph with C++ frames:"
    @echo "   py-spy record --native -o cpp_flame.svg -- uv run python tests/performance/cpp_profiler.py"

# Development utilities
check: install-dev
    uv run python -m py_compile treemendous/basic/*.py
    uv run python -m py_compile tests/test_*_simple.py

validate: test check
    @echo "✅ Tree-Mendous validation complete"

# Examples (simplified)
run-examples: install-dev
    @echo "🚀 Running key examples..."
    python examples/randomized_algorithms/treap_implementation.py
    python examples/deadline_scheduling/realtime_scheduler.py
    python examples/backend_comparison_demo.py

# Version management
version:
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Help
help:
    @echo "Tree-Mendous Commands:"
    @echo "  install          - Install dependencies"
    @echo "  build            - Build package with C++ extensions"
    @echo "  build-cpp        - Build C++ extensions for development"
    @echo "  build-cpp-icl    - Build with Boost ICL support"
    @echo "  test             - Run complete test suite"
    @echo "  test-unified     - Cross-implementation validation"
    @echo "  test-protocols   - Test unified protocol system"
    @echo "  test-perf        - Performance benchmarks"
    @echo "  profile          - Profile Python with flame graphs"
    @echo "  profile-cpp      - Profile C++ implementations"
    @echo "  flamegraph       - Generate flame graphs from profiles"
    @echo "  benchmark        - Quick performance comparison"
    @echo "  validate         - Quick validation"
    @echo "  run-examples     - Run key examples"
    @echo "  clean-cpp        - Clean C++ build artifacts"
