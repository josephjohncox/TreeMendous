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

# Performance Testing
test-perf: install-dev
    uv run python tests/performance/comprehensive_benchmark.py

test-perf-full: install-dev
    timeout 600 uv run python tests/performance/comprehensive_benchmark.py

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
    @echo "  install        - Install dependencies"
    @echo "  build-cpp      - Build all C++ extensions (with clean)"
    @echo "  build-cpp-icl  - Build with Boost ICL support"
    @echo "  test           - Run complete test suite"
    @echo "  test-unified   - Cross-implementation validation" 
    @echo "  test-perf      - Performance benchmarks"
    @echo "  validate       - Quick validation"
    @echo "  run-examples   - Run key examples"
    @echo "  clean-cpp      - Clean C++ build artifacts"
