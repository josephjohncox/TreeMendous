# Tree-Mendous Just Commands (using uv)

# Install dependencies
install:
    uv sync --all-extras

# Install development dependencies  
install-dev:
    uv sync --dev

# Clean build artifacts
clean:
    rm -rf build
    rm -rf dist
    rm -rf treemendous.egg-info
    rm -rf .uv_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Build the package
build:
    uv build

# Build C++ extensions (treap only)
build-cpp-treap: install
    @echo "🔧 Building C++ treap extension..."
    BUILD_TREAP=1 BUILD_SUMMARY=0 uv run python build.py
    @echo "✅ C++ treap built"

# Build C++ extensions (boundary only - safe fallback)
build-cpp-boundary: install
    @echo "🔧 Building C++ boundary extension..."
    BUILD_TREAP=0 BUILD_SUMMARY=0 BUILD_BOUNDARY_SUMMARY=0 uv run python build.py
    @echo "✅ C++ boundary built"

# Build C++ boundary summary extension  
build-cpp-boundary-summary: install
    @echo "🔧 Building C++ boundary summary extension..."
    BUILD_TREAP=0 BUILD_SUMMARY=0 BUILD_BOUNDARY_SUMMARY=1 uv run python build.py
    @echo "✅ C++ boundary summary built"

# Build C++ extensions (summary only)
build-cpp-summary: install
    @echo "🔧 Building C++ summary extensions..."
    BUILD_TREAP=0 BUILD_SUMMARY=1 uv run python build.py
    @echo "✅ C++ summary extensions built"

# Build all C++ extensions
build-cpp: install
    @echo "🔧 Building all C++ extensions..."
    BUILD_TREAP=1 BUILD_SUMMARY=1 BUILD_BOUNDARY_SUMMARY=1 uv run python build.py
    @echo "✅ All C++ extensions built"

# Build C++ extensions with Boost ICL support
build-cpp-full: install
    @echo "🔧 Building C++ extensions with Boost ICL..."
    BUILD_TREAP=1 BUILD_SUMMARY=1 TREE_MENDOUS_WITH_ICL=1 uv run python build.py
    @echo "✅ C++ extensions built with ICL support"

# Run tests with pytest
test:
    uv run pytest

# Run hypothesis tests specifically
test-hypothesis:
    uv run pytest tests/unit/hypothesis/ -v

# Run treap tests specifically
test-treap:
    uv run pytest tests/unit/hypothesis/test_treap.py -v
    python tests/unit/hypothesis/test_treap_cpp.py

# Run boundary summary tests specifically
test-boundary-summary:
    uv run pytest tests/unit/hypothesis/test_boundary_summary.py -v
    uv run python tests/test_boundary_summary_simple.py

# Run simple performance tests (no dependencies)
test-perf-simple:
    python tests/test_summary_simple.py
    python tests/performance/simple_benchmark.py

# Run comprehensive performance tests (requires dependencies)
test-perf: install
    uv run python tests/performance/comprehensive_benchmark.py

# Run large-scale performance tests (100s of MB, long duration)
test-perf-large:
    python tests/performance/large_scale_benchmark.py

# Run large-scale performance tests in quick mode (smaller scales)
test-perf-large-quick:
    python tests/performance/large_scale_benchmark.py --quick

# Run treap-specific performance tests
test-perf-treap:
    python tests/performance/treap_benchmark.py

# Run legacy performance comparison
test-perf-legacy: install
    uv run python tests/performance/boundry_vs_avl.py

# Get current version from pyproject.toml
version:
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Bump version and create release
release: build
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" | xargs -I {} gh release create v{} --generate-notes

# Delete release and tag
delete-release: delete-tag
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" | xargs -I {} gh release delete v{}

delete-tag:
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" | xargs -I {} git tag -d v{}
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" | xargs -I {} git push origin :refs/tags/v{}

# Development commands
dev-setup: install-dev
    echo "Development environment ready"

# Check code style and types (if tools are available)
check:
    uv run python -m py_compile treemendous/basic/*.py
    uv run python -m py_compile tests/test_summary_simple.py

# Quick validation
validate: test-perf-simple test-treap-simple check
    echo "✅ Tree-Mendous validation complete"

# Simple treap test (no dependencies)
test-treap-simple:
    python tests/test_treap_simple.py

# Run examples with auto backend selection
run-examples:
    @echo "🚀 Running Tree-Mendous Examples (auto backend)..."
    python examples/randomized_algorithms/treap_implementation.py
    @echo ""
    python examples/deadline_scheduling/realtime_scheduler.py --backend=auto
    @echo ""
    python examples/bellman_iteration/queue_network_optimization.py

# Run examples with specific backend
run-examples-with-backend backend:
    @echo "🚀 Running Tree-Mendous Examples ({{backend}} backend)..."
    python examples/cp_sat_applications/job_shop_scheduling.py --backend={{backend}}
    @echo ""
    python examples/deadline_scheduling/realtime_scheduler.py --backend={{backend}}

# Show backend comparison (comprehensive)
demo-backend-comparison:
    python examples/backend_comparison_demo.py

# List available backends  
list-backends:
    python examples/backend_switching_demo.py --list-backends

# Benchmark backends
benchmark-backends:
    python examples/backend_comparison_demo.py --benchmark-backends

# Run specific example category
run-examples-randomized:
    python examples/randomized_algorithms/treap_implementation.py

run-examples-cp-sat:
    python examples/cp_sat_applications/job_shop_scheduling.py

run-examples-deadline:
    python examples/deadline_scheduling/realtime_scheduler.py

run-examples-bellman:
    python examples/bellman_iteration/queue_network_optimization.py

run-examples-manufacturing:
    python examples/practical_applications/manufacturing_scheduler.py

run-examples-cloud:
    python examples/practical_applications/cloud_resource_manager.py

# Backend switching demonstrations
demo-backends:
    python examples/backend_switching_demo.py

demo-backends-treap:
    python examples/backend_switching_demo.py --backend=py_treap

demo-backends-cpp:
    python examples/backend_switching_demo.py --backend=cpp_boundary

demo-backends-summary:
    python examples/backend_switching_demo.py --backend=py_summary

list-backends-switch:
    python examples/backend_switching_demo.py --list-backends

benchmark-backends-switch:
    python examples/backend_switching_demo.py --benchmark-backends

# Show project status
status:
    @echo "Tree-Mendous Status:"
    @echo "  Version: $(just version)"
    @echo "  Python: $(python --version)"
    @echo "  uv: $(uv --version)"
    @echo "  Git: $(git rev-parse --short HEAD)"
    @echo "  Examples: $(find examples/ -name '*.py' | wc -l | tr -d ' ') Python files"

