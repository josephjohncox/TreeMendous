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
build-cpp: install-dev clean-cpp
    @echo "🔧 Building all C++ extensions with portable CPU flags..."
    uv run python setup.py build_ext --inplace
    @echo "✅ All C++ extensions built"

# Explicit developer-only host instruction tuning; never used for wheels.
build-cpp-native: install-dev clean-cpp
    TREE_MENDOUS_LOCAL_NATIVE=1 uv run python setup.py build_ext --inplace

# Build C++ extensions with Boost ICL support  
build-cpp-icl: install-dev clean-cpp
    @echo "🔧 Building C++ extensions with Boost ICL..."
    TREE_MENDOUS_WITH_ICL=1 uv run python setup.py build_ext --inplace
    @echo "✅ C++ extensions built with ICL support"

# Build GPU/CUDA extensions (Linux/Windows)
build-gpu: install-dev clean-cpp
    @echo "🔧 Building GPU/CUDA extensions..."
    @echo "   Requires: CUDA Toolkit installed"
    WITH_CUDA=1 uv run python setup_gpu.py build_ext --inplace
    @echo "✅ GPU extensions built"

# Build Metal extensions (macOS)
build-metal: install-dev clean-cpp
    @echo "🍎 Building Metal/MPS extensions for macOS..."
    @echo "   Requires: Xcode Command Line Tools"
    uv run python setup_metal.py build_ext --inplace
    @echo "✅ Metal extensions built"

# Clean GPU build artifacts
clean-gpu:
    @echo "🧹 Cleaning GPU build artifacts..."
    rm -rf treemendous/cpp/gpu/*.o treemendous/cpp/gpu/*.so treemendous/cpp/gpu/__pycache__
    @echo "✅ GPU artifacts cleaned"

# Clean Metal build artifacts
clean-metal:
    @echo "🧹 Cleaning Metal build artifacts..."
    rm -rf treemendous/cpp/metal/*.o treemendous/cpp/metal/*.so treemendous/cpp/metal/__pycache__
    rm -rf treemendous/cpp/metal/*.air treemendous/cpp/metal/*.metallib
    rm -rf treemendous/cpp/metal/resources
    @echo "✅ Metal artifacts cleaned"

verify-artifacts: install-dev
    uv run python scripts/verify_artifact_contents.py dist wheelhouse

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
    uv run python tests/performance/protocol_benchmark.py --quick --validate

test-perf-full: install-dev
    timeout 600 uv run python tests/performance/protocol_benchmark.py --validate --operations 5000 --output benchmark.json

# GPU performance testing (CUDA - Linux/Windows)
test-gpu: install-dev
    @echo "🎮 Running GPU performance benchmarks..."
    uv run python tests/performance/gpu_benchmark.py

test-gpu-quick: install-dev
    @echo "🎮 Running correctness-checked experimental CUDA benchmark..."
    uv run python tests/performance/gpu_benchmark.py --operations 100 --intervals 16

# Metal performance testing (macOS)
test-metal: install-dev
    @echo "🍎 Running Metal performance benchmarks..."
    uv run python tests/performance/metal_benchmark.py

test-metal-quick: install-dev
    @echo "🍎 Running correctness-checked experimental Metal benchmark..."
    uv run python tests/performance/metal_benchmark.py --operations 100 --intervals 16

# Experimental accelerator benchmarks. All use identical oracle-validated traces.
benchmark-gpu-large: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 10000 --operations 5000

benchmark-gpu-quick: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 16 --operations 100

benchmark-gpu-focused: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 1000 --operations 5000

stress-gpu: install-dev
    @echo "stress-gpu was retired; use the validated experimental CUDA benchmark"
    uv run python tests/performance/gpu_benchmark.py --intervals 1000 --operations 5000

# Batch operations benchmark (realistic GPU use cases)
benchmark-batch backend="metal_boundary_summary": install-dev
    @echo "📦 Running semantics-preserving contiguous batch benchmark..."
    uv run python tests/performance/batch_operations_benchmark.py --backend {{backend}}

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
    @echo "📊 Running correctness-checked local directional benchmark..."
    uv run python tests/performance/protocol_benchmark.py --validate

benchmark-optimizations: install-dev
    @echo "benchmark-optimizations was retired; running the common validated harness"
    uv run python tests/performance/protocol_benchmark.py --validate

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

# Get current version (alias for compatibility)
get-version:
    @just version

# Show current version and next version options
version-info:
    #!/usr/bin/env bash
    current=$(just version)
    echo "📦 Current version: $current"
    echo ""
    echo "🔢 Next version options:"
    
    # Parse semantic version
    IFS='.' read -ra PARTS <<< "$current"
    major=${PARTS[0]}
    minor=${PARTS[1]}
    patch=${PARTS[2]}
    
    echo "  patch: $major.$minor.$((patch + 1))"
    echo "  minor: $major.$((minor + 1)).0"
    echo "  major: $((major + 1)).0.0"

# Bump version (patch, minor, or major)
bump-version type:
    #!/usr/bin/env bash
    set -e
    
    current=$(just version)
    echo "📦 Current version: $current"
    
    # Parse semantic version
    IFS='.' read -ra PARTS <<< "$current"
    major=${PARTS[0]}
    minor=${PARTS[1]}
    patch=${PARTS[2]}
    
    case "{{type}}" in
        "patch")
            new_version="$major.$minor.$((patch + 1))"
            ;;
        "minor")
            new_version="$major.$((minor + 1)).0"
            ;;
        "major")
            new_version="$((major + 1)).0.0"
            ;;
        *)
            echo "❌ Error: Version type must be 'patch', 'minor', or 'major'"
            exit 1
            ;;
    esac
    
    echo "🔄 Bumping version: $current → $new_version"
    
    # Update pyproject.toml
    sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
    rm pyproject.toml.bak
    
    # Update __init__.py if it exists
    if [[ -f "treemendous/__init__.py" ]]; then
        if grep -q "__version__" treemendous/__init__.py; then
            sed -i.bak "s/__version__ = \".*\"/__version__ = \"$new_version\"/" treemendous/__init__.py
            rm treemendous/__init__.py.bak
        else
            echo "__version__ = \"$new_version\"" >> treemendous/__init__.py
        fi
    fi
    
    echo "✅ Version bumped to $new_version"

# Create a release (fully automated with GitHub CLI)
release type="patch" message="":
    ./scripts/create-release.sh {{type}} "{{message}}"

# Manual tag-only release (for troubleshooting)
tag-release type="patch" message="":
    ./scripts/create-tag.sh {{type}} "{{message}}"

# Create a pre-release (alpha/beta/rc)
prerelease type="alpha" message="":
    #!/usr/bin/env bash
    set -e
    
    current=$(just version)
    echo "📦 Current version: $current"
    
    # Parse semantic version
    IFS='.' read -ra PARTS <<< "$current"
    major=${PARTS[0]}
    minor=${PARTS[1]}
    patch=${PARTS[2]}
    
    # Generate pre-release version
    timestamp=$(date +%Y%m%d%H%M)
    case "{{type}}" in
        "alpha")
            new_version="$major.$minor.$patch-alpha.$timestamp"
            ;;
        "beta")
            new_version="$major.$minor.$patch-beta.$timestamp"
            ;;
        "rc")
            new_version="$major.$minor.$patch-rc.$timestamp"
            ;;
        *)
            echo "❌ Error: Pre-release type must be 'alpha', 'beta', or 'rc'"
            exit 1
            ;;
    esac
    
    echo "🔄 Creating pre-release: $current → $new_version"
    
    # Update version temporarily
    sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
    
    # Build and test
    just test
    just build
    
    # Create pre-release commit and tag
    release_message="Pre-release v$new_version"
    if [[ -n "{{message}}" ]]; then
        release_message="$release_message: {{message}}"
    fi
    
    git add pyproject.toml
    git commit -m "$release_message"
    git tag -a "v$new_version" -m "$release_message"
    
    # Push
    git push origin $(git branch --show-current)
    git push origin "v$new_version"
    
    # Restore original version
    mv pyproject.toml.bak pyproject.toml
    git add pyproject.toml
    git commit -m "Restore version after pre-release"
    git push origin $(git branch --show-current)
    
    echo "✅ Pre-release v$new_version created!"

# Publish to PyPI (manual override - uses uv like CI/CD)
publish-pypi:
    #!/usr/bin/env bash
    set -e
    
    echo "📦 Publishing to PyPI..."
    
    # Build first using uv (same as CI/CD)
    echo "🔧 Building with uv..."
    uv build
    
    # Check if we have twine
    if ! command -v twine >/dev/null 2>&1; then
        echo "Installing twine..."
        uv add --dev twine
    fi
    
    # Upload to PyPI
    echo "🚀 Uploading to PyPI..."
    uv run twine upload dist/*
    
    echo "✅ Published to PyPI successfully!"

# Show release status
release-status:
    #!/usr/bin/env bash
    echo "📊 Release Status"
    echo "════════════════════════════════════════════════════════════════"
    
    current_version=$(just version)
    echo "📦 Current version: $current_version"
    
    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        echo "🔄 Working directory: DIRTY"
        git status --short
    else
        echo "✅ Working directory: CLEAN"
    fi
    
    # Show current branch
    current_branch=$(git branch --show-current)
    echo "🌿 Current branch: $current_branch"
    
    # Show latest tags
    echo ""
    echo "🏷️  Recent tags:"
    git tag --sort=-version:refname | head -5 || echo "No tags found"
    
    # Show commits since last tag
    latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "No tags")
    if [[ "$latest_tag" != "No tags" ]]; then
        echo ""
        echo "📝 Commits since $latest_tag:"
        git log --oneline "$latest_tag"..HEAD | head -10
    fi

# Help
help:
    @echo "Tree-Mendous Commands:"
    @echo ""
    @echo "📦 Build & Install:"
    @echo "  install          - Install dependencies"
    @echo "  build            - Build package with C++ extensions"
    @echo "  build-cpp        - Build C++ extensions for development"
    @echo "  build-cpp-icl    - Build with Boost ICL support"
    @echo "  build-gpu        - Build GPU/CUDA extensions (Linux/Windows)"
    @echo "  build-metal      - Build Metal/MPS extensions (macOS)"
    @echo "  clean-cpp        - Clean C++ build artifacts"
    @echo "  clean-gpu        - Clean GPU build artifacts"
    @echo "  clean-metal      - Clean Metal build artifacts"
    @echo ""
    @echo "🧪 Testing:"
    @echo "  test             - Run complete test suite"
    @echo "  test-unified     - Cross-implementation validation"
    @echo "  test-protocols   - Test unified protocol system"
    @echo "  test-perf        - Performance benchmarks"
    @echo "  test-gpu         - GPU performance benchmarks (CUDA)"
    @echo "  test-gpu-quick   - Quick GPU availability check (CUDA)"
    @echo "  test-metal       - Metal performance benchmarks (macOS)"
    @echo "  test-metal-quick - Quick Metal availability check (macOS)"
    @echo "  validate         - Quick validation"
    @echo ""
    @echo "📊 Profiling & Benchmarks:"
    @echo "  profile          - Profile Python with flame graphs"
    @echo "  profile-cpp      - Profile C++ implementations"
    @echo "  flamegraph       - Generate flame graphs from profiles"
    @echo "  benchmark        - Quick performance comparison"
    @echo ""
    @echo "🚀 GPU Large-Scale Benchmarks:"
    @echo "  benchmark-gpu-large   - Full scale test (10K-5M intervals)"
    @echo "  benchmark-gpu-quick   - Quick test (up to 500K intervals)"
    @echo "  benchmark-gpu-focused - GPU sweet spot (500K-5M intervals)"
    @echo "  benchmark-batch       - Batch operations (realistic GPU use cases)"
    @echo "  stress-gpu            - GPU stress test (sustained load, memory pressure)"
    @echo ""
    @echo "🚀 Release Management:"
    @echo "  version          - Show current version"
    @echo "  version-info     - Show version bump options"
    @echo "  bump-version TYPE - Bump version (patch|minor|major)"
    @echo "  release [TYPE] [MSG] - Create release (patch|minor|major)"
    @echo "  prerelease [TYPE] [MSG] - Create pre-release (alpha|beta|rc)"
    @echo "  release-status   - Show release status"
    @echo "  publish-pypi     - Publish to PyPI manually"
    @echo ""
    @echo "📝 Examples:"
    @echo "  run-examples     - Run key examples"
    @echo ""
    @echo "🔗 Release Examples:"
    @echo "  just release patch 'Bug fixes and improvements'"
    @echo "  just release minor 'New features added'"
    @echo "  just prerelease alpha 'Testing new functionality'"
