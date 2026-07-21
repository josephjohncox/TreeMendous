# Tree-Mendous build, test, benchmark, and release-verification commands.

default: install

install:
    uv sync

install-dev:
    uv sync --all-extras

build: install-dev
    rm -rf dist
    uv build
    uv run python scripts/verify_artifact_contents.py dist

clean-cpp:
    find build -mindepth 1 -maxdepth 1 ! -name benchmarks -exec rm -rf {} + 2>/dev/null || true
    rm -rf treemendous/cpp/*.so treemendous/__pycache__ treemendous/basic/__pycache__ treemendous/cpp/__pycache__

build-cpp: install-dev clean-cpp
    uv run python setup.py build_ext --inplace

# Developer-only host instruction tuning; never used for release wheels.
build-cpp-native: install-dev clean-cpp
    TREE_MENDOUS_LOCAL_NATIVE=1 uv run python setup.py build_ext --inplace

build-cpp-icl: install-dev clean-cpp
    TREE_MENDOUS_WITH_ICL=1 uv run python setup.py build_ext --inplace

build-gpu: install-dev clean-cpp
    WITH_CUDA=1 uv run python setup_gpu.py build_ext --inplace

build-metal: install-dev clean-cpp
    uv run python setup_metal.py build_ext --inplace

clean-gpu:
    rm -rf treemendous/cpp/gpu/*.o treemendous/cpp/gpu/*.so treemendous/cpp/gpu/__pycache__

clean-metal:
    rm -rf treemendous/cpp/metal/*.o treemendous/cpp/metal/*.so treemendous/cpp/metal/__pycache__
    rm -rf treemendous/cpp/metal/*.air treemendous/cpp/metal/*.metallib
    rm -rf treemendous/cpp/metal/resources

verify-artifacts: install-dev
    uv run python scripts/verify_artifact_contents.py dist wheelhouse

test: install-dev
    uv run pytest
    just test-stable-backends

test-hypothesis: install-dev
    uv run pytest tests/unit/hypothesis/ -v

test-stable-backends: install-dev
    uv run pytest tests/unit/test_stable_backends.py -v --tb=short

backend-status: install-dev
    uv run python -c 'from treemendous import BackendRegistry; registry = BackendRegistry.discover(); print("\n".join(f"{spec.id}: {registry.states[spec.id]}" for spec in registry.specs))'

benchmark-smoke output="build/benchmarks/smoke.json": build-cpp
    uv run python -m tests.performance.benchmark_suite --profile smoke --require-all-stable --output "{{output}}"

benchmark-standard output="build/benchmarks/standard.json": build-cpp
    uv run python -m tests.performance.benchmark_suite --profile standard --require-all-stable --output "{{output}}"

benchmark-attribution baseline_root candidate_root="." output="build/benchmarks/mutation-attribution.json": install-dev
    uv run python -m tests.performance.mutation_attribution \
        --baseline-root "{{baseline_root}}" \
        --candidate-root "{{candidate_root}}" \
        --samples 20 \
        --warmups 1 \
        --output "{{output}}"

verify-attribution artifact="build/benchmarks/mutation-attribution.json": install-dev
    uv run python scripts/verify_mutation_attribution.py "{{artifact}}"

gate-attribution artifact primary_ratio_limit regression_ratio_limit control_ratio_minimum control_ratio_maximum require_samples: install-dev
    uv run python scripts/verify_mutation_attribution.py \
        "{{artifact}}" \
        --gate \
        --expected-primary-ratio-limit "{{primary_ratio_limit}}" \
        --expected-regression-ratio-limit "{{regression_ratio_limit}}" \
        --expected-control-ratio-minimum "{{control_ratio_minimum}}" \
        --expected-control-ratio-maximum "{{control_ratio_maximum}}" \
        --require-samples "{{require_samples}}"

benchmark-applications-smoke output="build/benchmarks/applications-smoke.json": install-dev
    uv run python -m tests.performance.application_benchmark_suite --profile smoke --output "{{output}}"

benchmark-applications-standard output="build/benchmarks/applications-standard.json": install-dev
    uv run python -m tests.performance.application_benchmark_suite --profile standard --output "{{output}}"

benchmark-large output="build/benchmarks/large.json": build-cpp
    uv run python -m tests.performance.benchmark_suite --profile large --require-all-stable --output "{{output}}"

test-gpu: install-dev
    uv run python tests/performance/gpu_benchmark.py

test-gpu-quick: install-dev
    uv run python tests/performance/gpu_benchmark.py --operations 100 --intervals 16

test-metal: install-dev
    uv run python tests/performance/metal_benchmark.py

test-metal-quick: install-dev
    uv run python tests/performance/metal_benchmark.py --operations 100 --intervals 16

benchmark: benchmark-standard benchmark-applications-standard

benchmark-gpu-large: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 10000 --operations 5000

benchmark-gpu-quick: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 16 --operations 100

benchmark-gpu-focused: install-dev
    uv run python tests/performance/gpu_benchmark.py --intervals 1000 --operations 5000

benchmark-batch backend="metal_boundary_summary": install-dev
    uv run python tests/performance/batch_operations_benchmark.py --backend {{backend}}

check-layout: install-dev
    uv run pytest tests/packaging/test_repository_layout.py -q

check-scenarios: install-dev
    uv run python scripts/generate_scenario_catalog.py --check
    uv run pytest tests/applications/test_registry.py -q

check: install-dev check-layout check-scenarios
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy treemendous
    uv run pytest -m "not benchmark and not cuda and not metal and not icl" --cov=treemendous --cov-branch --cov-config=pyproject.toml --cov-report=term-missing
    uv run pytest tests/packaging/test_artifact_policy.py tests/docs -q
    uv run python -m compileall -q treemendous tests scripts setup.py setup_gpu.py setup_metal.py

validate: test check
    @echo "Tree-Mendous validation complete"

run-examples: install-dev
    #!/usr/bin/env bash
    set -euo pipefail
    root="$PWD"
    examples=()
    while IFS= read -r example; do
        examples+=("$example")
    done < <(
        find examples -type f -name '*.py' -print | LC_ALL=C sort
    )
    for example in "${examples[@]}"; do
        echo "==> $example"
        (cd /tmp && uv run --project "$root" python "$root/$example")
    done

version:
    @python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

release-status:
    #!/usr/bin/env bash
    current_version=$(just version)
    current_branch=$(git branch --show-current)
    latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "No tags")
    echo "version: $current_version"
    echo "branch: $current_branch"
    echo "latest tag: $latest_tag"
    git status --short

help:
    @just --list
