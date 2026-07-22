# Tree-Mendous build, test, benchmark, and release-verification commands.

default: install

install:
    uv sync

install-dev:
    uv sync --all-extras

build: install-dev
    rm -rf dist
    uv build
    uv run python -m scripts.verify_artifact_contents dist

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
    uv run python -m scripts.verify_artifact_contents dist wheelhouse

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

# Compare two explicitly supplied native roots. The shipped checkout remains on
# vector storage, so there is deliberately no current-tree candidate default.
experiment-exact-batch-storage-matrix baseline_root candidate_root output="build/experiments/exact-batch-storage-matrix.json": install-dev
    uv run python -m tests.performance.experiments.exact_batch_storage_matrix \
        --baseline-root "{{baseline_root}}" \
        --candidate-root "{{candidate_root}}" \
        --blocks 20 \
        --output "{{output}}"

verify-exact-batch-storage-matrix artifact="build/experiments/exact-batch-storage-matrix.json": install-dev
    uv run python -m tests.performance.experiments.exact_batch_storage_matrix \
        --verify \
        --output "{{artifact}}"

verify-exact-batch-storage-archive artifact="docs/evidence/experiments/exact-batch-storage-segmented-tuned-rejection.json": install-dev
    uv run python -m tests.performance.experiments.exact_batch_storage_matrix \
        --verify \
        --archive \
        --output "{{artifact}}"

# Intentionally expensive: two detached builds plus the historical 20-block smoke.
reproduce-exact-batch-storage-rejection output="build/experiments/exact-batch-storage-segmented-reproduced.json": install-dev
    scripts/reproduce_exact_batch_storage_rejection.sh "{{output}}"

# Bounded diagnostic only; this does not replace any stable exact-batch gate.
experiment-exact-batch-application-matrix profile="smoke" output="build/experiments/exact-batch-application-matrix.json": build-cpp
    uv run python -m tests.performance.experiments.exact_batch_application_matrix \
        --profile "{{profile}}" \
        --samples 10 \
        --output "{{output}}"

verify-exact-batch-application-matrix artifact="build/experiments/exact-batch-application-matrix.json": install-dev
    uv run python -m tests.performance.experiments.exact_batch_application_matrix \
        --verify \
        --output "{{artifact}}"

# Private RangeSet transaction candidate; "full" omits --bounded.
experiment-rangeset-transaction profile="bounded" output="build/experiments/rangeset-transaction.json": install-dev
    #!/usr/bin/env bash
    set -euo pipefail
    bounded=()
    if [[ "{{profile}}" == "bounded" ]]; then
        bounded+=(--bounded)
    elif [[ "{{profile}}" != "full" ]]; then
        echo "profile must be bounded or full" >&2
        exit 2
    fi
    uv run python -m tests.performance.experiments.rangeset_transaction \
        --samples 15 \
        --output "{{output}}" \
        "${bounded[@]}"

verify-rangeset-transaction artifact="build/experiments/rangeset-transaction.json": install-dev
    uv run python -m tests.performance.experiments.rangeset_transaction \
        --verify \
        --output "{{artifact}}"

experiment-rangeset-snapshot-scaling output="build/experiments/rangeset-snapshot-scaling.json": install-dev
    uv run python -m tests.performance.experiments.rangeset_snapshot_scaling \
        --blocks 40 \
        --output "{{output}}"

verify-rangeset-snapshot-scaling artifact="build/experiments/rangeset-snapshot-scaling.json": install-dev
    uv run python -m tests.performance.experiments.rangeset_snapshot_scaling \
        --verify \
        --output "{{artifact}}"

# Experimental concrete-application qualification; never changes runtime selection.
experiment-application-backend-matrix output="build/experiments/application-backend-matrix.json": build-cpp
    uv run python -m tests.performance.experiments.application_backend_matrix \
        --blocks 20 \
        --output "{{output}}"

verify-application-backend-matrix artifact="build/experiments/application-backend-matrix.json": install-dev
    uv run python -m tests.performance.experiments.application_backend_matrix \
        --verify \
        --output "{{artifact}}"

experiment-lease-state-scaling blocks="30" output="build/experiments/lease-state-scaling.json": install-dev
    uv run python -m tests.performance.experiments.lease_state_scaling \
        --blocks "{{blocks}}" \
        --output "{{output}}"

verify-lease-state-scaling artifact="build/experiments/lease-state-scaling.json": install-dev
    uv run python -m tests.performance.experiments.lease_state_scaling \
        --verify \
        --output "{{artifact}}"

# Experiment-only RadioSpectrumScheduler index injection; runtime remains linear
# unless every fixed training, memory, and held-out gate passes.
experiment-radio-spectrum-index-matrix output="build/experiments/radio-spectrum-index-matrix.json" timeout_seconds="3600": install-dev
    uv run python -c 'import subprocess, sys; subprocess.run([sys.executable, "-m", "tests.performance.experiments.radio_spectrum_index_matrix", "--profile", "full", "--blocks", "25", "--output", "{{output}}"], check=True, timeout=int("{{timeout_seconds}}"))'

verify-radio-spectrum-index-matrix artifact="build/experiments/radio-spectrum-index-matrix.json": install-dev
    uv run python -m tests.performance.experiments.radio_spectrum_index_matrix \
        --verify \
        --output "{{artifact}}"

benchmark-attribution baseline_root candidate_root="." output="build/benchmarks/mutation-attribution.json": install-dev
    uv run python -m tests.performance.mutation_attribution \
        --baseline-root "{{baseline_root}}" \
        --candidate-root "{{candidate_root}}" \
        --samples 20 \
        --warmups 1 \
        --output "{{output}}"

verify-attribution artifact="build/benchmarks/mutation-attribution.json": install-dev
    uv run python -m scripts.verify_mutation_attribution "{{artifact}}"

gate-attribution artifact primary_ratio_limit regression_ratio_limit control_ratio_minimum control_ratio_maximum require_samples: install-dev
    uv run python -m scripts.verify_mutation_attribution \
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
    uv run python -m tests.performance.gpu_benchmark

test-gpu-quick: install-dev
    uv run python -m tests.performance.gpu_benchmark --operations 100 --intervals 16

test-metal: install-dev
    uv run python -m tests.performance.metal_benchmark

test-metal-quick: install-dev
    uv run python -m tests.performance.metal_benchmark --operations 100 --intervals 16

benchmark: benchmark-standard benchmark-applications-standard

benchmark-gpu-large: install-dev
    uv run python -m tests.performance.gpu_benchmark --intervals 10000 --operations 5000

benchmark-gpu-quick: install-dev
    uv run python -m tests.performance.gpu_benchmark --intervals 16 --operations 100

benchmark-gpu-focused: install-dev
    uv run python -m tests.performance.gpu_benchmark --intervals 1000 --operations 5000

benchmark-batch backend="metal_boundary_summary": install-dev
    uv run python -m tests.performance.batch_operations_benchmark --backend {{backend}}

# Generate and strictly verify the cpp_boundary hot-path evidence triplet.
# Verification requires a clean worktree and the loaded native binary.
benchmark-hotpath output="build/benchmarks/rangeset-hotpath.json" samples="30" target_operations="20000": build-cpp
    uv run python -m tests.performance.rangeset_hotpath_benchmark --samples {{samples}} --target-operations {{target_operations}} --output "{{output}}"
    uv run python -m scripts.verify_rangeset_hotpath_benchmark "{{output}}"

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
