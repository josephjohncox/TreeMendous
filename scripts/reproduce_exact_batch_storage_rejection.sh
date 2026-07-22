#!/usr/bin/env bash
# Rebuild the historical vector/segmented N64 smoke comparison from source.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="${1:-$ROOT/build/experiments/exact-batch-storage-segmented-reproduced.json}"
if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$PWD/$OUTPUT"
fi
BASELINE_COMMIT="2a384f74d29949fefd0286a147b30c1ef0a190d4"
EXPECTED_SOURCE_SHA256="f5a368f011bcbbe9f49ba954b7014268ab7c711ecfb1d46b8b4d9da6a8858267"
PATCH="$ROOT/tests/performance/experiments/fixtures/exact_batch_segmented_tuned.patch"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 2
fi

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/treemendous-storage-reproduction.XXXXXX")"
BASELINE="$TEMP_ROOT/baseline"
CANDIDATE="$TEMP_ROOT/candidate"
cleanup() {
    set +e
    git -C "$ROOT" worktree remove --force "$CANDIDATE" >/dev/null 2>&1
    git -C "$ROOT" worktree remove --force "$BASELINE" >/dev/null 2>&1
    rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT INT TERM

git -C "$ROOT" worktree add --detach "$BASELINE" "$BASELINE_COMMIT"
git -C "$ROOT" worktree add --detach "$CANDIDATE" "$BASELINE_COMMIT"
git -C "$CANDIDATE" apply "$PATCH"

ACTUAL_SOURCE_SHA256="$($PYTHON - "$CANDIDATE/treemendous/cpp/exact_batch_bindings.cpp" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "$ACTUAL_SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]]; then
    echo "Patched source SHA-256 mismatch: $ACTUAL_SOURCE_SHA256" >&2
    exit 1
fi
printf 'proved patched source SHA-256: %s\n' "$ACTUAL_SOURCE_SHA256"

(
    cd "$BASELINE"
    "$PYTHON" setup.py build_ext --inplace --force
)
(
    cd "$CANDIDATE"
    "$PYTHON" setup.py build_ext --inplace --force
)

mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT"
"$PYTHON" -m tests.performance.experiments.exact_batch_storage_matrix \
    --baseline-root "$BASELINE" \
    --candidate-root "$CANDIDATE" \
    --profile smoke \
    --blocks 20 \
    --output "$OUTPUT"
