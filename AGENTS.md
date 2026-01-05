# Repository Guidelines

## Project Structure & Module Organization
Tree-Mendous is a Python package with optional C++/GPU/Metal backends.

- `treemendous/`: core package.
  - `treemendous/basic/`: pure-Python interval trees and protocols.
  - `treemendous/cpp/`: pybind11 extensions; `gpu/` and `metal/` for platform builds.
  - `treemendous/backend_manager.py`: backend selection helpers.
- `tests/`: `tests/unit/` for protocol/unified checks, `tests/performance/` for benchmarks, and `tests/test_*_simple.py` for quick smoke tests.
- `examples/`, `docs/`, `scripts/`: demos, docs, release utilities.

## Build, Test, and Development Commands
- `uv sync` installs base deps; `uv sync --all-extras` enables dev/profiling/visualization.
- `just install` / `just install-dev` mirror the `uv` setup.
- `just build` builds the wheel and compiles C++ extensions; `just build-cpp` forces a clean rebuild.
- `just build-metal` (macOS) or `just build-gpu` (CUDA) for platform backends.
- `just test` runs pytest + unified validation; `just test-hypothesis` runs property-based tests.
- `just test-perf` or `just test-perf-full` for benchmarks; `just validate` runs tests plus bytecode checks.

## Coding Style & Naming Conventions
- Python, 4-space indentation, type hints where practical.
- Public classes use `PascalCase`; functions/variables use `snake_case`.
- Keep docstrings on public classes/functions; follow existing patterns in `treemendous/basic/`.

## Testing Guidelines
- Frameworks: `pytest` + `hypothesis` (see `tests/unit/hypothesis/`).
- Test files follow `test_*.py` or `*_simple.py`.
- Unified implementation checks live in `tests/unit/test_unified_implementations.py`; run `just test` before PRs.
- Performance tests are optional and can be slow; use when changing algorithms or C++ backends.

## Commit & Pull Request Guidelines
- Commit subjects generally follow Conventional Commits: `feat:`, `fix:`, `chore:`, `build:`; release commits use `Release vX.Y.Z`.
- PRs should include: a concise summary, relevant commands run (e.g., `just test`), and any performance numbers if behavior changes.
- If touching C++/GPU/Metal code, note the platform tested and build command used.

## Configuration & Optional Backends
- Boost ICL build: `TREE_MENDOUS_WITH_ICL=1` via `just build-cpp-icl`.
- CUDA build: `WITH_CUDA=1` via `just build-gpu`; Metal build uses `setup_metal.py`.
- Python version: `pyproject.toml` requires Python >=3.11 (use 3.11+ for development).
