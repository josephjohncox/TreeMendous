# Releasing

`pyproject.toml` is the only version source. The package reads installed metadata
for `treemendous.__version__`, and the release workflow rejects a tag that does
not equal `v<project.version>`. The public API introduced on this branch is the
`0.3.0` minor release; `v0.2.5` already exists and must not be reused.

1. Run `just validate`, `uv lock --check`, and the platform/package matrix.
2. Inspect source and wheel contents with `just verify-artifacts`.
3. Update the project version with `just bump-version patch` (or `minor`/`major`)
   and review the diff.
4. Create the signed review/release evidence, then publish a GitHub release with
   the matching `vX.Y.Z` tag.

The release workflow builds exactly one sdist, builds the Python 3.11–3.13 wheel
matrix, performs artifact and arbitrary-CWD clean-install checks, aggregates all
artifacts, and then enables one trusted PyPI publisher. No OS-specific job has
independent upload authority.

Metal acceptance requires macOS SDK/architecture/device evidence. CUDA remains
experimental unless its separate hardware parity and compute-sanitizer evidence
is attached; a successful build alone is insufficient.
