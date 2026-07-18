# Releasing

Releases use one trusted publishing authority. Platform jobs build artifacts;
the final release job verifies the complete set and publishes once.

## Versioning

The project follows semantic versioning:

- Patch: compatible fixes.
- Minor: compatible features.
- Major: breaking API or behavior changes.

Version `1.0.0` establishes `RangeSet` and `BackendRegistry` as the public API.
Earlier factory, manager, protocol, compatibility, and raw-backend exports were
removed rather than retained as shims.

## Pre-release checks

From a clean checkout:

```bash
uv sync --all-extras
uv lock --check
just check
just validate
just build
uv run python scripts/verify_artifact_contents.py dist/*.whl dist/*.tar.gz
```

`just check` enforces formatting, lint, typing, unit tests, and branch coverage.
`just validate` reruns the test contract and compiles Python sources.

## Platform artifacts

The release workflow builds wheels for the supported Python versions and
platforms. Native artifacts must contain:

- the canonical Python package,
- the stable C++ boundary extension,
- package-relative resources required by any included optional extension,
- no repository-local build products or generated benchmark output.

Every wheel is installed into an isolated environment and exercised through
the public `RangeSet` API before publication. The sdist is rebuilt and tested
without relying on files from the source checkout.

## Hardware gates

Experimental accelerator artifacts do not become stable merely because they
compile. Promotion requires parity on real hardware as described in
[Backend support](backends.md).

- Metal runs on the hosted macOS hardware lane.
- CUDA requires the optional self-hosted NVIDIA workflow.

A skipped CUDA lane does not promote or validate CUDA.

## Publish

1. Update `pyproject.toml` and regenerate `uv.lock`.
2. Ensure the release notes describe user-visible changes and removals.
3. Merge only after all required checks pass.
4. Create and push the matching tag, for example `v1.0.0`.
5. Let the release workflow aggregate, verify, and publish all artifacts once.

Do not upload artifacts manually from a platform build job. Do not run multiple
publishers for one release.

## Post-release verification

Install the published wheel in a clean environment and exercise the public API:

```bash
python -m venv /tmp/treemendous-release-check
/tmp/treemendous-release-check/bin/python -m pip install treemendous==1.0.0
/tmp/treemendous-release-check/bin/python - <<'PY'
from treemendous import Span, create_range_set

ranges = create_range_set((0, 100))
result = ranges.allocate(10, not_before=0)
assert result is not None and result.span == Span(0, 10)
assert tuple(item.span for item in ranges.intervals()) == (Span(10, 100),)
PY
```
