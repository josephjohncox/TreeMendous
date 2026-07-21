# Releasing

Releases use one trusted publishing authority. Platform jobs build artifacts;
the final release job verifies the complete set and publishes once.

## Versioning

The project follows semantic versioning:

- Patch: compatible fixes.
- Minor: compatible features.
- Major: breaking API or behavior changes.

Version `1.0.0` established `RangeSet` and `BackendRegistry` as the public API.
Earlier factory, manager, protocol, compatibility, and raw-backend exports were
removed rather than retained as shims. Version `1.1.0` compatibly adds the
contained `treemendous.exact_batch` API without root exports, backend
integration, or protocol integration.

## Pre-release checks

From a clean checkout:

```bash
uv sync --all-extras
uv lock --check
just check
just validate
just build
uv run python -m scripts.verify_artifact_contents dist/*.whl dist/*.tar.gz
just run-examples
```

`just check` enforces formatting, lint, typing, focused/full non-hardware tests,
documentation and packaging contracts, and branch coverage. `just validate`
reruns the test contract and compiles Python sources. For 1.1.0, the reusable
exact-batch evidence workflow must also pass against immutable pre-promotion
commit `fdb4efd5f407717c8e18b94e6f4c21cbfb8e5daa`; its correctness-attested
benchmark, scaling, and scalar-attribution artifacts are release evidence rather
than a substitute for the test suite.

## Platform artifacts

The release workflow builds wheels for the supported Python versions and
platforms. Native artifacts must contain:

- the canonical Python package,
- the stable C++ boundary and exact-batch extensions,
- `treemendous/py.typed`, `treemendous/exact_batch.py`, and
  `treemendous/cpp/_exact_batch.pyi`,
- package-relative resources required by any included optional extension,
- no repository-local build products or generated benchmark output.

Every wheel is installed into an isolated environment and exercised through
the public `RangeSet` and stable exact-batch APIs from an unrelated working
directory before publication. The freshly built sdist is installed into a
separate isolated environment and receives the same arbitrary-working-directory
smoke without importing from the source checkout.

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
4. Confirm the PyPI `treemendous` project trusts this repository's release
   workflow and that the protected GitHub `pypi` environment permits OIDC
   publication.
5. Create and push the matching tag, for example `v1.1.0`.
6. Let the release workflow aggregate, verify, run exact-batch evidence against
   the immutable pre-promotion commit, and publish all artifacts once.

Do not upload artifacts manually from a platform build job. Do not run multiple
publishers for one release.

## Post-release verification

Install the published wheel in a clean environment and exercise the public API:

```bash
python -m venv /tmp/treemendous-release-check
/tmp/treemendous-release-check/bin/python -m pip install treemendous==1.1.0
(cd /tmp && /tmp/treemendous-release-check/bin/python - <<'PY'
from treemendous import Span, create_range_set
from treemendous.exact_batch import BatchMutation, ExactBatchRangeSet, MutationOpcode

ranges = create_range_set((0, 100))
result = ranges.allocate(10, not_before=0)
assert result is not None and result.span == Span(0, 10)
assert tuple(item.span for item in ranges.intervals()) == (Span(10, 100),)

exact = ExactBatchRangeSet((0, 100), initially_available=False)
results = exact.mutate([
    BatchMutation(MutationOpcode.ADD, 10, 20),
    BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 12, 14),
])
assert [item.changed_length for item in results] == [10, 2]
PY
)
```
