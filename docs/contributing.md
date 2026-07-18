# Contributing

Use Python 3.11 or newer and install all development groups:

```bash
uv sync --all-extras
just check
```

`just check` runs Ruff lint and format checks across tracked Python, mypy across
the package, branch coverage with the checked-in threshold, documentation
contracts, packaging policy checks, and bytecode compilation. `just test` runs
the full suite and stable-backend validation. `just validate` combines the test
and quality gates.

Tests use strict marker and configuration handling. Mark platform/toolchain-only
coverage with one of `native`, `icl`, `metal`, or `cuda`, always with an explicit
reason when unavailable. Use `benchmark` only for timing suites. Semantic
mismatches must fail rather than become skips. PR CI runs `benchmark-smoke` as a
correctness/load gate but never treats hosted-runner timing as a threshold.

Before review, also run:

```bash
uv lock --check
git diff --check
git diff --cached --name-only
```

Do not commit generated shared libraries, bytecode, profiler output, coverage
data, local benchmark artifacts, or host metallibs. Changes to native code
should name the platform and build command tested. Benchmark changes must
preserve trace/oracle validation, compact durable reports, and the separation
between sampled measurements and single-run load qualification.
