# Building

## Development environment

```bash
uv sync --all-extras
```

Build a source distribution and local wheel with `just build`. Build the CPU
extensions in place with `just build-cpp`. Portable wheel builds intentionally
do not use `-march=native`, fast-math, or unconditional SIMD flags. Developers
may opt into host-specific instruction tuning only with `just build-cpp-native`;
those artifacts are not distributable wheels.

## Optional CPU ICL build

Install Boost headers/libraries and run `just build-cpp-icl`. ICL remains outside
the stable production catalog until the provisioned Linux parity lane passes.

## Metal

On macOS with Xcode Command Line Tools, run `just build-metal`. The build compiles
the shader into a wheel resource beside the extension. Runtime loading is
package-relative, not current-working-directory-relative. Metal is 32-bit and
experimental until its wheel, arbitrary-CWD, device, batch, and parity gates
pass.

## CUDA

With a CUDA toolkit and supported compiler, run `just build-gpu`. Building or
importing the extension does not mark CUDA stable or available to automatic
selection. Promotion requires the hardware contract and compute-sanitizer lane.

## Artifact policy

`just verify-artifacts` inspects existing files under `dist/` and `wheelhouse/`.
An sdist must contain source and metadata but no host libraries. A wheel must
contain the expected native modules for its platform; a Metal wheel must also
contain its generated metallib resource. Clean-install checks run from an
unrelated working directory.
