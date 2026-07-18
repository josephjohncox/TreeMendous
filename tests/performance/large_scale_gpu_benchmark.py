#!/usr/bin/env python3
"""Retired accelerator benchmark that lacked identical trace semantics."""

from __future__ import annotations

from tests.performance.legacy import fail_legacy

if __name__ == "__main__":
    raise SystemExit(
        fail_legacy(
            "large_scale_gpu_benchmark.py", "gpu_benchmark.py or metal_benchmark.py"
        )
    )
