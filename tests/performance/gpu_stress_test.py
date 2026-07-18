#!/usr/bin/env python3
"""Retired stress script that converted device exceptions into samples."""

from __future__ import annotations

from tests.performance.legacy import fail_legacy


if __name__ == "__main__":
    raise SystemExit(
        fail_legacy("gpu_stress_test.py", "gpu_benchmark.py or metal_benchmark.py")
    )
