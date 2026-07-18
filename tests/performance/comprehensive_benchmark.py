#!/usr/bin/env python3
"""Compatibility entry point for the correctness-checked benchmark harness."""

from __future__ import annotations

from tests.performance.protocol_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
