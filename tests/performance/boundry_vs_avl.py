#!/usr/bin/env python3
"""Retired legacy measurement entry point."""

from __future__ import annotations

from tests.performance.legacy import fail_legacy


if __name__ == "__main__":
    raise SystemExit(fail_legacy("boundry_vs_avl.py"))
