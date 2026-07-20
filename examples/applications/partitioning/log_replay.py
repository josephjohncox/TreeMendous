#!/usr/bin/env python3
"""Run idempotent offset replay from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
)


def main() -> None:
    events = (ReplayEvent(2, "count", "increment", 2), ReplayEvent(0, "count", "set", 1))
    state = LogReplayEngine(events).run(window_size=1)
    if state != (("count", 3),):
        raise RuntimeError("unexpected replay state")
    print("log-replay: count=3")


if __name__ == "__main__":
    main()
