from __future__ import annotations

from treemendous.applications.scheduling.ci_runners import CIRunnerScheduler, Runner


def test_ci_labels_concurrency_and_cancellation() -> None:
    scheduler = CIRunnerScheduler(
        (Runner("linux", frozenset({"linux", "x86"}), concurrency=2),)
    )
    first = scheduler.schedule("a", 2, labels=frozenset({"linux"}), deadline=6)
    second = scheduler.schedule("b", 2, labels=frozenset({"x86"}), deadline=6)
    third = scheduler.schedule("c", 2, labels=frozenset({"linux"}), deadline=6)
    assert first.start == 0
    assert second.start == 0
    assert third.start == 2
    scheduler.cancel("a", first.id)
    replacement = scheduler.schedule("d", 2, labels=frozenset({"linux"}), deadline=2)
    assert replacement.start == 0
