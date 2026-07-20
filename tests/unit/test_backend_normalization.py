from __future__ import annotations

from types import SimpleNamespace

import pytest

from treemendous.backends.normalize import normalize_interval, normalize_intervals
from treemendous.domain import IntervalResult


def test_normalize_interval_accepts_every_backend_result_shape() -> None:
    canonical = IntervalResult(1, 3, data="A")
    shaped = SimpleNamespace(start=7, end=9, data="B")
    pair = tuple([1, 3])
    triple = tuple([1, 3, "A"])

    assert normalize_interval(None) is None
    assert normalize_interval(canonical) is canonical
    assert normalize_interval(pair) == IntervalResult(1, 3)
    assert normalize_interval(triple) == canonical
    assert normalize_interval(shaped) == IntervalResult(7, 9, data="B")


def test_normalize_intervals_filters_none_and_preserves_order() -> None:
    values = [None, tuple([0, 2]), SimpleNamespace(start=4, end=6)]
    assert normalize_intervals(values) == [IntervalResult(0, 2), IntervalResult(4, 6)]
    assert normalize_intervals(None) == []


def test_normalize_interval_rejects_unknown_shapes() -> None:
    with pytest.raises(ValueError, match="cannot normalize interval result"):
        normalize_interval(tuple([1, 2, 3, 4]))
    with pytest.raises(ValueError, match="cannot normalize interval result"):
        normalize_interval(object())
