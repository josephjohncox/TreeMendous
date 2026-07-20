"""Time-series alert window contract."""

from tests.oracles.applications.catalogs.timeseries_alert_windows import active
from treemendous.applications.catalogs.timeseries_alert_windows import (
    AlertCatalog,
    WindowKind,
)


def test_alert_and_suppression_windows_apply_priority_deterministically() -> None:
    catalog = AlertCatalog()
    low = catalog.add(
        "low", 0, 20, series="cpu", kind=WindowKind.ALERT, priority=3, label="warm"
    )
    high = catalog.add(
        "high", 0, 20, series="cpu", kind="alert", priority=9, label="hot"
    )
    catalog.add(
        "mute", 5, 15, series="cpu", kind="suppression", priority=5, label="work"
    )
    rows = [
        ("low", "cpu", "alert", 3, 0, 20),
        ("high", "cpu", "alert", 9, 0, 20),
        ("mute", "cpu", "suppression", 5, 5, 15),
    ]
    expected_firing, expected_suppressed = active(rows, "cpu", 10)
    result = catalog.active_at("cpu", 10)
    assert (
        tuple(record.payload.window_id for record in result.alerts) == expected_firing
    )
    assert (
        tuple(record.payload.window_id for record in result.suppressed)
        == expected_suppressed
    )
    assert catalog.update(low, priority=10).handle == low
    assert [record.handle for record in catalog.active_at("cpu", 10).alerts] == [
        low,
        high,
    ]
    assert catalog.remove(high).handle == high
    assert len(catalog.snapshot().records) == 2
