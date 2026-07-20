"""Subtitle cue catalog contract."""

from tests.oracles.applications.catalogs.subtitle_cue_ranges import active
from treemendous.applications.catalogs.subtitle_cue_ranges import SubtitleCatalog


def test_subtitle_cues_keep_language_layer_identity_and_render_order() -> None:
    catalog = SubtitleCatalog()
    upper = catalog.add("upper", 100, 200, language="en", layer=2, text="top")
    lower = catalog.add("lower", 90, 180, language="en", layer=0, text="bottom")
    catalog.add("fr", 90, 180, language="fr", layer=0, text="bas")
    rows = [("upper", "en", 2, 100, 200), ("lower", "en", 0, 90, 180)]
    actual = catalog.active_at(120, language="en")
    assert tuple(record.payload.cue_id for record in actual) == active(rows, 120, "en")
    assert catalog.update(upper, layer=1).handle == upper
    assert catalog.remove(lower).handle == lower
    assert catalog.snapshot().records[0].handle == upper
