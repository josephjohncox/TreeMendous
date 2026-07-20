"""Video edit region contract."""

from tests.oracles.applications.catalogs.video_edit_regions import affected
from treemendous.applications.catalogs.video_edit_regions import VideoEditCatalog


def test_video_edits_retain_tracks_effects_and_invalidation_coverage() -> None:
    catalog = VideoEditCatalog()
    grade = catalog.add("grade", 10, 20, track="v1", effect="color")
    blur = catalog.add("blur", 15, 25, track="v1", effect="blur")
    catalog.add("audio", 10, 30, track="a1", effect="gain")
    rows = [("grade", "v1", "color", 10, 20), ("blur", "v1", "blur", 15, 25)]
    invalidation = catalog.invalidation(12, 22, tracks=frozenset({"v1"}))
    assert tuple(
        record.payload.region_id for record in invalidation.records
    ) == affected(rows, 12, 22, frozenset({"v1"}))
    assert invalidation.coverage.maximum_count == 2
    assert catalog.update(blur, effect="sharpen").handle == blur
    assert catalog.remove(grade).handle == grade
    assert catalog.snapshot().records[0].payload.effect == "sharpen"
