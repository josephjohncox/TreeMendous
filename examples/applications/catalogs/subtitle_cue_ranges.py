"""Render active subtitle cues in layer order."""

from treemendous.applications.catalogs.subtitle_cue_ranges import SubtitleCatalog


def main() -> None:
    catalog = SubtitleCatalog()
    catalog.add("speaker", 0, 2000, language="en", layer=0, text="Hello")
    catalog.add("sign", 500, 1500, language="en", layer=1, text="EXIT")
    print([cue.payload.text for cue in catalog.active_at(1000, language="en")])


if __name__ == "__main__":
    main()
