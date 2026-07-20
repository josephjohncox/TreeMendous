# Subtitle cue ranges

## Model

`SubtitleCatalog` retains each timed cue independently, including cue ID, language, rendering layer, text, half-open playback interval, and insertion order. Overlapping dialogue, signs, translations, and corrected cues are not merged.

`active_at` implements half-open playback semantics: a cue is active at its start and inactive at its end. Optional language filtering happens before rendering order. Active and window queries return lower layers first, then earlier cue starts, then original insertion order. That deterministic policy makes coincident cues reproducible without discarding their identities.

## Mutation and validation

Cue IDs, language tags, and text are nonempty strings. Layers are nonnegative integers, and cue ranges must be nonempty. `update` may change timing, language, layer, or text while retaining the stable handle and original insertion position. `remove` affects one cue only. `snapshot` is immutable and insertion ordered.

## Example

```python
catalog.add("dialogue", 1_000, 3_000, language="en", layer=0,
            text="Look out!")
catalog.add("sign", 1_500, 2_500, language="en", layer=1,
            text="DANGER")
render_stack = catalog.active_at(2_000, language="en")
```

Time units are caller-defined but must be consistent, typically milliseconds or media timescale ticks.
