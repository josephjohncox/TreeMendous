"""Independent whole-input word-count oracle."""


def expected_word_counts(data: bytes) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for word in data.decode().split():
        word = word.lower()
        counts[word] = counts.get(word, 0) + 1
    return tuple(sorted(counts.items()))
