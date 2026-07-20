"""Independent two-dimensional guarded-rectangle reference."""


def overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    lc0, lc1, lt0, lt1 = left
    rc0, rc1, rt0, rt1 = right
    return lc0 < rc1 and rc0 < lc1 and lt0 < rt1 and rt0 < lt1
