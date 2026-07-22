"""Find broad-phase robot volume-time conflicts in a local 4D index.

``BoxIndex4D`` is experimental and process-local. Its axis-aligned integer
boxes do not supply robot/trajectory identity, continuous motion or rotation,
narrow-phase collision geometry, uncertainty margins, planning, durability, or
multi-controller coordination.
"""

from treemendous.multidimensional import Box, BoxIndex4D


def main() -> None:
    plans = BoxIndex4D()
    sweep = Box((-2, -2, 0, 50), (2, 2, 4, 70))
    arm_a = plans.insert(sweep, "arm-a")
    arm_b = plans.insert(sweep, "arm-b")
    after = plans.insert(Box((-2, -2, 0, 70), (2, 2, 4, 90)), "arm-c")

    probe = Box((-1, -1, 1, 60), (1, 1, 2, 61))
    conflicts = plans.overlaps(probe)
    assert tuple(entry.handle for entry in conflicts) == (arm_a, arm_b)
    assert tuple(entry.data for entry in conflicts) == ("arm-a", "arm-b")
    assert plans.overlaps(Box((-1, -1, 1, 70), (1, 1, 2, 71))) == (plans.get(after),)
    moved = plans.update(arm_b, box=Box((8, 8, 0, 50), (12, 12, 4, 70)))
    assert moved.handle == arm_b
    assert tuple(entry.handle for entry in plans.overlaps(probe)) == (arm_a,)
    assert plans.remove(arm_a).data == "arm-a"
    print("conflicts=arm-a,arm-b handles=1,2,3 moved=2 remaining=2,3")


if __name__ == "__main__":
    main()
