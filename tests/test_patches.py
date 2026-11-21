import pytest

from coverslip.patches import (
    Patch,
    PatchError,
    PatchSet,
    Zero,
    merge_ranges,
    ranges_intersect,
    subtract_ranges,
)


def test_merge_joins_touching_and_overlapping_ranges() -> None:
    assert merge_ranges([(10, 20), (0, 5), (5, 8), (15, 30), (40, 40)]) == [
        (0, 8),
        (10, 30),
    ]


def test_subtract_cuts_holes_out_of_ranges() -> None:
    assert subtract_ranges([(0, 100)], [(10, 20), (50, 60)]) == [
        (0, 10),
        (20, 50),
        (60, 100),
    ]
    assert subtract_ranges([(0, 10), (20, 30)], [(5, 25)]) == [(0, 5), (25, 30)]
    assert subtract_ranges([(0, 10)], [(0, 10)]) == []
    assert subtract_ranges([(0, 10)], []) == [(0, 10)]


def test_ranges_intersect() -> None:
    ranges = [(10, 20), (30, 40)]
    assert ranges_intersect(ranges, (19, 25))
    assert ranges_intersect(ranges, (0, 31))
    assert not ranges_intersect(ranges, (20, 30))
    assert not ranges_intersect(ranges, (40, 50))


def test_patch_set_sizes() -> None:
    patch_set = PatchSet(100, [Patch(100, b"abcd"), Patch(10, b"xy")], [Zero(20, 5)])
    patch_set.validate()
    assert patch_set.final_size == 104
    assert patch_set.bytes_appended == 4
    assert patch_set.bytes_written == 6
    assert patch_set.bytes_zeroed == 5


def test_validate_rejects_overlap_gap_and_zero_past_end() -> None:
    with pytest.raises(PatchError):
        PatchSet(100, [Patch(10, b"abcd")], [Zero(12, 4)]).validate()
    with pytest.raises(PatchError):
        PatchSet(100, [Patch(104, b"abcd")]).validate()
    with pytest.raises(PatchError):
        PatchSet(100, [], [Zero(98, 4)]).validate()
