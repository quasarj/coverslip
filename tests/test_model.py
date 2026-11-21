import pytest

from coverslip.model import Region, RegionSpec


def test_region_parse_and_bounds() -> None:
    region = Region.parse("10, 20,30,40")
    assert (region.x, region.y, region.width, region.height) == (10, 20, 30, 40)
    assert (region.x1, region.y1) == (40, 60)


@pytest.mark.parametrize("text", ["1,2,3", "a,b,c,d", "1,2,0,4", "1,2,3,-1"])
def test_region_parse_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        Region.parse(text)


def test_region_intersection_and_clip() -> None:
    a = Region(0, 0, 10, 10)
    b = Region(5, 5, 10, 10)
    assert a.intersection(b) == Region(5, 5, 5, 5)
    assert a.intersection(Region(10, 0, 5, 5)) is None
    assert Region(-5, -5, 10, 10).clip(100, 100) == Region(0, 0, 5, 5)
    assert Region(200, 200, 10, 10).clip(100, 100) is None
    assert a.translate(-2, 3) == Region(-2, 3, 10, 10)


def test_region_spec_parse_with_and_without_frame() -> None:
    assert RegionSpec.parse("1,2,3,4") == RegionSpec(Region(1, 2, 3, 4), None)
    assert RegionSpec.parse("1,2,3,4@210x150") == RegionSpec(
        Region(1, 2, 3, 4), (210, 150)
    )
    assert RegionSpec.parse("1,2,3,4 @ 210X150").frame == (210, 150)


@pytest.mark.parametrize(
    "text", ["1,2,3,4@", "1,2,3,4@210", "1,2,3,4@axb", "1,2,3,4@0x150", "1,2,3@210x150"]
)
def test_region_spec_parse_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        RegionSpec.parse(text)
