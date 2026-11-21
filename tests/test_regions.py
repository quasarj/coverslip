from pathlib import Path

import numpy as np
import pytest
import tifffile

from coverslip import regions
from coverslip.model import Region, RegionSpec
from coverslip.redact import redact_file

from conftest import load_layout


def test_drag_rounds_outward(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    region = regions.region_from_drag(half, 10.2, 20.5, 50.7, 60.0)
    assert region == Region(10, 20, 41, 40)


def test_drag_in_any_direction(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    forward = regions.region_from_drag(half, 10, 20, 50, 60)
    assert regions.region_from_drag(half, 50, 60, 10, 20) == forward


def test_drag_is_clipped_to_level(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    assert regions.region_from_drag(half, -5, -5, 300, 10) == Region(0, 0, 210, 10)


@pytest.mark.parametrize(
    "drag", [(10, 10, 10, 40), (10, 10, 40, 10), (-20, 5, -1, 40), (215, 5, 230, 9)]
)
def test_empty_drag_is_none(lzw_pyramid: Path, drag: tuple) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    assert regions.region_from_drag(half, *drag) is None


def test_region_arg(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    arg = regions.region_arg(Region(10, 20, 41, 40), half)
    assert arg == "10,20,41,40@210x150"
    assert RegionSpec.parse(arg) == RegionSpec(Region(10, 20, 41, 40), (210, 150))


def test_redact_command_quotes_path() -> None:
    command = regions.redact_command("my slide.svs", ["1,2,3,4@10x10", "5,6,7,8"])
    assert command == (
        "coverslip redact 'my slide.svs' OUTPUT -r 1,2,3,4@10x10 -r 5,6,7,8"
    )


def test_preview_maps_specs_onto_shown_level(lzw_pyramid: Path) -> None:
    layout = load_layout(lzw_pyramid)
    full, half, quarter = layout.images[0].levels
    specs = [
        RegionSpec(Region(10, 20, 41, 40), (210, 150)),
        RegionSpec(Region(100, 100, 40, 40)),
        RegionSpec(Region(5000, 5000, 10, 10)),
    ]
    shown = regions.preview_on_level(layout, full, quarter, specs)
    # outward rounding: 10..51 on the half level is 5..26 on the quarter
    assert shown == [Region(5, 10, 21, 20), Region(25, 25, 10, 10)]
    assert regions.preview_on_level(layout, full, half, specs[:1]) == [
        Region(10, 20, 41, 40)
    ]


@pytest.mark.parametrize("level_index", [0, 1, 2])
def test_drawn_region_is_fully_redacted_on_its_level(
    tmp_path: Path, lzw_pyramid: Path, level_index: int
) -> None:
    """What the viewer prints, redact must black out, every pixel of it."""
    layout = load_layout(lzw_pyramid)
    level = layout.images[0].levels[level_index]
    region = regions.region_from_drag(level, 7.3, 11.6, 33.2, 29.9)
    assert region is not None
    spec = RegionSpec.parse(regions.region_arg(region, level))
    out = tmp_path / "out.tif"
    redact_file(str(lzw_pyramid), str(out), [spec])
    with tifffile.TiffFile(out) as tf:
        levels = [tf.pages[0]] + list(tf.pages[0].pages)
        pixels = levels[level_index].asarray()
    painted = pixels[region.y : region.y1, region.x : region.x1]
    assert painted.max() == 0
    assert np.any(pixels[region.y1 + 2 :] != 0)
