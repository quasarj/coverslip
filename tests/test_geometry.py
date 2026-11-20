import pytest

from tiffredact.geometry import (
    RegionFrameError,
    chunk_at,
    covers_chunk,
    intersecting_chunks,
    region_to_full,
    resolve_region,
    scale_region,
)
from tiffredact.model import (
    Chunk,
    ImageIFD,
    Layout,
    PyramidImage,
    Region,
    RegionSpec,
    Role,
)


def make_ifd(width: int, height: int, tiled: bool, cw: int, ch: int) -> ImageIFD:
    return ImageIFD(
        offset=0,
        width=width,
        height=height,
        tiled=tiled,
        chunk_width=cw,
        chunk_height=ch,
        compression=1,
        photometric=2,
        bits_per_sample=8,
        samples_per_pixel=3,
        planar_config=1,
        predictor=1,
        sample_format=1,
        subsampling=None,
        jpegtables=None,
        new_subfile_type=0,
        description="",
        data_offsets=[],
        data_bytecounts=[],
        big_endian=False,
        role=Role.FULL,
    )


def test_scale_region_rounds_outward_with_non_integer_downsample() -> None:
    full = make_ifd(21911, 18129, True, 240, 240)
    level = make_ifd(5477, 4532, True, 240, 240)
    scaled = scale_region(Region(5000, 4000, 3000, 2000), full, level)
    assert scaled is not None
    assert scaled.x <= 5000 / (21911 / 5477)
    assert scaled.x1 >= 8000 / (21911 / 5477)
    assert scaled.y <= 4000 / (18129 / 4532)
    assert scaled.y1 >= 6000 / (18129 / 4532)


def test_scale_region_never_collapses_to_empty() -> None:
    full = make_ifd(10000, 10000, True, 240, 240)
    thumb = make_ifd(100, 100, False, 100, 16)
    scaled = scale_region(Region(10, 10, 3, 3), full, thumb)
    assert scaled == Region(0, 0, 1, 1)


def test_tile_chunks_at_edges_keep_nominal_size() -> None:
    ifd = make_ifd(250, 250, True, 240, 240)
    assert ifd.chunks_across == 2 and ifd.chunk_count == 4
    assert chunk_at(ifd, 3) == Chunk(3, 240, 240, 240, 240)


def test_strip_chunks_are_clipped_to_image_height() -> None:
    ifd = make_ifd(928, 768, False, 928, 16)
    assert ifd.chunk_count == 48
    assert chunk_at(ifd, 47) == Chunk(47, 0, 752, 928, 16)
    short = make_ifd(100, 70, False, 100, 16)
    assert chunk_at(short, 4) == Chunk(4, 0, 64, 100, 6)


def test_intersecting_chunks_reports_local_rects() -> None:
    ifd = make_ifd(1000, 1000, True, 240, 240)
    found = intersecting_chunks(ifd, Region(230, 10, 20, 20))
    assert [c.index for c, _ in found] == [0, 1]
    assert found[0][1] == Region(230, 10, 10, 20)
    assert found[1][1] == Region(0, 10, 10, 20)


def test_intersecting_chunks_outside_image_is_empty() -> None:
    ifd = make_ifd(100, 100, True, 64, 64)
    assert intersecting_chunks(ifd, Region(500, 500, 10, 10)) == []


def test_covers_chunk() -> None:
    chunk = Chunk(0, 0, 0, 64, 64)
    assert covers_chunk([Region(-5, -5, 100, 100)], chunk)
    assert not covers_chunk([Region(0, 0, 63, 64)], chunk)


def make_layout() -> Layout:
    full = make_ifd(21911, 18129, True, 240, 240)
    level = make_ifd(5477, 4532, True, 240, 240)
    return Layout("slide.svs", False, "generic", [PyramidImage([full, level])])


def test_region_to_full_round_trip_covers_the_level_region() -> None:
    layout = make_layout()
    full, level = layout.images[0].levels
    drawn = Region(1201, 977, 333, 211)
    back = scale_region(region_to_full(drawn, level, full), full, level)
    assert back is not None
    assert back.intersection(drawn) == drawn


def test_region_to_full_is_exact_for_integer_downsample() -> None:
    full = make_ifd(4000, 2000, True, 240, 240)
    level = make_ifd(1000, 500, True, 240, 240)
    assert region_to_full(Region(10, 20, 30, 40), level, full) == Region(
        40, 80, 120, 160
    )


def test_resolve_region_leaves_full_resolution_input_alone() -> None:
    layout = make_layout()
    region = Region(5, 6, 7, 8)
    assert resolve_region(layout, region) == region
    assert resolve_region(layout, RegionSpec(region)) == region
    assert resolve_region(layout, RegionSpec(region, (21911, 18129))) == region


def test_resolve_region_requires_an_exact_level_match() -> None:
    with pytest.raises(RegionFrameError, match="21911x18129, 5477x4532"):
        resolve_region(make_layout(), RegionSpec(Region(0, 0, 10, 10), (5477, 4533)))


def test_resolve_region_rejects_region_outside_its_level() -> None:
    spec = RegionSpec(Region(6000, 0, 10, 10), (5477, 4532))
    with pytest.raises(RegionFrameError, match="outside"):
        resolve_region(make_layout(), spec)
