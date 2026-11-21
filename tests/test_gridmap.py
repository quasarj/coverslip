from pathlib import Path

import numpy as np
import pytest
import tifffile

from coverslip import gridmap

from conftest import load_layout, make_image


def test_byte_grid_matches_bytecounts(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    grid = gridmap.byte_grid(full)
    assert grid.shape == (5, 7)
    assert grid[0, 0] == full.data_bytecounts[0]
    assert grid[1, 2] == full.data_bytecounts[1 * 7 + 2]
    assert grid.sum() == sum(full.data_bytecounts)


def test_byte_grid_sums_planes(tmp_path: Path) -> None:
    path = tmp_path / "planar.tif"
    planar = np.ascontiguousarray(np.moveaxis(make_image(100, 130), 2, 0))
    tifffile.imwrite(
        path, planar, tile=(64, 64), photometric="rgb", planarconfig="separate"
    )
    ifd = load_layout(path).images[0].full
    grid = gridmap.byte_grid(ifd)
    assert grid.shape == (2, 3)
    per_plane = ifd.chunks_per_plane
    counts = ifd.data_bytecounts
    assert grid[0, 1] == counts[1] + counts[1 + per_plane] + counts[1 + 2 * per_plane]


def test_tile_under_same_level(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    assert gridmap.tile_under(full, full, 0, 0) == (0, 0)
    assert gridmap.tile_under(full, full, 63.9, 64) == (0, 1)
    assert gridmap.tile_under(full, full, 419.5, 299.5) == (6, 4)


@pytest.mark.parametrize("x,y", [(-1, 5), (5, -0.1), (420, 5), (5, 300)])
def test_tile_under_outside_image(lzw_pyramid: Path, x: float, y: float) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    assert gridmap.tile_under(full, full, x, y) is None


def test_tile_under_across_levels(lzw_pyramid: Path) -> None:
    full, half, quarter = load_layout(lzw_pyramid).images[0].levels
    # quarter-res (105x75) point (40, 20) is full-res (160, 80): tile (2, 1)
    assert gridmap.tile_under(quarter, full, 40, 20) == (2, 1)
    # and half-res (210x150) tile (1, 0) of 64px covers quarter x 32..64
    assert gridmap.tile_under(quarter, half, 40, 20) == (1, 0)


def test_tile_box_on_view(lzw_pyramid: Path) -> None:
    full, _, quarter = load_layout(lzw_pyramid).images[0].levels
    assert gridmap.tile_box_on_view(quarter, full, 2, 1) == (32, 16, 48, 32)
    # edge tile is clipped to the image
    assert gridmap.tile_box_on_view(full, full, 6, 4) == (384, 256, 420, 300)


def test_grid_extent_covers_nominal_tiles(lzw_pyramid: Path) -> None:
    full, _, quarter = load_layout(lzw_pyramid).images[0].levels
    assert gridmap.grid_extent(full, full) == (0, 448, 320, 0)
    assert gridmap.grid_extent(quarter, full) == (0, 112, 80, 0)
