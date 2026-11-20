from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tiffview import chunks

from conftest import (
    downsample,
    load_layout,
    make_image,
    requires_openslide,
    requires_sample,
    sample_path,
    write_pyramid,
)


def test_level_assembly_matches_source(
    lzw_pyramid: Path, rgb_image: np.ndarray
) -> None:
    levels = load_layout(lzw_pyramid).images[0].levels
    for n, ifd in enumerate(levels):
        level = chunks.read_level(str(lzw_pyramid), ifd)
        assert np.array_equal(level, downsample(rgb_image, 2**n))


def test_jpeg_level_is_close(tmp_path: Path, rgb_image: np.ndarray) -> None:
    # tifffile's photometric="ycbcr" JPEG stores unconverted RGB samples, so
    # this uses RGB JPEG; YCbCr conversion is covered in codecs/test_pixel.py
    # and against OpenSlide on real samples below.
    path = write_pyramid(
        tmp_path / "rgbjpeg.tif",
        rgb_image,
        levels=1,
        compression="jpeg",
        compressionargs={"level": 95},
    )
    ifd = load_layout(path).images[0].full
    level = chunks.read_level(str(path), ifd)
    assert level.shape == rgb_image.shape
    assert np.abs(level.astype(int) - rgb_image.astype(int)).mean() < 8


def test_edge_tile_is_cropped(lzw_pyramid: Path, rgb_image: np.ndarray) -> None:
    ifd = load_layout(lzw_pyramid).images[0].full
    assert (ifd.chunks_across, ifd.chunks_down) == (7, 5)
    tile = chunks.read_tile(str(lzw_pyramid), ifd, 6, 4)
    assert tile.shape == (300 - 256, 420 - 384, 3)
    assert np.array_equal(tile, rgb_image[256:, 384:])


def test_interior_tile(lzw_pyramid: Path, rgb_image: np.ndarray) -> None:
    ifd = load_layout(lzw_pyramid).images[0].full
    tile = chunks.read_tile(str(lzw_pyramid), ifd, 2, 1)
    assert np.array_equal(tile, rgb_image[64:128, 128:192])


def test_stripped_image(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = write_pyramid(
        tmp_path / "strips.tif", rgb_image, levels=1, tile=None, rowsperstrip=70
    )
    ifd = load_layout(path).images[0].full
    assert not ifd.tiled
    assert np.array_equal(chunks.read_level(str(path), ifd), rgb_image)
    last = chunks.read_tile(str(path), ifd, 0, ifd.chunks_down - 1)
    assert np.array_equal(last, rgb_image[280:])


def test_separate_planes(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = tmp_path / "planar.tif"
    planar = np.ascontiguousarray(np.moveaxis(rgb_image, 2, 0))
    tifffile.imwrite(
        path, planar, tile=(64, 64), photometric="rgb", planarconfig="separate"
    )
    ifd = load_layout(path).images[0].full
    assert ifd.planes == 3
    assert np.array_equal(chunks.read_level(str(path), ifd), rgb_image)


def test_big_endian_is_made_native(tmp_path: Path) -> None:
    path = tmp_path / "be.tif"
    image = (make_image(100, 80)[:, :, 0].astype(np.uint16) * 257)[:, :, None]
    tifffile.imwrite(path, image[:, :, 0], tile=(32, 32), byteorder=">")
    ifd = load_layout(path).images[0].full
    level = chunks.read_level(str(path), ifd)
    assert level.dtype.isnative
    assert np.array_equal(level, image)


def test_empty_tile(lzw_pyramid: Path, rgb_image: np.ndarray) -> None:
    ifd = load_layout(lzw_pyramid).images[0].full
    counts = list(ifd.data_bytecounts)
    counts[chunks.tile_index(ifd, 1, 1)] = 0
    ifd = replace(ifd, data_bytecounts=counts)
    with pytest.raises(chunks.EmptyChunkError):
        chunks.read_tile(str(lzw_pyramid), ifd, 1, 1)
    level = chunks.read_level(str(lzw_pyramid), ifd)
    assert level[64:128, 64:128].max() == 0
    assert np.array_equal(level[:64], rgb_image[:64])


def test_level_size_limit(lzw_pyramid: Path) -> None:
    ifd = load_layout(lzw_pyramid).images[0].full
    with pytest.raises(chunks.LevelTooLargeError):
        chunks.read_level(str(lzw_pyramid), ifd, max_pixels=1000)
    assert chunks.read_level(str(lzw_pyramid), ifd, max_pixels=None).shape[0] == 300


def test_tile_outside_grid(lzw_pyramid: Path) -> None:
    ifd = load_layout(lzw_pyramid).images[0].full
    with pytest.raises(chunks.TileIndexError, match=r"\(7,0\).*7x5"):
        chunks.read_tile(str(lzw_pyramid), ifd, 7, 0)


@requires_openslide
@pytest.mark.parametrize(
    "name", ["CMU-1.tiff", "CMU-1-Small-Region.svs", "CMU-1-JP2K-33005.svs"]
)
def test_tiles_match_openslide(name: str) -> None:
    import openslide

    path = sample_path(name)
    if path is None:
        pytest.skip(f"samples/{name} not downloaded")
    full = load_layout(Path(path)).images[0].full
    col, row = full.chunks_across // 2, full.chunks_down // 2
    tile = chunks.read_tile(path, full, col, row)
    x, y = col * full.chunk_width, row * full.chunk_height
    with openslide.OpenSlide(path) as slide:
        region = slide.read_region((x, y), 0, (tile.shape[1], tile.shape[0]))
    expected = np.asarray(region)[:, :, :3]
    assert np.abs(tile.astype(int) - expected.astype(int)).mean() < 1
