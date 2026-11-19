from pathlib import Path

import numpy as np
import pytest

from tiffredact import layout as layout_mod
from tiffredact.model import Role

from conftest import make_image, requires_svs, sample_svs_paths, write_pyramid


def discover(path: Path):
    return layout_mod.discover(layout_mod.read_info(str(path)))


def test_subifd_pyramid(lzw_pyramid: Path) -> None:
    layout = discover(lzw_pyramid)
    assert layout.vendor == "generic"
    assert len(layout.images) == 1
    roles = [level.role for level in layout.images[0].levels]
    assert roles == [Role.FULL, Role.LEVEL, Role.LEVEL]
    assert [lvl.width for lvl in layout.images[0].levels] == [420, 210, 105]
    assert not layout.label and not layout.macro and not layout.other


def test_chain_pyramid(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = write_pyramid(tmp_path / "chain.tif", rgb_image, chain=True)
    layout = discover(path)
    assert len(layout.images) == 1
    assert [lvl.width for lvl in layout.images[0].levels] == [420, 210, 105]


def test_ome_multi_plane(tmp_path: Path) -> None:
    import tifffile

    image = make_image(200, 260)
    path = tmp_path / "multi.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        for seed in range(2):
            plane = make_image(200, 260, seed=seed)
            writer.write(plane, tile=(64, 64), compression="lzw", subifds=1, metadata={"axes": "YXS"})
            writer.write(plane[::2, ::2], subfiletype=1, tile=(64, 64), compression="lzw")
    layout = discover(path)
    assert layout.vendor == "ome"
    assert len(layout.images) == 2
    for image_layout in layout.images:
        assert [lvl.role for lvl in image_layout.levels] == [Role.FULL, Role.LEVEL]
    del image


def test_stripped_image_has_strip_chunks(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = write_pyramid(tmp_path / "strips.tif", rgb_image, levels=1, tile=None, rowsperstrip=32)
    layout = discover(path)
    full = layout.images[0].full
    assert not full.tiled and full.chunk_height == 32 and full.chunk_count == 10


@requires_svs
@pytest.mark.parametrize("path", sample_svs_paths())
def test_aperio_sample_layout(path: str) -> None:
    layout = discover(Path(path))
    assert layout.vendor == "aperio"
    levels = layout.images[0].levels
    assert levels[0].role == Role.FULL and levels[0].tiled
    assert any(level.role == Role.THUMBNAIL for level in levels)
    assert len(layout.label) == 1 and len(layout.macro) == 1
    assert not layout.other
    for level in levels:
        assert len(level.data_offsets) == level.chunk_count
