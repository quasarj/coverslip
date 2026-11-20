import logging
from pathlib import Path

import numpy as np
import pytest

from tiffview import layout as layout_mod
from tiffview.model import Role

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


def _link_subifds_by_next_pointer(path: Path) -> None:
    """Rewrite a classic-TIFF SubIFD pyramid so each sub-IFD also points at
    the next one through its next-IFD field, as Bio-Formats does."""
    import struct

    import tifffile

    with tifffile.TiffFile(path) as tif:
        offsets = list(tif.pages[0].subifds)
    with open(path, "r+b") as handle:
        for offset, following in zip(offsets, offsets[1:]):
            handle.seek(offset)
            count = struct.unpack("<H", handle.read(2))[0]
            handle.seek(offset + 2 + 12 * count)
            handle.write(struct.pack("<I", following))


def test_double_referenced_subifds_are_explained(
    tmp_path: Path, rgb_image: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_pyramid(tmp_path / "linked.tif", rgb_image)
    _link_subifds_by_next_pointer(path)
    with caplog.at_level(logging.INFO):
        layout = discover(path)
    assert [lvl.width for lvl in layout.images[0].levels] == [420, 210, 105]
    messages = [record.getMessage() for record in caplog.records]
    assert "SubIFDs are double referenced" not in messages
    assert any("linked both by the SubIFDs tag" in m for m in messages)
    assert all(record.levelno < logging.WARNING for record in caplog.records)


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
            writer.write(
                plane,
                tile=(64, 64),
                compression="lzw",
                subifds=1,
                metadata={"axes": "YXS"},
            )
            writer.write(
                plane[::2, ::2], subfiletype=1, tile=(64, 64), compression="lzw"
            )
    layout = discover(path)
    assert layout.vendor == "ome"
    assert len(layout.images) == 2
    for image_layout in layout.images:
        assert [lvl.role for lvl in image_layout.levels] == [Role.FULL, Role.LEVEL]
    del image


def test_stripped_image_has_strip_chunks(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = write_pyramid(
        tmp_path / "strips.tif", rgb_image, levels=1, tile=None, rowsperstrip=32
    )
    layout = discover(path)
    full = layout.images[0].full
    assert not full.tiled and full.chunk_height == 32 and full.chunk_count == 10


def test_per_sample_sample_format_is_accepted() -> None:
    ifd = {
        "offset": 8,
        "tags": {
            256: {"data": [64]},
            257: {"data": [64]},
            258: {"data": [8, 8, 8]},
            277: {"data": [3]},
            339: {"data": [1, 1, 1]},
        },
    }
    assert layout_mod.image_from_ifd(ifd).sample_format == 1


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


def test_prune_leaked_subifd_chain() -> None:
    main = {"offset": 500, "tags": {}}
    leaked = [{"offset": 100, "tags": {}}, main]
    info = {
        "ifds": [
            {
                "offset": 16,
                "tags": {330: {"ifds": [[{"offset": 50, "tags": {}}], leaked]}},
            },
            main,
        ]
    }
    assert layout_mod.prune_leaked_subifd_chains(info) == 1
    chains = info["ifds"][0]["tags"][330]["ifds"]
    assert [[sub["offset"] for sub in chain] for chain in chains] == [[50], [100]]


def test_ome_secondary_image_is_classified_as_macro(tmp_path: Path) -> None:
    import tifffile

    path = tmp_path / "with_macro.ome.tif"
    main = make_image(256, 320)
    macro = make_image(120, 80, seed=3)
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(
            macro, compression="lzw", metadata={"axes": "YXS", "Name": "macro"}
        )
        writer.write(
            main,
            tile=(64, 64),
            compression="lzw",
            subifds=1,
            metadata={"axes": "YXS", "Name": "slide"},
        )
        writer.write(main[::2, ::2], subfiletype=1, tile=(64, 64), compression="lzw")
    layout = discover(path)
    assert layout.vendor == "ome"
    assert len(layout.images) == 1
    assert layout.images[0].full.width == 320
    assert [ifd.width for ifd in layout.macro] == [80]
    assert [ifd.role for ifd in layout.macro] == [Role.MACRO]
