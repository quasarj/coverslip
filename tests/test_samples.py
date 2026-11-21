"""End-to-end checks on public sample slides in ``samples/`` (skipped if absent)."""

from pathlib import Path

import numpy as np
import pytest

from coverslip import layout as layout_mod
from coverslip.model import Region, Role
from coverslip.redact import redact_file
from coverslip.verify import verify_file

from conftest import requires_openslide, requires_sample, sample_path


def _openslide_checks(
    src: str, out: Path, origin: tuple[int, int], size: tuple[int, int]
) -> None:
    import openslide

    with openslide.OpenSlide(str(out)) as slide, openslide.OpenSlide(src) as original:
        for level, ds in enumerate(slide.level_downsamples):
            # Stay 32 level-pixels inside the rectangle to avoid MCU/rounding edges.
            margin = int(32 * ds)
            w, h = int((size[0] - 2 * margin) / ds), int((size[1] - 2 * margin) / ds)
            if w < 4 or h < 4:
                continue
            inner = slide.read_region(
                (origin[0] + margin, origin[1] + margin), level, (w, h)
            )
            assert np.asarray(inner.convert("RGB")).max() == 0
            probe = (0, 0)
            a = np.asarray(slide.read_region(probe, level, (24, 24)).convert("RGB"))
            b = np.asarray(original.read_region(probe, level, (24, 24)).convert("RGB"))
            assert np.array_equal(a, b)


@requires_sample("CMU-1-JP2K-33005.svs")
@requires_openslide
def test_aperio_jpeg2000(tmp_path: Path) -> None:
    src = sample_path("CMU-1-JP2K-33005.svs")
    layout = layout_mod.discover(layout_mod.read_info(src))
    assert layout.images[0].full.compression == 33005
    out = tmp_path / "out.svs"
    region = Region(10000, 8000, 3000, 2500)
    redact_file(src, str(out), [region])
    assert verify_file(src, str(out), [region]).ok
    _openslide_checks(src, out, (10000, 8000), (3000, 2500))


@requires_sample("CMU-1.tiff")
@requires_openslide
def test_generic_ycbcr_chain_pyramid(tmp_path: Path) -> None:
    src = sample_path("CMU-1.tiff")
    layout = layout_mod.discover(layout_mod.read_info(src))
    levels = layout.images[0].levels
    assert layout.vendor == "generic" and len(levels) == 9
    assert levels[0].photometric == 6 and all(
        lvl.role != Role.THUMBNAIL for lvl in levels
    )
    out = tmp_path / "out.tiff"
    region = Region(12000, 9000, 4000, 3000)
    redact_file(src, str(out), [region])
    assert verify_file(src, str(out), [region]).ok
    _openslide_checks(src, out, (12000, 9000), (4000, 3000))


@requires_sample("Leica-1.ome.tiff")
def test_ome_planar_bigtiff_with_macro(tmp_path: Path) -> None:
    import tifffile

    src = sample_path("Leica-1.ome.tiff")
    layout = layout_mod.discover(layout_mod.read_info(src))
    assert layout.vendor == "ome" and layout.bigtiff
    assert len(layout.images) == 1 and len(layout.images[0].levels) == 5
    assert layout.images[0].full.planar_config == 2
    assert [ifd.width for ifd in layout.macro] == [1616, 404, 101]
    out = tmp_path / "out.ome.tiff"
    region = Region(15000, 12000, 5000, 4000)
    redact_file(src, str(out), [region])
    assert verify_file(src, str(out), [region]).ok
    with tifffile.TiffFile(src) as a, tifffile.TiffFile(out) as b:
        assert a.ome_metadata == b.ome_metadata
        main = [s for s in b.series if 38432 in s.shape][0]
        for level in main.levels:
            data = np.moveaxis(level.asarray(), 0, -1)
            h, w = data.shape[:2]
            x0, y0 = int(15100 * w / 36832), int(12100 * h / 38432)
            x1, y1 = int(19900 * w / 36832), int(15900 * h / 38432)
            if x1 > x0 and y1 > y0:
                assert data[y0:y1, x0:x1].max() == 0
