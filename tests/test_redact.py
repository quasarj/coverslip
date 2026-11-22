from pathlib import Path

import numpy as np
import pytest
import tifffile

from coverslip.model import Region, RegionSpec
from coverslip.redact import RedactionError, redact_file
from coverslip.verify import verify_file

from conftest import (
    make_image,
    requires_openslide,
    requires_svs,
    sample_svs_paths,
    write_pyramid,
)
from test_inplace import write_aperio_like

REGION = Region(100, 50, 120, 90)


def assert_levels_redacted(
    output: Path, regions: list[Region], tolerance: int = 0, inset: int = 1
) -> None:
    with tifffile.TiffFile(output) as tf:
        series = tf.series[0]
        full_w = series.levels[0].shape[1]
        full_h = series.levels[0].shape[0]
        for level in series.levels:
            data = level.asarray()
            h, w = data.shape[:2]
            for region in regions:
                x0 = int(region.x * w / full_w) + inset
                y0 = int(region.y * h / full_h) + inset
                x1 = int(region.x1 * w / full_w) - inset
                y1 = int(region.y1 * h / full_h) - inset
                if x1 <= x0 or y1 <= y0:
                    continue
                assert (
                    data[y0:y1, x0:x1].max() <= tolerance
                ), f"level {w}x{h} not filled"


@pytest.mark.parametrize("bigtiff", [False, True])
def test_lzw_subifd_pyramid(
    tmp_path: Path, rgb_image: np.ndarray, bigtiff: bool
) -> None:
    src = write_pyramid(
        tmp_path / "src.tif", rgb_image, predictor=True, bigtiff=bigtiff
    )
    out = tmp_path / "out.tif"
    report = redact_file(str(src), str(out), [REGION])
    assert report.chunks_redacted > 0
    assert_levels_redacted(out, [REGION])
    audit = verify_file(str(src), str(out), [REGION])
    assert audit.ok, audit.summary()
    with tifffile.TiffFile(out) as tf:
        assert tf.is_bigtiff == bigtiff
        assert len(tf.series[0].levels) == 3


def test_jpeg_420_pyramid_dct_and_pixel_paths(
    tmp_path: Path, jpeg_pyramid: Path
) -> None:
    for use_dct in (True, False):
        out = tmp_path / f"out_{use_dct}.tif"
        redact_file(str(jpeg_pyramid), str(out), [REGION], use_dct=use_dct)
        assert_levels_redacted(out, [REGION], tolerance=2, inset=16)
        audit = verify_file(str(jpeg_pyramid), str(out), [REGION])
        assert audit.ok, audit.summary()


def test_chain_pyramid_and_multiple_regions(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    src = write_pyramid(
        tmp_path / "chain.tif", rgb_image, chain=True, compression="zlib"
    )
    out = tmp_path / "out.tif"
    regions = [REGION, Region(300, 200, 100, 80)]
    redact_file(str(src), str(out), regions)
    assert_levels_redacted(out, regions)
    assert verify_file(str(src), str(out), regions).ok


def test_ome_multi_plane_applies_to_every_plane(tmp_path: Path) -> None:
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
    out = tmp_path / "out.ome.tif"
    region = Region(70, 70, 60, 60)
    redact_file(str(path), str(out), [region])
    with tifffile.TiffFile(path) as src_tf, tifffile.TiffFile(out) as tf:
        assert tf.ome_metadata == src_tf.ome_metadata
        for series in tf.series:
            for level in series.levels:
                data = level.asarray()
                scale = 260 / data.shape[1]
                x0, y0 = int(71 / scale), int(71 / scale)
                x1, y1 = int(129 / scale), int(129 / scale)
                assert data[y0:y1, x0:x1].max() == 0
    assert verify_file(str(path), str(out), [region]).ok


def test_fill_colour(tmp_path: Path, rgb_image: np.ndarray) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, levels=1)
    out = tmp_path / "out.tif"
    redact_file(str(src), str(out), [REGION], fill=(10, 200, 30))
    data = tifffile.imread(out)
    assert (data[REGION.y : REGION.y1, REGION.x : REGION.x1] == (10, 200, 30)).all()
    assert verify_file(str(src), str(out), [REGION], fill=(10, 200, 30)).ok


@pytest.mark.parametrize("chain", [False, True])
def test_region_given_on_a_reduced_level(
    tmp_path: Path, rgb_image: np.ndarray, chain: bool
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, predictor=True, chain=chain)
    out = tmp_path / "out.tif"
    spec = RegionSpec(Region(25, 12, 30, 22), (105, 75))
    report = redact_file(str(src), str(out), [spec])
    assert report.regions == [Region(100, 48, 120, 88)]
    assert_levels_redacted(out, report.regions, inset=0)
    assert verify_file(str(src), str(out), [spec]).ok
    assert verify_file(str(src), str(out), report.regions).ok
    untouched = tifffile.imread(out)[:48]
    assert (untouched == rgb_image[:48]).all()


def test_region_on_unknown_level_is_rejected(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, predictor=True)
    spec = RegionSpec(Region(25, 12, 30, 22), (105, 76))
    with pytest.raises(ValueError, match="No pyramid level is 105x76"):
        redact_file(str(src), str(tmp_path / "out.tif"), [spec])
    assert not (tmp_path / "out.tif").exists()


def test_region_outside_image_is_rejected(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, levels=1)
    with pytest.raises(RedactionError):
        redact_file(str(src), str(tmp_path / "out.tif"), [Region(5000, 5000, 10, 10)])


def test_refuses_to_overwrite_without_flag(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    from coverslip.writer import WriterError

    src = write_pyramid(tmp_path / "src.tif", rgb_image, levels=1)
    out = tmp_path / "out.tif"
    redact_file(str(src), str(out), [REGION])
    with pytest.raises(WriterError):
        redact_file(str(src), str(out), [REGION])
    redact_file(str(src), str(out), [REGION], overwrite=True)


@requires_svs
@requires_openslide
def test_aperio_sample_end_to_end(tmp_path: Path) -> None:
    import openslide

    src = min(sample_svs_paths(), key=lambda p: Path(p).stat().st_size)
    out = tmp_path / "out.svs"
    regions = [Region(5000, 4000, 3000, 2000)]
    report = redact_file(src, str(out), regions, drop_label=True, drop_macro=True)
    assert report.chunks_redacted > 0 and len(report.dropped) == 2
    audit = verify_file(src, str(out), regions)
    assert audit.ok, audit.summary()
    with openslide.OpenSlide(str(out)) as slide, openslide.OpenSlide(src) as original:
        assert set(slide.associated_images) == {"thumbnail"}
        for level, ds in enumerate(slide.level_downsamples):
            size = (int(2000 / ds), int(1000 / ds))
            filled = np.asarray(
                slide.read_region((5500, 4500), level, size).convert("RGB")
            )
            assert filled.max() == 0
            a = np.asarray(
                slide.read_region((12000, 12000), level, (200, 200)).convert("RGB")
            )
            b = np.asarray(
                original.read_region((12000, 12000), level, (200, 200)).convert("RGB")
            )
            assert np.array_equal(a, b)


@pytest.mark.parametrize("compression", ["lzw", "jpeg"])
def test_planar_separate_pyramid(
    tmp_path: Path, rgb_image: np.ndarray, compression: str
) -> None:
    src = tmp_path / "planar.tif"
    kwargs = {"compression": compression, "planarconfig": "separate", "tile": (64, 64)}
    if compression == "jpeg":
        kwargs["compressionargs"] = {"level": 90}
    planar = np.ascontiguousarray(np.moveaxis(rgb_image, -1, 0))
    with tifffile.TiffWriter(src) as writer:
        writer.write(planar, subifds=1, photometric="rgb", **kwargs)
        writer.write(
            np.ascontiguousarray(planar[:, ::2, ::2]),
            subfiletype=1,
            photometric="rgb",
            **kwargs,
        )
    out = tmp_path / "out.tif"
    region = Region(100, 50, 120, 90)
    report = redact_file(str(src), str(out), [region], fill=(200, 30, 90))
    assert report.chunks_redacted > 0
    tolerance = 3 if compression == "jpeg" else 0
    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].planarconfig == 2
        for level in tf.series[0].levels:
            data = np.moveaxis(level.asarray(), 0, -1)
            scale = 420 / data.shape[1]
            x0, y0 = int(116 / scale), int(66 / scale)
            x1, y1 = int(204 / scale), int(124 / scale)
            window = data[y0:y1, x0:x1].astype(int)
            assert np.abs(window - np.array([200, 30, 90])).max() <= tolerance
    audit = verify_file(str(src), str(out), [region], fill=(200, 30, 90))
    assert audit.ok, audit.summary()


def test_dropping_label_and_macro_is_a_job_on_its_own(tmp_path: Path) -> None:
    src = write_aperio_like(tmp_path / "src.svs")
    out = tmp_path / "out.svs"
    report = redact_file(str(src), str(out), [], drop_label=True, drop_macro=True)
    assert len(report.dropped) == 2
    assert report.chunks_redacted == 0 and report.scrub is None
    with tifffile.TiffFile(src) as before, tifffile.TiffFile(out) as after:
        assert [page.shape for page in after.pages] == [(256, 384, 3), (64, 96, 3)]
        assert after.pages[0].description == before.pages[0].description
        assert np.array_equal(after.pages[0].asarray(), before.pages[0].asarray())
    audit = verify_file(str(src), str(out), [])
    assert audit.ok, audit.summary()


def test_nothing_to_do_is_refused(tmp_path: Path) -> None:
    src = write_aperio_like(tmp_path / "src.svs")
    with pytest.raises(RedactionError, match="Nothing to do"):
        redact_file(str(src), str(tmp_path / "out.svs"), [])
