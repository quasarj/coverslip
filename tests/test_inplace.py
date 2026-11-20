import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
import tifftools

from tiffredact import layout as layout_mod
from tiffredact import scrub
from tiffredact.cli import main
from tiffredact.inplace import InPlaceError, redact_in_place
from tiffredact.manifest import Manifest, default_manifest_path
from tiffredact.model import Region
from tiffredact.redact import redact_file
from tiffredact.storage import LocalFileStorage
from tiffredact.verify import verify_in_place

from conftest import (
    make_image,
    requires_openslide,
    requires_svs,
    sample_svs_paths,
    write_pyramid,
)
from test_scrub import APERIO_DESC

REGION = Region(100, 50, 120, 90)


def in_place_copy(src: Path, name: str = "inplace.tif") -> Path:
    return Path(shutil.copyfile(src, src.parent / name))


def level_arrays(path: Path) -> list[np.ndarray]:
    with tifffile.TiffFile(path) as tf:
        return [level.asarray() for series in tf.series for level in series.levels]


def page_tags(path: Path) -> list[dict[int, object]]:
    """Tag values of every page, without the chunk position arrays."""
    positions = {273, 279, 324, 325, 330}
    with tifffile.TiffFile(path) as tf:
        pages = [
            page
            for top in tf.pages
            for page in [top, *(top.pages if top.subifds else [])]
        ]
        return [
            {t.code: t.value for t in page.tags if t.code not in positions}
            for page in pages
        ]


def touched_source_chunks(src: Path, regions: list[Region]) -> list[bytes]:
    """Original bytes of the chunks a redaction of ``regions`` replaces."""
    from tiffredact.redact import plan_work

    layout = layout_mod.discover(layout_mod.read_info(str(src)))
    chunks = []
    with open(src, "rb") as fh:
        for ifd_offset, per_chunk in plan_work(layout, regions).items():
            for item in per_chunk.values():
                fh.seek(item.ifd.data_offsets[item.chunk.index])
                chunks.append(fh.read(item.ifd.data_bytecounts[item.chunk.index]))
    return chunks


def assert_matches_full_rewrite(src: Path, target: Path, **options) -> None:
    """Same pixels and same tags as the full rewrite; only the layout differs."""
    reference = src.parent / "reference.tif"
    redact_file(str(src), str(reference), overwrite=True, **options)
    for a, b in zip(level_arrays(target), level_arrays(reference), strict=True):
        assert np.array_equal(a, b)
    assert page_tags(target) == page_tags(reference)


@pytest.mark.parametrize("bigtiff", [False, True])
def test_lzw_pyramid_matches_full_rewrite(
    tmp_path: Path, rgb_image: np.ndarray, bigtiff: bool
) -> None:
    src = write_pyramid(
        tmp_path / "src.tif", rgb_image, predictor=True, bigtiff=bigtiff
    )
    target = in_place_copy(src)
    report = redact_in_place(str(target), [REGION])
    assert report.chunks_redacted > 0
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    assert audit.chunks_checked_filled == report.chunks_redacted
    assert_matches_full_rewrite(src, target, regions=[REGION])
    with tifffile.TiffFile(target) as tf:
        assert tf.is_bigtiff == bigtiff


def test_no_replaced_chunk_survives_anywhere(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, predictor=True)
    target = in_place_copy(src)
    originals = touched_source_chunks(src, [REGION])
    redact_in_place(str(target), [REGION])
    content = target.read_bytes()
    assert originals and all(chunk not in content for chunk in originals)


@pytest.fixture
def flat_pyramid(tmp_path: Path) -> Path:
    """Flat tiles compress to almost nothing, so painting a rectangle grows them."""
    flat = np.full((300, 420, 3), 90, np.uint8)
    return write_pyramid(tmp_path / "flat.tif", flat)


def test_jpeg_pyramid(tmp_path: Path, jpeg_pyramid: Path) -> None:
    target = in_place_copy(jpeg_pyramid)
    redact_in_place(str(target), [REGION])
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    assert_matches_full_rewrite(jpeg_pyramid, target, regions=[REGION])


def test_chunks_that_grow_move_to_the_end(tmp_path: Path, flat_pyramid: Path) -> None:
    target = in_place_copy(flat_pyramid)
    source_size = target.stat().st_size
    redact_in_place(str(target), [REGION])
    manifest = Manifest.load(default_manifest_path(str(target)))
    moved = [c for c in manifest.chunks if c.new_offset >= source_size]
    stayed = [c for c in manifest.chunks if c.new_offset == c.old_offset]
    assert moved and stayed and len(moved) + len(stayed) == len(manifest.chunks)
    assert all(c.new_length > c.old_length for c in moved)
    assert target.stat().st_size == manifest.final_size > source_size
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    assert_matches_full_rewrite(flat_pyramid, target, regions=[REGION])


def test_only_patched_and_zeroed_ranges_change(
    tmp_path: Path, flat_pyramid: Path
) -> None:
    target = in_place_copy(flat_pyramid)
    redact_in_place(str(target), [REGION])
    manifest = Manifest.load(default_manifest_path(str(target)))
    before = bytearray(flat_pyramid.read_bytes())
    after = bytearray(target.read_bytes()[: len(before)])
    touched = [(p.offset, p.offset + p.length) for p in manifest.patched]
    for start, end in touched + manifest.zeroed:
        before[start:end] = after[start:end] = bytes(len(before[start:end]))
    assert before == after


def test_strips_and_chain_pyramid(tmp_path: Path, rgb_image: np.ndarray) -> None:
    src = write_pyramid(
        tmp_path / "src.tif",
        rgb_image,
        tile=None,
        rowsperstrip=32,
        chain=True,
        compression="zlib",
    )
    target = in_place_copy(src)
    regions = [REGION, Region(300, 200, 100, 80)]
    redact_in_place(str(target), regions)
    assert verify_in_place(str(target)).ok
    assert_matches_full_rewrite(src, target, regions=regions)


def write_tagged(path: Path, image: np.ndarray) -> Path:
    extratags = [
        (315, "s", 0, "Dr. Who", False),  # Artist
        (316, "s", 0, "workstation-7", False),  # HostComputer
        (700, "B", 0, b"<x:xmpmeta>secret-xmp</x:xmpmeta>", False),  # XMP
        (269, "s", 0, "case-42.tif", False),  # DocumentName
    ]
    tifffile.imwrite(
        path,
        image,
        compression="lzw",
        software="unit-test",
        datetime=True,
        description="free text about a patient",
        metadata=None,
        extratags=extratags,
    )
    return path


def test_scrub_in_place_removes_tags_and_their_bytes(tmp_path: Path) -> None:
    image = make_image(64, 96)
    src = write_tagged(tmp_path / "src.tif", image)
    target = in_place_copy(src)
    report = redact_in_place(str(target), [], scrub=True)
    assert set(report.scrub.removed_tags) == {306, 315, 316, 700, 269}
    assert target.stat().st_size == src.stat().st_size
    content = target.read_bytes()
    for secret in (b"Dr. Who", b"workstation-7", b"secret-xmp", b"case-42", b"patient"):
        assert secret in src.read_bytes() and secret not in content
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    assert np.array_equal(tifffile.imread(target), image)
    assert_matches_full_rewrite(src, target, regions=[], scrub=True)


def test_scrub_and_redact_together_with_kept_tag(tmp_path: Path) -> None:
    src = write_tagged(tmp_path / "src.tif", make_image(64, 96))
    target = in_place_copy(src)
    region = Region(10, 10, 30, 30)
    redact_in_place(str(target), [region], scrub=True, keep_tags=[315])
    with tifffile.TiffFile(target) as tf:
        assert tf.pages[0].tags[315].value == "Dr. Who"
        assert 316 not in tf.pages[0].tags
        assert tf.pages[0].asarray()[10:40, 10:40].max() == 0
    assert verify_in_place(str(target)).ok
    assert_matches_full_rewrite(
        src, target, regions=[region], scrub=True, keep_tags=[315]
    )


def write_aperio_like(path: Path) -> Path:
    """Tiled main image, thumbnail, label (subfiletype 1), macro (subfiletype 9)."""
    image = make_image(256, 384)
    label = np.full((40, 60, 3), 77, np.uint8)
    macro = np.full((50, 120, 3), 201, np.uint8)
    with tifffile.TiffWriter(path) as writer:
        common = {"photometric": "rgb", "metadata": None, "compression": "lzw"}
        writer.write(image, tile=(64, 64), description=APERIO_DESC, **common)
        writer.write(image[::4, ::4], description=APERIO_DESC, **common)
        writer.write(label, subfiletype=1, description="Aperio label", **common)
        writer.write(macro, subfiletype=9, description="Aperio macro", **common)
    return path


def test_drop_label_and_macro_in_place(tmp_path: Path) -> None:
    src = write_aperio_like(tmp_path / "src.svs")
    with tifffile.TiffFile(src) as tf:
        label_strip = tf.pages[2].dataoffsets[0], tf.pages[2].databytecounts[0]
    label_bytes = src.read_bytes()[label_strip[0] : sum(label_strip)]
    target = in_place_copy(src, "inplace.svs")
    options = {"drop_label": True, "drop_macro": True, "scrub": True}
    region = Region(64, 64, 100, 80)
    report = redact_in_place(str(target), [region], **options)
    assert len(report.dropped) == 2
    assert label_bytes not in target.read_bytes()
    with tifffile.TiffFile(target) as tf:
        assert [page.shape for page in tf.pages] == [(256, 384, 3), (64, 96, 3)]
        assert "Filename" not in tf.pages[0].description
        assert "AppMag = 20" in tf.pages[0].description
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    assert_matches_full_rewrite(src, target, regions=[region], **options)


def test_deduplicated_tiles_are_not_clobbered(tmp_path: Path) -> None:
    """Tiles that share one slot: redacting one must leave the others intact."""
    flat = np.full((128, 256, 3), 90, np.uint8)
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, flat, tile=(64, 64), compression="lzw", metadata=None)
    src = tmp_path / "src.tif"
    tifftools.write_tiff(tifftools.read_tiff(str(plain)), str(src), dedup=True)
    with tifffile.TiffFile(src) as tf:
        assert len(set(tf.pages[0].dataoffsets)) == 1
    target = in_place_copy(src)
    region = Region(0, 0, 32, 32)
    redact_in_place(str(target), [region])
    data = tifffile.imread(target)
    expected = flat.copy()
    expected[0:32, 0:32] = 0
    assert np.array_equal(data, expected)
    manifest = Manifest.load(default_manifest_path(str(target)))
    assert [c.old_slot_shared for c in manifest.chunks] == [True]
    assert verify_in_place(str(target)).ok


def test_repeated_run_writes_nothing(tmp_path: Path, rgb_image: np.ndarray) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, predictor=True)
    target = in_place_copy(src)
    redact_in_place(str(target), [REGION])
    once = target.read_bytes()
    with pytest.raises(InPlaceError, match="Manifest"):
        redact_in_place(str(target), [REGION])
    redact_in_place(str(target), [REGION], overwrite_manifest=True)
    assert target.read_bytes() == once
    manifest = Manifest.load(default_manifest_path(str(target)))
    assert manifest.patched == [] and manifest.zeroed == []
    assert verify_in_place(str(target)).ok


def test_classic_tiff_refuses_to_grow_past_4gb(
    tmp_path: Path, flat_pyramid: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = in_place_copy(flat_pyramid)
    untouched = target.read_bytes()
    monkeypatch.setattr(LocalFileStorage, "size", lambda self: (1 << 32) - 16)
    with pytest.raises(InPlaceError, match="4 GB"):
        redact_in_place(str(target), [REGION])
    assert target.read_bytes() == untouched
    assert not Path(default_manifest_path(str(target))).exists()


def test_ome_drop_label_is_refused_before_writing(tmp_path: Path) -> None:
    from tiffredact.redact import RedactionError

    src = write_pyramid(tmp_path / "src.ome.tif", make_image(200, 260), ome=True)
    untouched = src.read_bytes()
    with pytest.raises(RedactionError):
        redact_in_place(str(src), [Region(10, 10, 50, 50)], drop_label=True)
    assert src.read_bytes() == untouched


def test_ome_scrub_in_place(tmp_path: Path) -> None:
    src = write_pyramid(tmp_path / "src.ome.tif", make_image(200, 260), ome=True)
    target = in_place_copy(src, "inplace.ome.tif")
    region = Region(70, 70, 60, 60)
    redact_in_place(str(target), [region], scrub=True)
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    info = layout_mod.read_info(str(target))
    assert scrub.scrub_problems(info, "ome") == []
    assert_matches_full_rewrite(src, target, regions=[region], scrub=True)


# --- the audit must notice a file that is not what the manifest describes ----


@pytest.fixture
def redacted(tmp_path: Path, flat_pyramid: Path) -> Path:
    target = in_place_copy(flat_pyramid)
    redact_in_place(str(target), [REGION])
    return target


def overwrite(path: Path, offset: int, data: bytes) -> None:
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(data)


def test_verify_detects_residue_in_a_zeroed_range(redacted: Path) -> None:
    manifest = Manifest.load(default_manifest_path(str(redacted)))
    start, _ = manifest.zeroed[0]
    overwrite(redacted, start, b"\x01")
    audit = verify_in_place(str(redacted))
    assert any("should be zeroed" in p for p in audit.problems)


def test_verify_detects_restored_original_chunk(
    redacted: Path, flat_pyramid: Path
) -> None:
    manifest = Manifest.load(default_manifest_path(str(redacted)))
    record = next(c for c in manifest.chunks if c.new_offset != c.old_offset)
    original = flat_pyramid.read_bytes()
    old_slot = slice(record.old_offset, record.old_offset + record.old_length)
    overwrite(redacted, record.old_offset, original[old_slot])
    audit = verify_in_place(str(redacted))
    assert any("Original bytes" in p for p in audit.problems)


def test_verify_detects_changed_patch_and_size(redacted: Path) -> None:
    manifest = Manifest.load(default_manifest_path(str(redacted)))
    patch = manifest.patched[0]
    overwrite(redacted, patch.offset + patch.length // 2, b"\xff\xfe\xfd")
    audit = verify_in_place(str(redacted))
    assert any("Patched range" in p for p in audit.problems)
    with open(redacted, "ab") as fh:
        fh.write(b"extra")
    assert any("manifest expects" in p for p in verify_in_place(str(redacted)).problems)


def test_verify_detects_unredacted_file(tmp_path: Path, jpeg_pyramid: Path) -> None:
    """A manifest claiming a redaction that never reached the file."""
    target = in_place_copy(jpeg_pyramid)
    probe = in_place_copy(jpeg_pyramid, "probe.tif")
    redact_in_place(str(probe), [REGION])
    manifest = Manifest.load(default_manifest_path(str(probe)))
    manifest.final_size = target.stat().st_size
    manifest.save(default_manifest_path(str(target)))
    assert not verify_in_place(str(target)).ok


# --- CLI ----------------------------------------------------------------------


def test_cli_in_place_round_trip(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image, predictor=True)
    target = in_place_copy(src)
    manifest = tmp_path / "audit.json"
    common = [str(target), "--inplace", "--manifest", str(manifest)]
    assert main(["redact", *common, "-r", "100,50,120,90", "--scrub"]) == 0
    assert "in place:" in capsys.readouterr().out
    assert manifest.exists()
    assert main(["verify", *common]) == 0
    assert "OK" in capsys.readouterr().out
    assert_matches_full_rewrite(src, target, regions=[REGION], scrub=True)


def test_cli_rejects_conflicting_destinations(tmp_path: Path) -> None:
    src = write_pyramid(tmp_path / "src.tif", make_image(100, 100), levels=1)
    region = ["-r", "1,1,10,10"]
    with pytest.raises(SystemExit, match="not both"):
        main(["redact", str(src), str(tmp_path / "out.tif"), "--in-place", *region])
    with pytest.raises(SystemExit, match="OUTPUT is required"):
        main(["redact", str(src), *region])
    with pytest.raises(SystemExit, match="manifest"):
        main(["verify", str(src), "--in-place", *region])


# --- real slides ----------------------------------------------------------------


@requires_svs
@requires_openslide
def test_aperio_sample_in_place(tmp_path: Path) -> None:
    import openslide

    src = min(sample_svs_paths(), key=lambda p: Path(p).stat().st_size)
    target = Path(shutil.copyfile(src, tmp_path / "inplace.svs"))
    regions = [Region(5000, 4000, 3000, 2000)]
    options = {"drop_label": True, "drop_macro": True, "scrub": True}
    report = redact_in_place(str(target), regions, **options)
    assert report.chunks_redacted > 0 and len(report.dropped) == 2
    audit = verify_in_place(str(target))
    assert audit.ok, audit.summary()
    reference = tmp_path / "reference.svs"
    redact_file(src, str(reference), regions, **options)
    with (
        openslide.OpenSlide(str(target)) as slide,
        openslide.OpenSlide(str(reference)) as expected,
    ):
        assert set(slide.associated_images) == {"thumbnail"}
        assert dict(slide.properties) == dict(expected.properties)
        for level, ds in enumerate(slide.level_downsamples):
            size = (int(4000 / ds), int(3000 / ds))
            a = slide.read_region((4500, 3500), level, size)
            b = expected.read_region((4500, 3500), level, size)
            assert np.array_equal(np.asarray(a), np.asarray(b))
