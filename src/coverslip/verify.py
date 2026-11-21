"""Check a redacted file against its source.

* Every pixel inside the mapped region on every level equals the fill colour.
* Every chunk outside the regions is byte-identical to the source.
* No replaced source chunk survives anywhere in the output (sampled).

A file redacted in place has no source to compare against;
:func:`verify_in_place` audits it from the manifest instead, reading only the
byte ranges the redaction wrote or zeroed.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
import mmap
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from coverslip import layout as layout_mod
from coverslip.codecs import pixel
from coverslip.codecs.fill import Rgb, chunk_fill_values
from coverslip.geometry import (
    chunk_at,
    intersecting_chunks,
    resolve_regions,
    scale_region,
)
from coverslip.inplace import TiffFormat, chunk_extents, tag_data_extent, walk_ifds
from coverslip.manifest import Manifest, default_manifest_path, sha256_hex
from coverslip.model import ImageIFD, Layout, Region, RegionInput
from coverslip.patches import merge_ranges, ranges_intersect
from coverslip.scrub import scrub_problems
from coverslip.storage import LocalFileStorage, Storage

logger = logging.getLogger(__name__)

RESIDUE_SAMPLE = 64
READ_BLOCK = 1 << 20
LOSSY_INSET = 16
LOSSY_COMPRESSIONS = {
    pixel.COMPRESSION_JPEG,
    *pixel.JPEG2000_COMPRESSIONS,
    pixel.COMPRESSION_WEBP,
}


class VerifyError(ValueError):
    """The two files cannot be compared level by level."""


@dataclass
class VerifyReport:
    problems: list[str] = field(default_factory=list)
    chunks_checked_identical: int = 0
    chunks_checked_filled: int = 0
    residue_probes: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [
            f"identical untouched chunks: {self.chunks_checked_identical}",
            f"filled chunks verified:     {self.chunks_checked_filled}",
            f"residue probes:             {self.residue_probes}",
        ]
        if self.problems:
            lines.append(f"PROBLEMS ({len(self.problems)}):")
            lines.extend(f"  - {p}" for p in self.problems[:50])
        else:
            lines.append("OK")
        return "\n".join(lines)


def _tolerance(ifd: ImageIFD) -> int:
    return 2 if ifd.compression in LOSSY_COMPRESSIONS else 0


def _inset(rect: Region, amount: int) -> Region | None:
    """Shrink a rect on every side; lossy codecs may ring at the painted edge."""
    if rect.width <= 2 * amount or rect.height <= 2 * amount:
        return None
    return Region(
        rect.x + amount,
        rect.y + amount,
        rect.width - 2 * amount,
        rect.height - 2 * amount,
    )


def _match_levels(
    src: Layout, dst: Layout
) -> list[tuple[ImageIFD, ImageIFD, ImageIFD]]:
    pairs = []
    if len(src.images) != len(dst.images):
        raise VerifyError("Source and output have different image counts")
    for s_img, d_img in zip(src.images, dst.images):
        if len(s_img.levels) != len(d_img.levels):
            raise VerifyError("Source and output have different level counts")
        for s_level, d_level in zip(s_img.levels, d_img.levels):
            pairs.append((s_img.full, s_level, d_level))
    return pairs


def verify_file(
    input_path: str,
    output_path: str,
    regions: Sequence[RegionInput],
    fill: Rgb = (0, 0, 0),
    scrubbed: bool = False,
    keep_tags: Iterable[int] = (),
) -> VerifyReport:
    report = VerifyReport()
    src_layout = layout_mod.discover(layout_mod.read_info(input_path))
    full_regions = resolve_regions(src_layout, regions)
    dst_info = layout_mod.read_info(output_path)
    dst_layout = layout_mod.discover(dst_info)
    if scrubbed:
        report.problems.extend(scrub_problems(dst_info, dst_layout.vendor, keep_tags))
    replaced: list[bytes] = []
    untouched_hashes: set[bytes] = set()
    with open(input_path, "rb") as src_fh, open(output_path, "rb") as dst_fh:
        for full, s_level, d_level in _match_levels(src_layout, dst_layout):
            touched = _touched_chunks(full_regions, full, s_level)
            _check_level(
                src_fh,
                dst_fh,
                s_level,
                d_level,
                touched,
                fill,
                report,
                replaced,
                untouched_hashes,
            )
    _check_residue(output_path, replaced, untouched_hashes, report)
    return report


def _touched_chunks(
    regions: list[Region], full: ImageIFD, level: ImageIFD
) -> dict[int, list[Region]]:
    """Chunk index -> chunk-local rectangles the regions cover on ``level``."""
    touched: dict[int, list[Region]] = {}
    for region in regions:
        scaled = scale_region(region, full, level)
        if scaled is None:
            continue
        for chunk, local in intersecting_chunks(level, scaled):
            touched.setdefault(chunk.index, []).append(local)
    return touched


def _read(fh, ifd: ImageIFD, index: int) -> bytes:
    fh.seek(ifd.data_offsets[index])
    return fh.read(ifd.data_bytecounts[index])


def _check_level(
    src_fh, dst_fh, s_level, d_level, touched, fill, report, replaced, untouched_hashes
) -> None:
    if (s_level.width, s_level.height, s_level.chunk_count) != (
        d_level.width,
        d_level.height,
        d_level.chunk_count,
    ):
        report.problems.append(f"Level geometry differs at ifd@{s_level.offset}")
        return
    for index in range(s_level.chunk_count):
        s_bytes = _read(src_fh, s_level, index)
        d_bytes = _read(dst_fh, d_level, index)
        rects = touched.get(index)
        if not rects:
            if s_bytes != d_bytes:
                report.problems.append(
                    f"Untouched chunk {index} of {s_level.role.value} "
                    f"{s_level.width}x{s_level.height} differs from source"
                )
            else:
                report.chunks_checked_identical += 1
            untouched_hashes.add(hashlib.sha256(s_bytes).digest())
            continue
        if len(replaced) < RESIDUE_SAMPLE:
            replaced.append(s_bytes)
        _check_filled(d_level, index, d_bytes, rects, fill, report)


def _check_filled(
    d_level: ImageIFD,
    index: int,
    d_bytes: bytes,
    rects: list[Region],
    fill: Rgb,
    report: VerifyReport,
) -> None:
    """Decode one output chunk and check its rectangles hold the fill colour."""
    tolerance = _tolerance(d_level)
    chunk = chunk_at(d_level, index)
    try:
        array = pixel.decode_chunk(d_level, d_bytes, chunk)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the audit
        report.problems.append(
            f"Chunk {index} of ifd@{d_level.offset} failed to decode: {exc}"
        )
        return
    expected = chunk_fill_values(
        fill,
        d_level.photometric,
        d_level.samples_per_pixel,
        d_level.planar_config,
        chunk.plane,
    )
    scaled = pixel.scale_fill(expected, array.dtype)
    inset = LOSSY_INSET if d_level.compression in LOSSY_COMPRESSIONS else 0
    for rect in rects:
        clipped = rect.clip(array.shape[1], array.shape[0])
        if clipped is not None and inset:
            clipped = _inset(clipped, inset)
        if clipped is None:
            continue
        window = array[clipped.y : clipped.y1, clipped.x : clipped.x1, :].astype(
            np.int64
        )
        diff = np.abs(window - np.asarray(scaled, dtype=np.int64)).max()
        if diff > tolerance:
            report.problems.append(
                f"Chunk {index} of {d_level.role.value} {d_level.width}x{d_level.height}: "
                f"region not filled (max deviation {int(diff)})"
            )
            return
    report.chunks_checked_filled += 1


def _check_residue(
    output_path: str,
    replaced: list[bytes],
    untouched_hashes: set[bytes],
    report: VerifyReport,
) -> None:
    """Search the output for complete replaced chunks.

    Chunks whose bytes also occur as an untouched chunk (blank background tiles
    are byte-identical to each other) are skipped, since finding them proves
    nothing.
    """
    probes = [b for b in replaced if hashlib.sha256(b).digest() not in untouched_hashes]
    if not probes:
        return
    size = os.path.getsize(output_path)
    with (
        open(output_path, "rb") as fh,
        mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ) as mm,
    ):
        for probe in probes:
            report.residue_probes += 1
            if mm.find(probe) != -1:
                report.problems.append(
                    "Bytes of a replaced source chunk were found in the output"
                )
                return


# --- in-place audit ----------------------------------------------------------


@dataclass
class InPlaceVerifyReport(VerifyReport):
    patched_ranges_checked: int = 0
    zeroed_bytes_checked: int = 0
    old_slots_checked: int = 0

    def summary(self) -> str:
        lines = [
            f"patched ranges read back:   {self.patched_ranges_checked}",
            f"zeroed bytes read back:     {self.zeroed_bytes_checked}",
            f"filled chunks verified:     {self.chunks_checked_filled}",
            f"old chunk slots checked:    {self.old_slots_checked}",
        ]
        if self.problems:
            lines.append(f"PROBLEMS ({len(self.problems)}):")
            lines.extend(f"  - {p}" for p in self.problems[:50])
        else:
            lines.append("OK")
        return "\n".join(lines)


def verify_in_place(path: str, manifest_path: str | None = None) -> InPlaceVerifyReport:
    """Audit a file redacted in place, reading only the ranges that changed.

    Regions, fill colour and scrub settings come from the manifest written by
    the redaction; there is no source file to compare against.
    """
    manifest = Manifest.load(manifest_path or default_manifest_path(path))
    report = InPlaceVerifyReport()
    with LocalFileStorage(path) as storage:
        if storage.size() != manifest.final_size:
            report.problems.append(
                f"File is {storage.size()} bytes; the manifest expects "
                f"{manifest.final_size}"
            )
            return report
        _check_patched_ranges(storage, manifest, report)
        _check_zeroed_ranges(storage, manifest, report)
        _check_old_slots(storage, manifest, report)
        info = layout_mod.read_info(path)
        layout = layout_mod.discover(info)
        _check_structure(info, manifest, report)
        _check_chunk_records(layout, manifest, report)
        _check_regions_filled(storage, layout, manifest, report)
    if manifest.scrubbed:
        report.problems.extend(scrub_problems(info, layout.vendor, manifest.keep_tags))
    return report


def _check_patched_ranges(
    storage: Storage, manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    for item in manifest.patched:
        if sha256_hex(storage.read(item.offset, item.length)) != item.sha256:
            report.problems.append(
                f"Patched range at {item.offset} (+{item.length}) does not hold "
                "the bytes the redaction wrote"
            )
        report.patched_ranges_checked += 1


def _check_zeroed_ranges(
    storage: Storage, manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    for start, end in manifest.zeroed:
        if not _is_zero(storage, start, end):
            report.problems.append(
                f"Range {start}..{end} should be zeroed but still holds data"
            )
        report.zeroed_bytes_checked += end - start


def _is_zero(storage: Storage, start: int, end: int) -> bool:
    position = start
    while position < end:
        block = storage.read(position, min(READ_BLOCK, end - position))
        if block.count(0) != len(block):
            return False
        position += len(block)
    return True


def _check_old_slots(
    storage: Storage, manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    """The only place a replaced chunk's bytes could survive is its old slot."""
    for record in manifest.chunks:
        if record.old_length == 0 or record.old_slot_shared:
            continue
        old_bytes = storage.read(record.old_offset, record.old_length)
        if sha256_hex(old_bytes) == record.old_sha256:
            report.problems.append(
                f"Original bytes of chunk {record.index} of ifd@{record.ifd_offset} "
                f"are still at offset {record.old_offset}"
            )
        report.old_slots_checked += 1


def _check_structure(
    info: dict[str, Any], manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    """Every IFD, tag value and chunk lies in the file and off zeroed ranges."""
    zeroed = merge_ranges(manifest.zeroed)
    fmt = TiffFormat(bool(info["bigtiff"]), info["endianPack"])
    for ifd in walk_ifds(info["ifds"]):
        offset = int(ifd["offset"])
        if offset in manifest.dropped_ifds:
            report.problems.append(f"Dropped IFD at {offset} is still linked")
        extents = [(offset, offset + fmt.table_length(ifd["tagcount"]))]
        extents += [
            extent
            for entry in ifd["tags"].values()
            if (extent := tag_data_extent(entry)) is not None
        ]
        extents += chunk_extents(ifd)
        for extent in extents:
            if extent[1] > manifest.final_size:
                report.problems.append(
                    f"IFD at {offset} references {extent[0]}..{extent[1]}, "
                    "past the end of the file"
                )
            elif ranges_intersect(zeroed, extent):
                report.problems.append(
                    f"IFD at {offset} references {extent[0]}..{extent[1]}, "
                    "which was zeroed"
                )


def _check_chunk_records(
    layout: Layout, manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    """The re-parsed IFDs point at the replaced chunks' new positions."""
    by_offset = {ifd.offset: ifd for ifd in layout.all_ifds()}
    for record in manifest.chunks:
        ifd = by_offset.get(record.ifd_offset)
        if ifd is None or record.index >= len(ifd.data_offsets):
            report.problems.append(
                f"Replaced chunk {record.index} of ifd@{record.ifd_offset} is "
                "not in the file"
            )
            continue
        position = (ifd.data_offsets[record.index], ifd.data_bytecounts[record.index])
        if position != (record.new_offset, record.new_length):
            report.problems.append(
                f"Chunk {record.index} of ifd@{record.ifd_offset} does not point "
                "at its replacement"
            )


def _check_regions_filled(
    storage: Storage, layout: Layout, manifest: Manifest, report: InPlaceVerifyReport
) -> None:
    for image in layout.images:
        for level in image.levels:
            touched = _touched_chunks(manifest.regions, image.full, level)
            for index, rects in sorted(touched.items()):
                data = storage.read(
                    level.data_offsets[index], level.data_bytecounts[index]
                )
                _check_filled(level, index, data, rects, manifest.fill, report)
