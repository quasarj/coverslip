"""Check a redacted file against its source.

* Every pixel inside the mapped region on every level equals the fill colour.
* Every chunk outside the regions is byte-identical to the source.
* No replaced source chunk survives anywhere in the output (sampled).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
import mmap
import os
from dataclasses import dataclass, field

import numpy as np

from tiffredact import layout as layout_mod
from tiffredact.codecs import pixel
from tiffredact.codecs.fill import Rgb, chunk_fill_values
from tiffredact.geometry import chunk_at, intersecting_chunks, scale_region
from tiffredact.model import ImageIFD, Layout, Region
from tiffredact.scrub import scrub_problems

logger = logging.getLogger(__name__)

RESIDUE_SAMPLE = 64
LOSSY_INSET = 16
LOSSY_COMPRESSIONS = {
    pixel.COMPRESSION_JPEG,
    *pixel.JPEG2000_COMPRESSIONS,
    pixel.COMPRESSION_WEBP,
}


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
        raise ValueError("Source and output have different image counts")
    for s_img, d_img in zip(src.images, dst.images):
        if len(s_img.levels) != len(d_img.levels):
            raise ValueError("Source and output have different level counts")
        for s_level, d_level in zip(s_img.levels, d_img.levels):
            pairs.append((s_img.full, s_level, d_level))
    return pairs


def verify_file(
    input_path: str,
    output_path: str,
    regions: list[Region],
    fill: Rgb = (0, 0, 0),
    scrubbed: bool = False,
    keep_tags: Iterable[int] = (),
) -> VerifyReport:
    report = VerifyReport()
    src_layout = layout_mod.discover(layout_mod.read_info(input_path))
    dst_info = layout_mod.read_info(output_path)
    dst_layout = layout_mod.discover(dst_info)
    if scrubbed:
        report.problems.extend(scrub_problems(dst_info, dst_layout.vendor, keep_tags))
    replaced: list[bytes] = []
    untouched_hashes: set[bytes] = set()
    with open(input_path, "rb") as src_fh, open(output_path, "rb") as dst_fh:
        for full, s_level, d_level in _match_levels(src_layout, dst_layout):
            touched: dict[int, list[Region]] = {}
            for region in regions:
                scaled = scale_region(region, full, s_level)
                if scaled is None:
                    continue
                for chunk, local in intersecting_chunks(s_level, scaled):
                    touched.setdefault(chunk.index, []).append(local)
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
    tolerance = _tolerance(d_level)
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
        chunk = chunk_at(d_level, index)
        try:
            array = pixel.decode_chunk(d_level, d_bytes, chunk)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the audit
            report.problems.append(
                f"Chunk {index} of ifd@{d_level.offset} failed to decode: {exc}"
            )
            continue
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
                break
        else:
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
