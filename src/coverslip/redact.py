"""Orchestrate a redaction: discover layout, map regions, rewrite chunks, write."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from coverslip import layout as layout_mod
from coverslip.codecs.dispatch import redact_chunk
from coverslip.codecs.fill import Rgb
from coverslip.geometry import intersecting_chunks, resolve_regions, scale_region
from coverslip.model import Chunk, ImageIFD, Layout, Region, RegionInput
from coverslip.scrub import ScrubReport, scrub_info
from coverslip.writer import (
    Placement,
    ScratchStore,
    VirtualSource,
    apply_replacements,
    drop_ifds,
    write_output,
)

logger = logging.getLogger(__name__)

ReadRange = Callable[[int, int], bytes]
"""``read(offset, length)`` against the slide being redacted."""
ChunkSink = Callable[[ImageIFD, int, bytes, bytes], None]
"""Receives ``(level, chunk index, original bytes, replacement bytes)``."""


class RedactionError(Exception):
    """The requested redaction cannot be carried out."""


@dataclass
class ChunkWork:
    ifd: ImageIFD
    chunk: Chunk
    rects: list[Region] = field(default_factory=list)


@dataclass
class LevelReport:
    offset: int
    role: str
    width: int
    height: int
    chunks_total: int
    chunks_redacted: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


@dataclass
class RedactionReport:
    input_path: str
    output_path: str
    regions: list[Region]
    levels: list[LevelReport] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    scrub: ScrubReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def chunks_redacted(self) -> int:
        return sum(level.chunks_redacted for level in self.levels)

    def summary(self) -> str:
        lines = [f"{self.input_path} -> {self.output_path}"]
        for level in self.levels:
            if level.chunks_redacted == 0:
                continue
            lines.append(
                f"  {level.role:9s} {level.width}x{level.height}: "
                f"{level.chunks_redacted}/{level.chunks_total} chunks rewritten, "
                f"{level.bytes_in} -> {level.bytes_out} bytes"
            )
        for name in self.dropped:
            lines.append(f"  dropped {name}")
        lines.append(f"  total chunks rewritten: {self.chunks_redacted}")
        if self.scrub is not None:
            lines.append("  metadata scrub:")
            lines.append(self.scrub.summary())
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def plan_work(layout: Layout, regions: list[Region]) -> dict[int, dict[int, ChunkWork]]:
    """For every level of every image, find chunks each region touches."""
    work: dict[int, dict[int, ChunkWork]] = defaultdict(dict)
    for image in layout.images:
        full = image.full
        for level in image.levels:
            _check_supported(level)
            for region in regions:
                scaled = scale_region(region, full, level)
                if scaled is None:
                    continue
                for chunk, local in intersecting_chunks(level, scaled):
                    item = work[level.offset].setdefault(
                        chunk.index, ChunkWork(level, chunk)
                    )
                    item.rects.append(local)
    return dict(work)


def _check_supported(ifd: ImageIFD) -> None:
    if len(ifd.data_offsets) != ifd.chunk_count:
        raise RedactionError(
            f"IFD at {ifd.offset} has {len(ifd.data_offsets)} data chunks but "
            f"geometry implies {ifd.chunk_count}"
        )


def _validate_regions(layout: Layout, regions: list[Region]) -> None:
    for image in layout.images:
        full = image.full
        for region in regions:
            if region.clip(full.width, full.height) is None:
                raise RedactionError(
                    f"Region {region} lies outside the {full.width}x{full.height} image"
                )


def _read_chunk(read: ReadRange, ifd: ImageIFD, index: int) -> bytes:
    data = read(ifd.data_offsets[index], ifd.data_bytecounts[index])
    if len(data) != ifd.data_bytecounts[index]:
        raise RedactionError(f"Short read for chunk {index} of IFD at {ifd.offset}")
    return data


def file_reader(fh: BinaryIO) -> ReadRange:
    def read(offset: int, length: int) -> bytes:
        fh.seek(offset)
        return fh.read(length)

    return read


def load_plan(
    input_path: str,
    regions: Sequence[RegionInput],
    scrub: bool,
    drop_label: bool,
    drop_macro: bool,
) -> tuple[dict[str, Any], Layout, list[Region], dict[int, dict[int, ChunkWork]]]:
    """Read the IFD tree, classify it and map ``regions`` onto chunks.

    Also returns the regions in full-resolution coordinates, which is the form
    reports and manifests record.
    """
    if not regions and not scrub:
        raise RedactionError("At least one region is required unless scrubbing")
    info = layout_mod.read_info(input_path)
    layout = layout_mod.discover(info)
    if (drop_label or drop_macro) and layout.vendor == "ome":
        raise RedactionError(
            "Dropping label/macro images from OME-TIFF is not supported: the "
            "OME-XML references IFDs by index and would need rewriting"
        )
    resolved = resolve_regions(layout, regions)
    _validate_regions(layout, resolved)
    return info, layout, resolved, plan_work(layout, resolved)


def rewrite_chunks(
    read: ReadRange,
    layout: Layout,
    work: dict[int, dict[int, ChunkWork]],
    fill: Rgb,
    use_dct: bool,
    report: RedactionReport,
    sink: ChunkSink,
) -> None:
    """Redact every planned chunk and hand (level, index, old, new) to ``sink``."""
    for image in layout.images:
        for level in image.levels:
            level_report = LevelReport(
                level.offset,
                level.role.value,
                level.width,
                level.height,
                level.chunk_count,
            )
            report.levels.append(level_report)
            for index, item in sorted(work.get(level.offset, {}).items()):
                original = _read_chunk(read, level, index)
                replacement = redact_chunk(
                    level, original, item.chunk, item.rects, fill, use_dct
                )
                sink(level, index, original, replacement)
                level_report.chunks_redacted += 1
                level_report.bytes_in += len(original)
                level_report.bytes_out += len(replacement)
            logger.info(
                "%s %dx%d: %d chunks rewritten",
                level.role.value,
                level.width,
                level.height,
                level_report.chunks_redacted,
            )


def apply_metadata_changes(
    info: dict[str, Any],
    layout: Layout,
    report: RedactionReport,
    drop_label: bool,
    drop_macro: bool,
    scrub: bool,
    keep_tags: Iterable[int],
) -> set[int]:
    """Drop label/macro IFDs and scrub tags in ``info``; returns dropped offsets."""
    to_drop: set[int] = set()
    if drop_label:
        to_drop.update(ifd.offset for ifd in layout.label)
        report.dropped.extend(f"label ifd@{ifd.offset}" for ifd in layout.label)
    if drop_macro:
        to_drop.update(ifd.offset for ifd in layout.macro)
        report.dropped.extend(f"macro ifd@{ifd.offset}" for ifd in layout.macro)
    if to_drop:
        drop_ifds(info, to_drop)
    if scrub:
        report.scrub = scrub_info(info, layout.vendor, keep_tags)
    return to_drop


def redact_file(
    input_path: str,
    output_path: str,
    regions: Sequence[RegionInput],
    fill: Rgb = (0, 0, 0),
    drop_label: bool = False,
    drop_macro: bool = False,
    use_dct: bool = True,
    overwrite: bool = False,
    scrub: bool = False,
    keep_tags: Iterable[int] = (),
) -> RedactionReport:
    """Redact ``regions`` into a new file.

    A bare :class:`Region` is in full-resolution coordinates; a
    :class:`RegionSpec` may name the pyramid level it was drawn on.

    With ``scrub`` the metadata is cleaned as well; ``regions`` may then be
    empty for a metadata-only rewrite.
    """
    info, layout, resolved, work = load_plan(
        input_path, regions, scrub, drop_label, drop_macro
    )
    report = RedactionReport(input_path, output_path, resolved)
    original_size = os.path.getsize(input_path)
    replacements: dict[int, dict[int, Placement]] = {}

    with ScratchStore(
        base=original_size, directory=os.path.dirname(os.path.abspath(output_path))
    ) as scratch:

        def store(level: ImageIFD, index: int, _old: bytes, new: bytes) -> None:
            replacements.setdefault(level.offset, {})[index] = scratch.add(new)

        with open(input_path, "rb") as fh:
            rewrite_chunks(file_reader(fh), layout, work, fill, use_dct, report, store)
        scratch.flush()
        apply_metadata_changes(
            info, layout, report, drop_label, drop_macro, scrub, keep_tags
        )

        source = VirtualSource(input_path, scratch.path, original_size)
        try:
            apply_replacements(info, replacements, source)
            write_output(info, output_path, overwrite=overwrite)
        finally:
            source.close()
    return report
