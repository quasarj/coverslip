"""Text report of a slide: its pyramid levels and associated images."""

from __future__ import annotations

from dataclasses import dataclass

from tifftools import constants

from coverslip.chunks import is_empty
from coverslip.model import ImageIFD, Layout


@dataclass(frozen=True)
class LevelSummary:
    """One row of the report."""

    name: str
    role: str
    width: int
    height: int
    tiled: bool
    chunk_width: int
    chunk_height: int
    chunks_across: int
    chunks_down: int
    chunk_count: int
    empty_chunks: int
    compression: str
    photometric: str
    samples: int
    bits: int
    downsample: float | None
    stored_bytes: int
    offset: int


def compression_name(code: int) -> str:
    known = constants.Compression.get(code)
    return known.name if known is not None else str(code)


def photometric_name(code: int) -> str:
    known = constants.Photometric.get(code)
    return known.name if known is not None else str(code)


def summarize(name: str, ifd: ImageIFD, full: ImageIFD | None) -> LevelSummary:
    return LevelSummary(
        name=name,
        role=ifd.role.value,
        width=ifd.width,
        height=ifd.height,
        tiled=ifd.tiled,
        chunk_width=ifd.chunk_width,
        chunk_height=ifd.chunk_height,
        chunks_across=ifd.chunks_across,
        chunks_down=ifd.chunks_down,
        chunk_count=ifd.chunk_count,
        empty_chunks=sum(is_empty(ifd, i) for i in range(ifd.chunk_count)),
        compression=compression_name(ifd.compression),
        photometric=photometric_name(ifd.photometric),
        samples=ifd.samples_per_pixel,
        bits=ifd.bits_per_sample,
        downsample=full.width / ifd.width if full is not None else None,
        stored_bytes=sum(ifd.data_bytecounts),
        offset=ifd.offset,
    )


@dataclass(frozen=True)
class LayoutEntry:
    """One image of a slide, under the name the report gives it.

    ``image_index`` and ``level_index`` place a pyramid level; both are
    ``None`` for an associated image, which belongs to no pyramid.
    """

    name: str
    ifd: ImageIFD
    full: ImageIFD | None
    image_index: int | None = None
    level_index: int | None = None


def _level_entries(layout: Layout) -> list[LayoutEntry]:
    entries = []
    for m, image in enumerate(layout.images):
        for n, level in enumerate(image.levels):
            name = str(n) if len(layout.images) == 1 else f"i{m}.{n}"
            entries.append(LayoutEntry(name, level, image.full, m, n))
    return entries


def _associated_entries(layout: Layout) -> list[LayoutEntry]:
    entries = []
    for group_name, group in (
        ("label", layout.label),
        ("macro", layout.macro),
        ("other", layout.other),
    ):
        for k, ifd in enumerate(group):
            name = group_name if len(group) == 1 else f"{group_name}{k}"
            entries.append(LayoutEntry(name, ifd, None))
    return entries


def layout_entries(layout: Layout) -> list[LayoutEntry]:
    """Pyramid levels first (named ``N`` or ``iM.N``), then associated images."""
    return _level_entries(layout) + _associated_entries(layout)


def summarize_entry(entry: LayoutEntry) -> LevelSummary:
    return summarize(entry.name, entry.ifd, entry.full)


def summarize_layout(layout: Layout) -> list[LevelSummary]:
    return [summarize_entry(entry) for entry in layout_entries(layout)]


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def row_cells(row: LevelSummary) -> list[str]:
    kind = "tile" if row.tiled else "strip"
    return [
        row.name,
        row.role,
        f"{row.width}x{row.height}",
        f"{kind} {row.chunk_width}x{row.chunk_height}",
        f"{row.chunks_across}x{row.chunks_down}",
        str(row.empty_chunks) if row.empty_chunks else "-",
        row.compression,
        row.photometric,
        f"{row.samples}x{row.bits}",
        f"{row.downsample:.3f}" if row.downsample is not None else "-",
        human_bytes(row.stored_bytes),
        f"@{row.offset}",
    ]


HEADER = [
    "level",
    "role",
    "size",
    "chunks",
    "grid",
    "empty",
    "compression",
    "photometric",
    "spp x bits",
    "downsample",
    "stored",
    "ifd",
]


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def format_line(layout: Layout) -> str:
    container = "BigTIFF" if layout.bigtiff else "classic TIFF"
    return f"format: {container}  vendor: {layout.vendor}"


def format_report(layout: Layout) -> str:
    rows = [HEADER] + [row_cells(r) for r in summarize_layout(layout)]
    return "\n".join([layout.path, format_line(layout), "", format_table(rows)])
