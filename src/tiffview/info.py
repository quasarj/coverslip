"""Text report: pyramid levels, associated images and, optionally, every tag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tifftools import constants

from tiffview.chunks import is_empty
from tiffview.layout import iter_all_ifds
from tiffview.model import ImageIFD, Layout

MAX_TAG_VALUES = 8
MAX_TAG_TEXT = 72


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
    )


def summarize_layout(layout: Layout) -> list[LevelSummary]:
    """Pyramid levels first (named ``N`` or ``iM.N``), then associated images."""
    rows: list[LevelSummary] = []
    for m, image in enumerate(layout.images):
        for n, level in enumerate(image.levels):
            name = str(n) if len(layout.images) == 1 else f"i{m}.{n}"
            rows.append(summarize(name, level, image.full))
    for group_name, group in (
        ("label", layout.label),
        ("macro", layout.macro),
        ("other", layout.other),
    ):
        for k, ifd in enumerate(group):
            name = group_name if len(group) == 1 else f"{group_name}{k}"
            rows.append(summarize(name, ifd, None))
    return rows


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _row_cells(row: LevelSummary) -> list[str]:
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
]


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def format_report(layout: Layout) -> str:
    container = "BigTIFF" if layout.bigtiff else "classic TIFF"
    rows = [HEADER] + [_row_cells(r) for r in summarize_layout(layout)]
    return "\n".join(
        [
            layout.path,
            f"format: {container}  vendor: {layout.vendor}",
            "",
            format_table(rows),
        ]
    )


def format_tag_value(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("latin-1")
    if isinstance(value, str):
        text = value.replace("\r", "\\r").replace("\n", "\\n")
        if len(text) > MAX_TAG_TEXT:
            return f"{text[:MAX_TAG_TEXT]}... ({len(value)} chars)"
        return text
    if isinstance(value, (list, tuple)):
        shown = ", ".join(str(v) for v in value[:MAX_TAG_VALUES])
        if len(value) > MAX_TAG_VALUES:
            return f"{shown}, ... ({len(value)} values)"
        return shown
    return str(value)


def tag_name(tag: int) -> str:
    known = constants.Tag.get(tag)
    return known.name if known is not None else "unknown"


def format_ifd_tags(ifd: dict[str, Any]) -> list[str]:
    lines = []
    for tag, entry in sorted(ifd["tags"].items()):
        value = format_tag_value(entry.get("data", "<unreadable>"))
        lines.append(f"  {tag:>5d} {tag_name(tag):<28s} {value}")
    return lines


def format_tags(info: dict[str, Any], layout: Layout) -> str:
    """Every tag of every IFD, headed by the IFD's role in the layout."""
    roles = {ifd.offset: ifd.role.value for ifd in layout.all_ifds()}
    lines: list[str] = []
    for ifd in iter_all_ifds(info):
        offset = int(ifd["offset"])
        lines.append(f"ifd@{offset} ({roles.get(offset, 'unclassified')}):")
        lines.extend(format_ifd_tags(ifd))
    return "\n".join(lines)
