"""Mapping regions between pyramid levels and onto tiles or strips."""

from __future__ import annotations

import math

from tiffredact.model import Chunk, ImageIFD, Region


def scale_region(region: Region, full: ImageIFD, level: ImageIFD) -> Region | None:
    """Map a full-resolution region onto ``level``, rounding outward.

    Downsample factors are computed per axis because vendor pyramids are not
    exact powers of two (and SVS thumbnails are not even uniformly scaled).
    """
    sx = full.width / level.width
    sy = full.height / level.height
    x0 = math.floor(region.x / sx)
    y0 = math.floor(region.y / sy)
    x1 = math.ceil(region.x1 / sx)
    y1 = math.ceil(region.y1 / sy)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return Region(x0, y0, x1 - x0, y1 - y0).clip(level.width, level.height)


def chunk_at(ifd: ImageIFD, index: int) -> Chunk:
    """Geometry of chunk ``index`` (row-major for tiles, top-down for strips)."""
    row, col = divmod(index, ifd.chunks_across)
    x = col * ifd.chunk_width
    y = row * ifd.chunk_height
    if ifd.tiled:
        return Chunk(index, x, y, ifd.chunk_width, ifd.chunk_height)
    height = min(ifd.chunk_height, ifd.height - y)
    return Chunk(index, x, y, ifd.width, height)


def intersecting_chunks(ifd: ImageIFD, region: Region) -> list[tuple[Chunk, Region]]:
    """Chunks touched by ``region`` with the overlap expressed in chunk-local pixels."""
    clipped = region.clip(ifd.width, ifd.height)
    if clipped is None:
        return []
    col0 = clipped.x // ifd.chunk_width
    col1 = (clipped.x1 - 1) // ifd.chunk_width
    row0 = clipped.y // ifd.chunk_height
    row1 = (clipped.y1 - 1) // ifd.chunk_height
    found: list[tuple[Chunk, Region]] = []
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            chunk = chunk_at(ifd, row * ifd.chunks_across + col)
            overlap = clipped.intersection(chunk.region)
            if overlap is None:
                continue
            found.append((chunk, overlap.translate(-chunk.x, -chunk.y)))
    return found


def covers_chunk(rects: list[Region], chunk: Chunk) -> bool:
    """True if a single rect covers the whole chunk (used for the solid-fill fast path)."""
    whole = Region(0, 0, chunk.width, chunk.height)
    return any(r.intersection(whole) == whole for r in rects)
