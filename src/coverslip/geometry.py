"""Mapping regions between pyramid levels and onto tiles or strips."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

from coverslip.model import (
    Chunk,
    ImageIFD,
    Layout,
    Region,
    RegionInput,
    RegionSpec,
)

logger = logging.getLogger(__name__)


class RegionFrameError(ValueError):
    """A region cannot be placed on the pyramid level it names."""


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def region_to_full(region: Region, level: ImageIFD, full: ImageIFD) -> Region:
    """Map a region drawn on ``level`` onto the full image, rounding outward.

    Integer arithmetic keeps the edges exact, so scaling the result back down
    with :func:`scale_region` never uncovers a pixel the caller named.
    """
    x0 = region.x * full.width // level.width
    y0 = region.y * full.height // level.height
    x1 = _ceil_div(region.x1 * full.width, level.width)
    y1 = _ceil_div(region.y1 * full.height, level.height)
    return Region(x0, y0, x1 - x0, y1 - y0)


def find_frame_level(
    layout: Layout, frame: tuple[int, int]
) -> tuple[ImageIFD, ImageIFD]:
    """The ``(full, level)`` pair whose level has exactly ``frame`` dimensions.

    The match is strict on purpose: mistyped dimensions fail here instead of
    redacting the wrong place.
    """
    for image in layout.images:
        for level in image.levels:
            if (level.width, level.height) == frame:
                return image.full, level
    sizes = {(lv.width, lv.height) for image in layout.images for lv in image.levels}
    available = ", ".join(f"{w}x{h}" for w, h in sorted(sizes, reverse=True))
    raise RegionFrameError(
        f"No pyramid level is {frame[0]}x{frame[1]}; this file has {available}"
    )


def resolve_region(layout: Layout, spec: RegionInput) -> Region:
    """Express ``spec`` in full-resolution coordinates."""
    if isinstance(spec, Region):
        return spec
    if spec.frame is None:
        return spec.region
    full, level = find_frame_level(layout, spec.frame)
    if spec.region.clip(level.width, level.height) is None:
        raise RegionFrameError(
            f"Region {spec.region} lies outside the {level.width}x{level.height} level"
        )
    resolved = region_to_full(spec.region, level, full)
    logger.info(
        "%s on the %dx%d level is %s at full resolution",
        spec.region,
        level.width,
        level.height,
        resolved,
    )
    return resolved


def resolve_regions(layout: Layout, specs: Iterable[RegionInput]) -> list[Region]:
    return [resolve_region(layout, spec) for spec in specs]


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
    """Geometry of chunk ``index``.

    Chunks are row-major within a plane; with ``PlanarConfiguration=2`` all
    chunks of plane 0 come first, then plane 1, and so on.
    """
    plane, local = divmod(index, ifd.chunks_per_plane)
    row, col = divmod(local, ifd.chunks_across)
    x = col * ifd.chunk_width
    y = row * ifd.chunk_height
    if ifd.tiled:
        return Chunk(index, x, y, ifd.chunk_width, ifd.chunk_height, plane)
    height = min(ifd.chunk_height, ifd.height - y)
    return Chunk(index, x, y, ifd.width, height, plane)


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
    for plane in range(ifd.planes):
        base = plane * ifd.chunks_per_plane
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                chunk = chunk_at(ifd, base + row * ifd.chunks_across + col)
                overlap = clipped.intersection(chunk.region)
                if overlap is None:
                    continue
                found.append((chunk, overlap.translate(-chunk.x, -chunk.y)))
    return found


def covers_chunk(rects: list[Region], chunk: Chunk) -> bool:
    """True if a single rect covers the whole chunk (used for the solid-fill fast path)."""
    whole = Region(0, 0, chunk.width, chunk.height)
    return any(r.intersection(whole) == whole for r in rects)
