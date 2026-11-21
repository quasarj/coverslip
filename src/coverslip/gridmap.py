"""Tile byte-count maps and mapping between a viewed level and a tiled level.

The *view* level is the image on screen; the *tile* level is the one whose
tiles a click opens. Both are levels of the same pyramid, so their pixel grids
differ only by a per-axis scale factor.
"""

from __future__ import annotations

import numpy as np

from coverslip.model import ImageIFD


def byte_grid(ifd: ImageIFD) -> np.ndarray:
    """Stored bytes per tile, shape (chunks_down, chunks_across).

    With separate planes, a tile's planes are summed.
    """
    counts = np.asarray(ifd.data_bytecounts, dtype=np.int64)
    planes = counts.reshape(ifd.planes, ifd.chunks_down, ifd.chunks_across)
    return planes.sum(axis=0)


def scale_between(source: ImageIFD, target: ImageIFD) -> tuple[float, float]:
    """Per-axis factor taking ``source`` pixel coordinates to ``target`` ones."""
    return target.width / source.width, target.height / source.height


def tile_under(
    view: ImageIFD, tiles: ImageIFD, x: float, y: float
) -> tuple[int, int] | None:
    """The ``(col, row)`` of the ``tiles`` tile under view point ``(x, y)``.

    ``None`` when the point lies outside the image.
    """
    if not (0 <= x < view.width and 0 <= y < view.height):
        return None
    sx, sy = scale_between(view, tiles)
    col = int(x * sx) // tiles.chunk_width
    row = int(y * sy) // tiles.chunk_height
    return min(col, tiles.chunks_across - 1), min(row, tiles.chunks_down - 1)


def tile_box_on_view(
    view: ImageIFD, tiles: ImageIFD, col: int, row: int
) -> tuple[float, float, float, float]:
    """Bounds ``(x0, y0, x1, y1)`` of a tile in view coordinates, clipped."""
    sx, sy = scale_between(tiles, view)
    x0 = col * tiles.chunk_width
    y0 = row * tiles.chunk_height
    x1 = min(x0 + tiles.chunk_width, tiles.width)
    y1 = min(y0 + tiles.chunk_height, tiles.height)
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def grid_extent(view: ImageIFD, tiles: ImageIFD) -> tuple[float, float, float, float]:
    """Matplotlib ``extent`` (left, right, bottom, top) placing the byte grid
    of ``tiles`` over an image of ``view``.

    The grid covers whole nominal tiles, so it reaches past the right and
    bottom edges of the image when edge tiles are partial.
    """
    sx, sy = scale_between(tiles, view)
    right = tiles.chunks_across * tiles.chunk_width * sx
    bottom = tiles.chunks_down * tiles.chunk_height * sy
    return 0.0, right, bottom, 0.0
