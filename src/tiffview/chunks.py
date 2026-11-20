"""Read and decode single tiles (or strips), and assemble whole levels from them."""

from __future__ import annotations

import logging
from typing import BinaryIO

import numpy as np

from tiffview.codecs.pixel import decode_chunk
from tiffview.model import Chunk, ImageIFD

logger = logging.getLogger(__name__)

DEFAULT_MAX_PIXELS = 8192 * 8192
"""Largest level :func:`read_level` assembles unless told otherwise."""


class EmptyChunkError(Exception):
    """The chunk has no stored data (zero offset or bytecount)."""


class LevelTooLargeError(Exception):
    """Assembling the level would exceed the pixel limit."""


class TileIndexError(ValueError):
    """The requested tile column/row lies outside the tile grid."""


def chunk_at(ifd: ImageIFD, index: int) -> Chunk:
    """Geometry of chunk ``index`` (copied from tiffredact's ``geometry.py``).

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


def tile_index(ifd: ImageIFD, col: int, row: int, plane: int = 0) -> int:
    """Index of the chunk at grid position ``(col, row)`` in ``plane``."""
    if not (0 <= col < ifd.chunks_across and 0 <= row < ifd.chunks_down):
        raise TileIndexError(
            f"Tile ({col},{row}) is outside the "
            f"{ifd.chunks_across}x{ifd.chunks_down} tile grid"
        )
    return plane * ifd.chunks_per_plane + row * ifd.chunks_across + col


def is_empty(ifd: ImageIFD, index: int) -> bool:
    return ifd.data_offsets[index] == 0 or ifd.data_bytecounts[index] == 0


def read_chunk_bytes(fh: BinaryIO, ifd: ImageIFD, index: int) -> bytes:
    """The stored (still compressed) bytes of chunk ``index``."""
    if is_empty(ifd, index):
        raise EmptyChunkError(f"Chunk {index} has no stored data")
    fh.seek(ifd.data_offsets[index])
    data = fh.read(ifd.data_bytecounts[index])
    if len(data) != ifd.data_bytecounts[index]:
        raise OSError(
            f"Chunk {index} is truncated: expected {ifd.data_bytecounts[index]} "
            f"bytes, got {len(data)}"
        )
    return data


def _to_native(array: np.ndarray) -> np.ndarray:
    if array.dtype.isnative:
        return array
    return array.astype(array.dtype.newbyteorder("="))


def _decode_tile(fh: BinaryIO, ifd: ImageIFD, col: int, row: int) -> np.ndarray:
    """Decode all planes of one tile; the nominal (uncropped) tile size."""
    planes = []
    for plane in range(ifd.planes):
        index = tile_index(ifd, col, row, plane)
        data = read_chunk_bytes(fh, ifd, index)
        planes.append(decode_chunk(ifd, data, chunk_at(ifd, index), to_rgb=True))
    array = planes[0] if len(planes) == 1 else np.concatenate(planes, axis=2)
    return _to_native(array)


def _crop_to_image(ifd: ImageIFD, chunk: Chunk, array: np.ndarray) -> np.ndarray:
    """Drop the padding of tiles at the right or bottom edge."""
    width = min(chunk.width, ifd.width - chunk.x)
    height = min(chunk.height, ifd.height - chunk.y)
    return array[:height, :width]


def read_tile(path: str, ifd: ImageIFD, col: int, row: int) -> np.ndarray:
    """Decode one tile, cropped to the image bounds; shape (height, width, samples)."""
    with open(path, "rb") as fh:
        array = _decode_tile(fh, ifd, col, row)
    chunk = chunk_at(ifd, tile_index(ifd, col, row))
    return _crop_to_image(ifd, chunk, array)


def check_level_size(ifd: ImageIFD, max_pixels: int | None) -> None:
    pixels = ifd.width * ifd.height
    if max_pixels is not None and pixels > max_pixels:
        raise LevelTooLargeError(
            f"Level {ifd.width}x{ifd.height} has {pixels:,} pixels, more than the "
            f"limit of {max_pixels:,}; pick a smaller level or force it"
        )


def read_level(
    path: str, ifd: ImageIFD, max_pixels: int | None = DEFAULT_MAX_PIXELS
) -> np.ndarray:
    """Assemble a whole level from its tiles; shape (height, width, samples).

    Empty tiles are left as zeros.
    """
    check_level_size(ifd, max_pixels)
    level: np.ndarray | None = None
    empty = 0
    with open(path, "rb") as fh:
        for row in range(ifd.chunks_down):
            for col in range(ifd.chunks_across):
                try:
                    array = _decode_tile(fh, ifd, col, row)
                except EmptyChunkError:
                    empty += 1
                    continue
                chunk = chunk_at(ifd, tile_index(ifd, col, row))
                if level is None:
                    shape = (ifd.height, ifd.width, array.shape[2])
                    level = np.zeros(shape, dtype=array.dtype)
                cropped = _crop_to_image(ifd, chunk, array)
                level[
                    chunk.y : chunk.y + cropped.shape[0],
                    chunk.x : chunk.x + cropped.shape[1],
                ] = cropped
    if empty:
        logger.info("%d empty tiles left blank", empty)
    if level is None:
        return np.zeros((ifd.height, ifd.width, ifd.samples_per_pixel), np.uint8)
    return level
