"""Pick the right redaction strategy for a chunk."""

from __future__ import annotations

import logging

from tiffredact.codecs import jpeg_dct, pixel
from tiffredact.codecs.fill import Rgb
from tiffredact.model import Chunk, ImageIFD, Region

logger = logging.getLogger(__name__)

DCT_PHOTOMETRICS = {0, 1, 2, 6}


def can_use_dct(ifd: ImageIFD) -> bool:
    return (
        ifd.compression == pixel.COMPRESSION_JPEG
        and ifd.bits_per_sample == 8
        and ifd.photometric in DCT_PHOTOMETRICS
        and ifd.planar_config == 1
    )


def redact_chunk(
    ifd: ImageIFD,
    data: bytes,
    chunk: Chunk,
    rects: list[Region],
    fill: Rgb,
    use_dct: bool = True,
) -> bytes:
    """Return replacement bytes for ``chunk`` with ``rects`` filled."""
    if use_dct and can_use_dct(ifd):
        return jpeg_dct.redact_jpeg(data, ifd.jpegtables, rects, fill, ifd.photometric)
    return pixel.redact_pixels(ifd, data, chunk, rects, fill)
