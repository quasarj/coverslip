"""Pick the right redaction strategy for a chunk."""

from __future__ import annotations

import logging

from coverslip.codecs import jpeg_dct, pixel
from coverslip.codecs.fill import Rgb, chunk_fill_values
from coverslip.model import Chunk, ImageIFD, Region

logger = logging.getLogger(__name__)

DCT_PHOTOMETRICS = {0, 1, 2, 6}


def can_use_dct(ifd: ImageIFD) -> bool:
    return (
        ifd.compression == pixel.COMPRESSION_JPEG
        and ifd.bits_per_sample == 8
        and ifd.photometric in DCT_PHOTOMETRICS
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
    values = chunk_fill_values(
        fill, ifd.photometric, ifd.samples_per_pixel, ifd.planar_config, chunk.plane
    )
    if use_dct and can_use_dct(ifd):
        return jpeg_dct.redact_jpeg(data, ifd.jpegtables, rects, values)
    return pixel.redact_pixels(ifd, data, chunk, rects, values)
