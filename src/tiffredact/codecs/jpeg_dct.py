"""Lossless DCT-domain redaction of a JPEG tile or strip.

Only the 8x8 blocks that intersect the redaction rectangles are changed; every
other block keeps its exact quantized coefficients, so the rest of the chunk
decodes bit-for-bit identically to the original.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile

import jpeglib
import numpy as np

from tiffredact.codecs.fill import Rgb, fill_samples
from tiffredact.model import Region

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
APP0 = 0xE0


class JpegDctError(Exception):
    """The chunk could not be edited in the DCT domain."""


def join_tables(tables: bytes | None, chunk: bytes) -> bytes:
    """Prepend a TIFF ``JPEGTables`` abbreviated stream to an abbreviated tile."""
    if not chunk.startswith(SOI):
        raise JpegDctError("Chunk does not start with a JPEG SOI marker")
    if not tables:
        return chunk
    if not tables.startswith(SOI) or not tables.endswith(EOI):
        raise JpegDctError("JPEGTables is not a valid abbreviated JPEG stream")
    return tables[:-2] + chunk[2:]


def strip_app0(data: bytes) -> bytes:
    """Remove a JFIF APP0 segment that libjpeg adds; vendor tiles never carry one.

    A JFIF marker would make decoders assume YCbCr components, while the TIFF
    Photometric tag is what readers of these files actually trust.
    """
    if not data.startswith(SOI):
        return data
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if marker == APP0 and data[pos + 4 : pos + 9] == b"JFIF\x00":
            return data[:pos] + data[pos + 2 + length :]
        if marker == 0xDA:  # start of scan: nothing more to look at
            break
        pos += 2 + length
    return data


def quantized_dc(value: int, quant: int) -> int:
    """Quantized DC coefficient that reconstructs a flat block of ``value``.

    Rounds so that 0 and 255 saturate after the IDCT instead of landing at 1 or
    254 when the quantizer does not divide 1024 evenly.
    """
    target = (value - 128) * 8
    if value <= 0:
        return -math.ceil(1024 / quant)
    if value >= 255:
        return math.ceil(1016 / quant)
    return int(round(target / quant))


def _block_span(start: int, stop: int, comp_factor: int, max_factor: int, limit: int) -> tuple[int, int]:
    """Block index range [b0, b1) of one component covering pixels [start, stop)."""
    pixels_per_block = 8 * max_factor / comp_factor
    b0 = math.floor(start / pixels_per_block)
    b1 = math.ceil(stop / pixels_per_block)
    return max(0, b0), min(limit, b1)


def _paint_component(
    coefs: np.ndarray,
    rects: list[Region],
    v_factor: int,
    h_factor: int,
    v_max: int,
    h_max: int,
    dc: int,
) -> np.ndarray:
    out = coefs.copy()
    rows, cols = out.shape[:2]
    for rect in rects:
        c0, c1 = _block_span(rect.x, rect.x1, h_factor, h_max, cols)
        r0, r1 = _block_span(rect.y, rect.y1, v_factor, v_max, rows)
        if c1 <= c0 or r1 <= r0:
            continue
        out[r0:r1, c0:c1] = 0
        out[r0:r1, c0:c1, 0, 0] = dc
    return out


def redact_jpeg(
    chunk: bytes,
    tables: bytes | None,
    rects: list[Region],
    fill: Rgb,
    photometric: int,
) -> bytes:
    """Return a standalone JPEG for ``chunk`` with ``rects`` painted ``fill``.

    The output embeds its own quantization and Huffman tables. Readers that
    prepend the shared ``JPEGTables`` still decode it, because the tile's own
    tables simply redefine the shared ones.
    """
    joined = join_tables(tables, chunk)
    with tempfile.TemporaryDirectory(prefix="tiffredact-") as tmp:
        src = os.path.join(tmp, "in.jpg")
        dst = os.path.join(tmp, "out.jpg")
        with open(src, "wb") as fh:
            fh.write(joined)
        jpeg = jpeglib.read_dct(src)
        _paint_jpeg(jpeg, rects, fill, photometric)
        jpeg.write_dct(dst)
        with open(dst, "rb") as fh:
            result = fh.read()
    return strip_app0(result)


def _paint_jpeg(jpeg: jpeglib.DCTJPEG, rects: list[Region], fill: Rgb, photometric: int) -> None:
    ncomp = int(jpeg.num_components)
    if ncomp not in (1, 3):
        raise JpegDctError(f"Unsupported component count {ncomp} for DCT redaction")
    values = fill_samples(fill, photometric, ncomp)
    factors = np.asarray(jpeg.samp_factor).reshape(ncomp, 2)  # rows are (v, h)
    v_max = int(factors[:, 0].max())
    h_max = int(factors[:, 1].max())
    quant_tables = np.asarray(jpeg.qt)
    table_numbers = np.asarray(jpeg.quant_tbl_no).reshape(-1)
    components = ["Y", "Cb", "Cr"][:ncomp]
    for index, name in enumerate(components):
        coefs = getattr(jpeg, name)
        if coefs is None:
            raise JpegDctError(f"Component {name} has no coefficients")
        quant = int(quant_tables[int(table_numbers[index])][0, 0])
        dc = quantized_dc(values[index], quant)
        v_factor, h_factor = (int(factors[index, 0]), int(factors[index, 1]))
        painted = _paint_component(coefs, rects, v_factor, h_factor, v_max, h_max, dc)
        setattr(jpeg, name, painted)
