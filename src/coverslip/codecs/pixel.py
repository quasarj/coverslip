"""Decode -> paint -> re-encode fallback for any chunk codec we understand.

Lossless codecs (none, LZW, Deflate, PackBits) round-trip exactly outside the
painted rectangles. JPEG 2000 and JPEG through this path take one generation
loss for the affected chunk only.
"""

from __future__ import annotations

import imagecodecs
import numpy as np

from coverslip.codecs.fill import Rgb, chunk_fill_values
from coverslip.model import Chunk, ImageIFD, Region

COMPRESSION_NONE = 1
COMPRESSION_LZW = 5
COMPRESSION_JPEG = 7
COMPRESSION_ADOBE_DEFLATE = 8
COMPRESSION_DEFLATE = 32946
COMPRESSION_PACKBITS = 32773
COMPRESSION_APERIO_JP2K_YCBCR = 33003
COMPRESSION_APERIO_JP2K_RGB = 33005
COMPRESSION_JP2K = 34712
COMPRESSION_ZSTD = 50000
COMPRESSION_WEBP = 50001

PREDICTOR_NONE = 1
PREDICTOR_HORIZONTAL = 2
PREDICTOR_FLOAT = 3

SAMPLE_FORMAT_UINT = 1
SAMPLE_FORMAT_INT = 2
SAMPLE_FORMAT_FLOAT = 3

JPEG2000_COMPRESSIONS = {
    COMPRESSION_APERIO_JP2K_YCBCR,
    COMPRESSION_APERIO_JP2K_RGB,
    COMPRESSION_JP2K,
}

PHOTOMETRIC_NAMES = {0: "MINISWHITE", 1: "MINISBLACK", 2: "RGB", 6: "YCBCR"}


class UnsupportedCodec(Exception):
    """We do not know how to decode or encode this chunk."""


def sample_dtype(ifd: ImageIFD) -> np.dtype:
    bits = ifd.bits_per_sample
    if bits not in (8, 16, 32, 64):
        raise UnsupportedCodec(f"{bits}-bit samples are not supported")
    if ifd.sample_format == SAMPLE_FORMAT_FLOAT:
        kind = "f"
    elif ifd.sample_format == SAMPLE_FORMAT_INT:
        kind = "i"
    else:
        kind = "u"
    order = ">" if ifd.big_endian else "<"
    return np.dtype(f"{order}{kind}{bits // 8}")


def _jpeg_colorspace(ifd: ImageIFD) -> str | None:
    """Tell libjpeg what the components are so it does no colour conversion."""
    if ifd.samples_per_chunk == 1:
        return "GRAYSCALE"
    return PHOTOMETRIC_NAMES.get(ifd.photometric)


def _jpeg_subsampling(ifd: ImageIFD) -> str | None:
    if ifd.photometric != 6 or ifd.subsampling is None:
        return None
    return {(1, 1): "444", (2, 1): "422", (2, 2): "420", (4, 1): "411"}.get(
        ifd.subsampling
    )


def decode_chunk(ifd: ImageIFD, data: bytes, chunk: Chunk) -> np.ndarray:
    """Decode one tile/strip to an array of shape (height, width, samples)."""
    comp = ifd.compression
    shape = (chunk.height, chunk.width, ifd.samples_per_chunk)
    if comp == COMPRESSION_JPEG:
        array = imagecodecs.jpeg8_decode(
            data,
            tables=ifd.jpegtables,
            colorspace=_jpeg_colorspace(ifd),
            outcolorspace=_jpeg_colorspace(ifd),
        )
        return _as_chunk_shape(array, shape)
    if comp in JPEG2000_COMPRESSIONS:
        array = imagecodecs.jpeg2k_decode(data)
        return _as_chunk_shape(array, shape)
    if comp == COMPRESSION_WEBP:
        return _as_chunk_shape(imagecodecs.webp_decode(data), shape)
    raw = _decode_raw(comp, data)
    array = np.frombuffer(raw, dtype=sample_dtype(ifd))
    expected = chunk.height * chunk.width * ifd.samples_per_chunk
    if array.size != expected:
        raise UnsupportedCodec(
            f"Decoded {array.size} samples for chunk {chunk.index} but expected {expected}"
        )
    array = array.reshape(shape)
    return _undo_predictor(ifd, array)


def _as_chunk_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.shape != shape:
        raise UnsupportedCodec(
            f"Decoded shape {array.shape} does not match chunk {shape}"
        )
    return np.ascontiguousarray(array)


def _decode_raw(comp: int, data: bytes) -> bytes:
    if comp == COMPRESSION_NONE:
        return data
    if comp == COMPRESSION_LZW:
        return imagecodecs.lzw_decode(data)
    if comp in (COMPRESSION_ADOBE_DEFLATE, COMPRESSION_DEFLATE):
        return imagecodecs.zlib_decode(data)
    if comp == COMPRESSION_PACKBITS:
        return imagecodecs.packbits_decode(data)
    if comp == COMPRESSION_ZSTD:
        return imagecodecs.zstd_decode(data)
    raise UnsupportedCodec(f"Compression {comp} is not supported")


def _encode_raw(comp: int, raw: bytes) -> bytes:
    if comp == COMPRESSION_NONE:
        return raw
    if comp == COMPRESSION_LZW:
        return imagecodecs.lzw_encode(raw)
    if comp in (COMPRESSION_ADOBE_DEFLATE, COMPRESSION_DEFLATE):
        return imagecodecs.zlib_encode(raw)
    if comp == COMPRESSION_PACKBITS:
        return imagecodecs.packbits_encode(raw)
    if comp == COMPRESSION_ZSTD:
        return imagecodecs.zstd_encode(raw)
    raise UnsupportedCodec(f"Compression {comp} is not supported")


def _undo_predictor(ifd: ImageIFD, array: np.ndarray) -> np.ndarray:
    if ifd.predictor == PREDICTOR_NONE:
        return array
    if ifd.predictor == PREDICTOR_HORIZONTAL:
        return imagecodecs.delta_decode(array, axis=1)
    if ifd.predictor == PREDICTOR_FLOAT:
        return imagecodecs.floatpred_decode(array, axis=1)
    raise UnsupportedCodec(f"Predictor {ifd.predictor} is not supported")


def _apply_predictor(ifd: ImageIFD, array: np.ndarray) -> np.ndarray:
    if ifd.predictor == PREDICTOR_NONE:
        return array
    if ifd.predictor == PREDICTOR_HORIZONTAL:
        return imagecodecs.delta_encode(array, axis=1)
    if ifd.predictor == PREDICTOR_FLOAT:
        return imagecodecs.floatpred_encode(array, axis=1)
    raise UnsupportedCodec(f"Predictor {ifd.predictor} is not supported")


def encode_chunk(ifd: ImageIFD, array: np.ndarray) -> bytes:
    """Encode an array back into the chunk's compression scheme."""
    comp = ifd.compression
    if comp == COMPRESSION_JPEG:
        image = array[:, :, 0] if array.shape[2] == 1 else array
        return imagecodecs.jpeg8_encode(
            np.ascontiguousarray(image),
            level=95,
            colorspace=_jpeg_colorspace(ifd),
            outcolorspace=_jpeg_colorspace(ifd),
            subsampling=_jpeg_subsampling(ifd),
        )
    if comp in JPEG2000_COMPRESSIONS:
        image = array[:, :, 0] if array.shape[2] == 1 else array
        mct = comp != COMPRESSION_APERIO_JP2K_YCBCR
        return imagecodecs.jpeg2k_encode(
            np.ascontiguousarray(image), level=0, codecformat="j2k", mct=mct
        )
    if comp == COMPRESSION_WEBP:
        return imagecodecs.webp_encode(np.ascontiguousarray(array), lossless=True)
    predicted = _apply_predictor(ifd, np.ascontiguousarray(array))
    return _encode_raw(comp, np.ascontiguousarray(predicted).tobytes())


def scale_fill(values: list[int], dtype: np.dtype) -> list[int | float]:
    """Scale 8-bit fill values to the sample type of the image."""
    if dtype.kind == "f":
        return [v / 255.0 for v in values]
    bits = dtype.itemsize * 8
    top = (1 << bits) - 1 if dtype.kind == "u" else (1 << (bits - 1)) - 1
    return [round(v / 255 * top) for v in values]


def paint(array: np.ndarray, rects: list[Region], values: list[int]) -> np.ndarray:
    """Paint ``rects`` (chunk-local) onto a decoded chunk in place and return it."""
    scaled = scale_fill(values, array.dtype)
    for rect in rects:
        clipped = rect.clip(array.shape[1], array.shape[0])
        if clipped is None:
            continue
        array[clipped.y : clipped.y1, clipped.x : clipped.x1, :] = scaled
    return array


def redact_pixels(
    ifd: ImageIFD, data: bytes, chunk: Chunk, rects: list[Region], values: list[int]
) -> bytes:
    array = decode_chunk(ifd, data, chunk).copy()
    paint(array, rects, values)
    return encode_chunk(ifd, array)
