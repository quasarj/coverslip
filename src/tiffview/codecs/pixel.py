"""Decode a single chunk (tile or strip) of any codec we understand.

Copied from tiffredact's ``codecs/pixel.py``; only the decode half is kept.
Added here: ``decode_chunk(..., to_rgb=True)``, because tiffredact wants
samples exactly as stored (it re-encodes them) but a viewer wants RGB.
"""

from __future__ import annotations

import imagecodecs
import numpy as np

from tiffview.model import Chunk, ImageIFD

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
    """We do not know how to decode this chunk."""


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


def _jpeg_outcolorspace(ifd: ImageIFD, to_rgb: bool) -> str | None:
    colorspace = _jpeg_colorspace(ifd)
    if to_rgb and colorspace == "YCBCR":
        return "RGB"
    return colorspace


def decode_chunk(
    ifd: ImageIFD, data: bytes, chunk: Chunk, *, to_rgb: bool = False
) -> np.ndarray:
    """Decode one tile/strip to an array of shape (height, width, samples).

    Samples come back as stored unless ``to_rgb`` is set, which converts
    YCbCr JPEG chunks to RGB.
    """
    comp = ifd.compression
    shape = (chunk.height, chunk.width, ifd.samples_per_chunk)
    if comp == COMPRESSION_JPEG:
        array = imagecodecs.jpeg8_decode(
            data,
            tables=ifd.jpegtables,
            colorspace=_jpeg_colorspace(ifd),
            outcolorspace=_jpeg_outcolorspace(ifd, to_rgb),
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


def _undo_predictor(ifd: ImageIFD, array: np.ndarray) -> np.ndarray:
    if ifd.predictor == PREDICTOR_NONE:
        return array
    if ifd.predictor == PREDICTOR_HORIZONTAL:
        return imagecodecs.delta_decode(array, axis=1)
    if ifd.predictor == PREDICTOR_FLOAT:
        return imagecodecs.floatpred_decode(array, axis=1)
    raise UnsupportedCodec(f"Predictor {ifd.predictor} is not supported")
