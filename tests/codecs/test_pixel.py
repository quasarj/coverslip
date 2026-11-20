import imagecodecs
import numpy as np
import pytest

from tiffview.codecs import pixel
from tiffview.model import Chunk, ImageIFD, Role

from conftest import make_image


def make_ifd(
    compression: int, predictor: int = 1, spp: int = 3, bits: int = 8
) -> ImageIFD:
    return ImageIFD(
        offset=0,
        width=64,
        height=48,
        tiled=True,
        chunk_width=64,
        chunk_height=48,
        compression=compression,
        photometric=2 if spp == 3 else 1,
        bits_per_sample=bits,
        samples_per_pixel=spp,
        planar_config=1,
        predictor=predictor,
        sample_format=1,
        subsampling=None,
        jpegtables=None,
        new_subfile_type=0,
        description="",
        data_offsets=[0],
        data_bytecounts=[0],
        big_endian=False,
        role=Role.FULL,
    )


def encode_raw(compression: int, raw: bytes) -> bytes:
    encoders = {
        pixel.COMPRESSION_NONE: lambda data: data,
        pixel.COMPRESSION_LZW: imagecodecs.lzw_encode,
        pixel.COMPRESSION_ADOBE_DEFLATE: imagecodecs.zlib_encode,
        pixel.COMPRESSION_DEFLATE: imagecodecs.zlib_encode,
        pixel.COMPRESSION_PACKBITS: imagecodecs.packbits_encode,
        pixel.COMPRESSION_ZSTD: imagecodecs.zstd_encode,
    }
    return encoders[compression](raw)


@pytest.mark.parametrize(
    "compression,predictor",
    [
        (pixel.COMPRESSION_NONE, 1),
        (pixel.COMPRESSION_LZW, 1),
        (pixel.COMPRESSION_LZW, 2),
        (pixel.COMPRESSION_ADOBE_DEFLATE, 2),
        (pixel.COMPRESSION_DEFLATE, 1),
        (pixel.COMPRESSION_PACKBITS, 1),
        (pixel.COMPRESSION_ZSTD, 2),
    ],
)
def test_lossless_decode(compression: int, predictor: int) -> None:
    ifd = make_ifd(compression, predictor)
    image = make_image(48, 64)
    predicted = imagecodecs.delta_encode(image, axis=1) if predictor == 2 else image
    encoded = encode_raw(compression, np.ascontiguousarray(predicted).tobytes())
    decoded = pixel.decode_chunk(ifd, encoded, Chunk(0, 0, 0, 64, 48))
    assert np.array_equal(decoded, image)


def test_jpeg_decode_is_close() -> None:
    ifd = make_ifd(pixel.COMPRESSION_JPEG)
    image = make_image(48, 64)
    encoded = imagecodecs.jpeg8_encode(
        image, level=95, colorspace="RGB", outcolorspace="RGB"
    )
    decoded = pixel.decode_chunk(ifd, encoded, Chunk(0, 0, 0, 64, 48))
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(int) - image.astype(int)).mean() < 4


def test_wrong_sample_count_raises() -> None:
    ifd = make_ifd(pixel.COMPRESSION_NONE)
    with pytest.raises(pixel.UnsupportedCodec):
        pixel.decode_chunk(ifd, b"\0" * 10, Chunk(0, 0, 0, 64, 48))


def test_unknown_compression_raises() -> None:
    ifd = make_ifd(99)
    with pytest.raises(pixel.UnsupportedCodec):
        pixel.decode_chunk(ifd, b"", Chunk(0, 0, 0, 64, 48))


def test_ycbcr_jpeg_to_rgb() -> None:
    ifd = make_ifd(pixel.COMPRESSION_JPEG)
    ifd.photometric = 6
    image = make_image(48, 64)
    encoded = imagecodecs.jpeg8_encode(
        image, level=95, colorspace="RGB", outcolorspace="YCBCR"
    )
    chunk = Chunk(0, 0, 0, 64, 48)
    rgb = pixel.decode_chunk(ifd, encoded, chunk, to_rgb=True)
    assert np.abs(rgb.astype(int) - image.astype(int)).mean() < 8
    stored = pixel.decode_chunk(ifd, encoded, chunk)
    assert np.abs(stored.astype(int) - image.astype(int)).mean() > 20
