import imagecodecs
import numpy as np
import pytest

from coverslip.codecs import pixel
from coverslip.codecs.fill import fill_samples
from coverslip.model import Chunk, ImageIFD, Region, Role

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
def test_lossless_round_trip_outside_rect(compression: int, predictor: int) -> None:
    ifd = make_ifd(compression, predictor)
    image = make_image(48, 64)
    chunk = Chunk(0, 0, 0, 64, 48)
    encoded = pixel.encode_chunk(ifd, image)
    assert np.array_equal(pixel.decode_chunk(ifd, encoded, chunk), image)
    out = pixel.redact_pixels(
        ifd, encoded, chunk, [Region(10, 10, 20, 20)], fill_samples((0, 0, 0), 2, 3)
    )
    decoded = pixel.decode_chunk(ifd, out, chunk)
    assert decoded[10:30, 10:30].max() == 0
    mask = np.ones((48, 64), dtype=bool)
    mask[10:30, 10:30] = False
    assert np.array_equal(decoded[mask], image[mask])


def test_sixteen_bit_fill_is_scaled() -> None:
    ifd = make_ifd(pixel.COMPRESSION_LZW, 2, spp=1, bits=16)
    image = make_image(48, 64)[:, :, :1].astype(np.uint16) * 257
    chunk = Chunk(0, 0, 0, 64, 48)
    encoded = pixel.encode_chunk(ifd, image)
    out = pixel.redact_pixels(
        ifd, encoded, chunk, [Region(0, 0, 64, 48)], fill_samples((255, 255, 255), 1, 1)
    )
    assert pixel.decode_chunk(ifd, out, chunk).min() == 65535


def test_jpeg_pixel_path_paints_region() -> None:
    ifd = make_ifd(pixel.COMPRESSION_JPEG)
    image = make_image(48, 64)
    chunk = Chunk(0, 0, 0, 64, 48)
    encoded = pixel.encode_chunk(ifd, image)
    out = pixel.redact_pixels(
        ifd, encoded, chunk, [Region(16, 16, 16, 16)], fill_samples((0, 0, 0), 2, 3)
    )
    assert pixel.decode_chunk(ifd, out, chunk)[16:32, 16:32].max() <= 2


def test_unknown_compression_raises() -> None:
    ifd = make_ifd(99)
    with pytest.raises(pixel.UnsupportedCodec):
        pixel.decode_chunk(ifd, b"", Chunk(0, 0, 0, 64, 48))


def test_wrong_sample_count_raises() -> None:
    ifd = make_ifd(pixel.COMPRESSION_NONE)
    with pytest.raises(pixel.UnsupportedCodec):
        pixel.decode_chunk(ifd, b"\0" * 10, Chunk(0, 0, 0, 64, 48))


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
