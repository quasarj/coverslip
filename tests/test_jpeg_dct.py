import imagecodecs
import numpy as np
import pytest

from tiffredact.codecs.jpeg_dct import (
    JpegDctError,
    join_tables,
    quantized_dc,
    redact_jpeg,
    strip_app0,
)
from tiffredact.model import Region

from conftest import make_image

PHOTOMETRIC_RGB = 2
PHOTOMETRIC_YCBCR = 6
PHOTOMETRIC_MINISBLACK = 1


def encode(image: np.ndarray, subsampling: str | None, colorspace: str | None = None) -> bytes:
    kwargs = {"level": 85}
    if subsampling:
        kwargs["subsampling"] = subsampling
    if colorspace:
        kwargs["colorspace"] = colorspace
        kwargs["outcolorspace"] = colorspace
    return imagecodecs.jpeg8_encode(np.ascontiguousarray(image), **kwargs)


def decode(data: bytes, colorspace: str | None = None) -> np.ndarray:
    if colorspace:
        return imagecodecs.jpeg8_decode(data, colorspace=colorspace, outcolorspace=colorspace)
    return imagecodecs.jpeg8_decode(data)


@pytest.mark.parametrize("subsampling,mcu_x,mcu_y", [("444", 8, 8), ("422", 16, 8), ("420", 16, 16)])
def test_untouched_blocks_are_bit_exact_and_region_is_black(
    subsampling: str, mcu_x: int, mcu_y: int
) -> None:
    image = make_image(96, 128)
    original = encode(image, subsampling)
    before = decode(original)
    rect = Region(40, 24, 30, 20)
    out = redact_jpeg(original, None, [rect], (0, 0, 0), PHOTOMETRIC_YCBCR)
    after = decode(out)
    assert after.shape == before.shape
    # Region (snapped outward to MCU bounds) is pure black. With subsampled
    # chroma the decoder's upsampling filter blends the outermost 1-2 pixels of
    # the painted area with neighbouring chroma, so those are excluded.
    x0, y0 = rect.x // mcu_x * mcu_x, rect.y // mcu_y * mcu_y
    x1 = -(-rect.x1 // mcu_x) * mcu_x
    y1 = -(-rect.y1 // mcu_y) * mcu_y
    bleed = 0 if subsampling == "444" else 2
    assert after[y0 + bleed : y1 - bleed, x0 + bleed : x1 - bleed].max() == 0
    # Everything outside the snapped rectangle is identical to the original decode.
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = False
    assert np.array_equal(before[mask], after[mask])


def test_rgb_codestream_fill_uses_rgb_values() -> None:
    image = make_image(64, 64)
    original = encode(image, "444", colorspace="RGB")
    out = redact_jpeg(original, None, [Region(0, 0, 16, 16)], (255, 0, 0), PHOTOMETRIC_RGB)
    after = decode(out, colorspace="RGB")
    block = after[0:16, 0:16].astype(int)
    assert block[..., 0].min() >= 253 and block[..., 1].max() <= 2 and block[..., 2].max() <= 2


def test_grayscale() -> None:
    image = make_image(64, 64)[:, :, 0]
    original = encode(image, None)
    out = redact_jpeg(original, None, [Region(8, 8, 8, 8)], (0, 0, 0), PHOTOMETRIC_MINISBLACK)
    after = decode(out)
    before = decode(original)
    assert after[8:16, 8:16].max() == 0
    assert np.array_equal(after[16:], before[16:])


def test_white_fill_saturates() -> None:
    image = make_image(64, 64)
    original = encode(image, "420")
    out = redact_jpeg(original, None, [Region(0, 0, 32, 32)], (255, 255, 255), PHOTOMETRIC_YCBCR)
    assert decode(out)[0:30, 0:30].min() == 255


def test_shared_tables_round_trip() -> None:
    """Abbreviated tile + JPEGTables, as TIFF stores them."""
    image = make_image(64, 64)
    full = encode(image, "444")
    tables, tile = split_tables(full)
    out = redact_jpeg(tile, tables, [Region(0, 0, 8, 8)], (0, 0, 0), PHOTOMETRIC_YCBCR)
    # The output is standalone *and* still decodes when readers prepend the tables.
    assert decode(out)[0:8, 0:8].max() == 0
    assert np.array_equal(decode(out), decode(tables[:-2] + out[2:]))
    assert b"JFIF" not in out


def split_tables(jpeg: bytes) -> tuple[bytes, bytes]:
    """Split a standalone JPEG into (tables-only stream, abbreviated image stream)."""
    import struct

    pos = 2
    table_segments = b""
    rest = b""
    while pos < len(jpeg):
        marker = jpeg[pos + 1]
        length = struct.unpack(">H", jpeg[pos + 2 : pos + 4])[0]
        segment = jpeg[pos : pos + 2 + length]
        if marker in (0xDB, 0xC4):
            table_segments += segment
        elif marker == 0xDA:
            rest += jpeg[pos:]
            break
        elif marker not in (0xE0,):
            rest += segment
        pos += 2 + length
    return b"\xff\xd8" + table_segments + b"\xff\xd9", b"\xff\xd8" + rest


def test_join_tables_validation() -> None:
    with pytest.raises(JpegDctError):
        join_tables(b"\xff\xd8\xff\xd9", b"not a jpeg")
    with pytest.raises(JpegDctError):
        join_tables(b"garbage", b"\xff\xd8\xff\xd9")
    assert join_tables(None, b"\xff\xd8\xff\xd9") == b"\xff\xd8\xff\xd9"


def test_strip_app0_only_removes_jfif() -> None:
    jfif = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    assert strip_app0(jfif) == b"\xff\xd8\xff\xd9"
    other = b"\xff\xd8\xff\xee\x00\x04ab\xff\xd9"
    assert strip_app0(other) == other


@pytest.mark.parametrize("quant", [1, 2, 3, 7, 10, 16, 255])
def test_quantized_dc_saturates_black_and_white(quant: int) -> None:
    assert quantized_dc(0, quant) * quant <= -1024
    assert quantized_dc(255, quant) * quant >= 1016
    assert quantized_dc(128, quant) == 0
