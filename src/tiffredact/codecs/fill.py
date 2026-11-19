"""Translate an RGB fill colour into the sample values a codestream stores."""

from __future__ import annotations

PHOTOMETRIC_WHITE_IS_ZERO = 0
PHOTOMETRIC_BLACK_IS_ZERO = 1
PHOTOMETRIC_RGB = 2
PHOTOMETRIC_YCBCR = 6

Rgb = tuple[int, int, int]


class UnsupportedFill(Exception):
    """The fill colour cannot be expressed for this photometric interpretation."""


def _clamp(value: float) -> int:
    return max(0, min(255, round(value)))


def luma(rgb: Rgb) -> int:
    r, g, b = rgb
    return _clamp(0.299 * r + 0.587 * g + 0.114 * b)


def rgb_to_ycbcr(rgb: Rgb) -> tuple[int, int, int]:
    """JFIF full-range YCbCr, which is what TIFF JPEG tiles use in practice."""
    r, g, b = rgb
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return _clamp(y), _clamp(cb), _clamp(cr)


def chunk_fill_values(
    rgb: Rgb, photometric: int, samples_per_pixel: int, planar_config: int, plane: int
) -> list[int]:
    """Sample values a single chunk stores for ``rgb``.

    With ``PlanarConfiguration=2`` each chunk holds one plane, so only that
    plane's value is returned.
    """
    values = fill_samples(rgb, photometric, samples_per_pixel)
    if planar_config == 2:
        return [values[plane]]
    return values


def fill_samples(rgb: Rgb, photometric: int, samples_per_pixel: int) -> list[int]:
    """Component values (8-bit) that paint ``rgb`` for a given photometric."""
    if samples_per_pixel == 1:
        value = luma(rgb)
        if photometric == PHOTOMETRIC_WHITE_IS_ZERO:
            value = 255 - value
        return [value]
    if photometric == PHOTOMETRIC_RGB:
        base = list(rgb)
    elif photometric == PHOTOMETRIC_YCBCR:
        base = list(rgb_to_ycbcr(rgb))
    else:
        raise UnsupportedFill(
            f"No fill mapping for photometric={photometric} with "
            f"{samples_per_pixel} samples per pixel"
        )
    # Extra samples (alpha etc.) are set opaque.
    return base + [255] * (samples_per_pixel - 3)
