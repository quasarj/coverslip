"""Synthetic pyramidal TIFF fixtures built with tifffile."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import tifffile

# tests never open windows
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent


def make_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Smooth-ish RGB test image so JPEG tiles are not trivially blank."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (x * 255 // max(width - 1, 1)),
            (y * 255 // max(height - 1, 1)),
            ((x + y) * 255 // max(width + height - 2, 1)),
        ],
        axis=-1,
    ).astype(np.uint8)
    noise = rng.integers(0, 24, base.shape, dtype=np.uint8)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    return np.ascontiguousarray(image[::factor, ::factor])


def write_pyramid(
    path: Path,
    image: np.ndarray,
    *,
    levels: int = 3,
    tile: tuple[int, int] | None = (64, 64),
    compression: str | None = "lzw",
    compressionargs: dict | None = None,
    photometric: str = "rgb",
    subsampling: tuple[int, int] | None = None,
    predictor: bool = False,
    bigtiff: bool = False,
    ome: bool = False,
    chain: bool = False,
    rowsperstrip: int | None = None,
) -> Path:
    """Write ``image`` plus ``levels - 1`` reduced copies.

    SubIFD pyramids are the default; ``chain=True`` writes reduced images as
    ordinary IFDs flagged with NewSubfileType=1 instead.
    """
    kwargs: dict = {"compression": compression, "photometric": photometric}
    if compressionargs:
        kwargs["compressionargs"] = compressionargs
    if subsampling:
        kwargs["subsampling"] = subsampling
    if predictor:
        kwargs["predictor"] = True
    if tile:
        kwargs["tile"] = tile
    elif rowsperstrip:
        kwargs["rowsperstrip"] = rowsperstrip
    with tifffile.TiffWriter(path, bigtiff=bigtiff, ome=ome) as writer:
        if ome:
            kwargs["metadata"] = {"axes": "YXS"}
        else:
            kwargs["metadata"] = None
        writer.write(image, subifds=0 if chain else levels - 1, **kwargs)
        kwargs.pop("metadata", None)
        for i in range(1, levels):
            reduced = downsample(image, 2**i)
            writer.write(reduced, subfiletype=1, **kwargs)
    return path


@pytest.fixture
def rgb_image() -> np.ndarray:
    return make_image(300, 420)


@pytest.fixture
def lzw_pyramid(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    return write_pyramid(tmp_path / "lzw.tif", rgb_image, predictor=True)


@pytest.fixture
def jpeg_pyramid(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    return write_pyramid(
        tmp_path / "jpeg.tif",
        rgb_image,
        compression="jpeg",
        compressionargs={"level": 90},
        photometric="ycbcr",
        subsampling=(2, 2),
    )


def sample_svs_paths() -> list[str]:
    return sorted(glob.glob(str(ROOT / "*.svs")))


def sample_path(name: str) -> str | None:
    candidate = ROOT / "samples" / name
    return str(candidate) if candidate.exists() else None


requires_svs = pytest.mark.skipif(
    not sample_svs_paths(), reason="no *.svs samples in repo root"
)


def requires_sample(name: str):
    return pytest.mark.skipif(
        sample_path(name) is None, reason=f"samples/{name} not downloaded"
    )


def has_openslide() -> bool:
    try:
        import openslide  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


requires_openslide = pytest.mark.skipif(
    not has_openslide(), reason="openslide not available"
)


def load_layout(path: Path):
    from tiffview import layout as layout_mod

    return layout_mod.discover(layout_mod.read_info(str(path)))
