"""Discover the pyramid structure of a slide from its TIFF IFDs.

The IFD tree comes from ``tifftools.read_tiff`` so that the very same structure
can later be handed back to ``tifftools.write_tiff``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import tifftools

from tiffredact.model import ImageIFD, Layout, PyramidImage, Role

logger = logging.getLogger(__name__)

TAG_NEW_SUBFILE_TYPE = 254
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_IMAGE_DESCRIPTION = 270
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIG = 284
TAG_PREDICTOR = 317
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_SUB_IFDS = 330
TAG_SAMPLE_FORMAT = 339
TAG_JPEG_TABLES = 347
TAG_YCBCR_SUBSAMPLING = 530

SUBFILE_REDUCED = 1
SUBFILE_APERIO_MACRO = 9

ASPECT_TOLERANCE = 0.02


class LayoutError(Exception):
    """The file structure is not something we know how to redact."""


def read_info(path: str) -> dict[str, Any]:
    """Read the IFD tree with tifftools."""
    return tifftools.read_tiff(path)


def tag_value(ifd: dict[str, Any], tag: int, default: Any = None) -> Any:
    entry = ifd["tags"].get(tag)
    if entry is None:
        return default
    data = entry["data"]
    if isinstance(data, (bytes, str)):
        return data
    if len(data) == 1:
        return data[0]
    return data


def _tag_list(ifd: dict[str, Any], tag: int) -> list[int]:
    entry = ifd["tags"].get(tag)
    if entry is None:
        return []
    data = entry["data"]
    return list(data) if isinstance(data, (list, tuple)) else [data]


def _tag_bytes(ifd: dict[str, Any], tag: int) -> bytes | None:
    entry = ifd["tags"].get(tag)
    if entry is None:
        return None
    data = entry["data"]
    if isinstance(data, str):
        return data.encode("latin-1")
    if isinstance(data, bytes):
        return data
    return bytes(data)


def _description(ifd: dict[str, Any]) -> str:
    data = tag_value(ifd, TAG_IMAGE_DESCRIPTION, "")
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def image_from_ifd(ifd: dict[str, Any]) -> ImageIFD:
    """Build an :class:`ImageIFD` from a tifftools IFD record."""
    tags = ifd["tags"]
    width = int(tag_value(ifd, TAG_IMAGE_WIDTH))
    height = int(tag_value(ifd, TAG_IMAGE_LENGTH))
    tiled = TAG_TILE_OFFSETS in tags
    if tiled:
        chunk_width = int(tag_value(ifd, TAG_TILE_WIDTH))
        chunk_height = int(tag_value(ifd, TAG_TILE_LENGTH))
        offsets = _tag_list(ifd, TAG_TILE_OFFSETS)
        counts = _tag_list(ifd, TAG_TILE_BYTE_COUNTS)
    else:
        chunk_width = width
        rows = tag_value(ifd, TAG_ROWS_PER_STRIP, height)
        chunk_height = min(int(rows), height) if rows else height
        offsets = _tag_list(ifd, TAG_STRIP_OFFSETS)
        counts = _tag_list(ifd, TAG_STRIP_BYTE_COUNTS)
    bits = _tag_list(ifd, TAG_BITS_PER_SAMPLE) or [1]
    subsampling_raw = _tag_list(ifd, TAG_YCBCR_SUBSAMPLING)
    subsampling = (
        (int(subsampling_raw[0]), int(subsampling_raw[1]))
        if len(subsampling_raw) == 2
        else None
    )
    return ImageIFD(
        offset=int(ifd["offset"]),
        width=width,
        height=height,
        tiled=tiled,
        chunk_width=chunk_width,
        chunk_height=chunk_height,
        compression=int(tag_value(ifd, TAG_COMPRESSION, 1)),
        photometric=int(tag_value(ifd, TAG_PHOTOMETRIC, 2)),
        bits_per_sample=int(bits[0]),
        samples_per_pixel=int(tag_value(ifd, TAG_SAMPLES_PER_PIXEL, 1)),
        planar_config=int(tag_value(ifd, TAG_PLANAR_CONFIG, 1)),
        predictor=int(tag_value(ifd, TAG_PREDICTOR, 1)),
        sample_format=int(tag_value(ifd, TAG_SAMPLE_FORMAT, 1)),
        subsampling=subsampling,
        jpegtables=_tag_bytes(ifd, TAG_JPEG_TABLES),
        new_subfile_type=int(tag_value(ifd, TAG_NEW_SUBFILE_TYPE, 0)),
        description=_description(ifd),
        data_offsets=[int(o) for o in offsets],
        data_bytecounts=[int(c) for c in counts],
        big_endian=bool(ifd.get("bigEndian", False)),
    )


def iter_sub_ifds(ifd: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every IFD reachable through this IFD's SubIFDs tag, depth first."""
    entry = ifd["tags"].get(TAG_SUB_IFDS)
    if entry is None:
        return
    for chain in entry.get("ifds", []):
        for sub in chain:
            yield sub
            yield from iter_sub_ifds(sub)


def iter_all_ifds(info: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every IFD in the file: the main chain and all SubIFD trees."""
    for ifd in info["ifds"]:
        yield ifd
        yield from iter_sub_ifds(ifd)


def detect_vendor(first_description: str) -> str:
    if first_description.startswith("Aperio"):
        return "aperio"
    if "<OME" in first_description or "<?xml" in first_description[:64]:
        return "ome"
    return "generic"


def same_aspect(a: ImageIFD, b: ImageIFD) -> bool:
    return abs(a.aspect - b.aspect) <= ASPECT_TOLERANCE * a.aspect


def is_reduced_copy(full: ImageIFD, candidate: ImageIFD) -> bool:
    return (
        candidate.width < full.width
        and candidate.height < full.height
        and same_aspect(full, candidate)
    )


def _assign_roles(image: PyramidImage) -> None:
    image.levels.sort(key=lambda ifd: ifd.width, reverse=True)
    for i, level in enumerate(image.levels):
        if i == 0:
            level.role = Role.FULL
        elif level.tiled:
            level.role = Role.LEVEL
        else:
            level.role = Role.THUMBNAIL


def _discover_aperio(chain: list[ImageIFD], layout: Layout) -> None:
    image = PyramidImage([chain[0]])
    for ifd in chain[1:]:
        if ifd.new_subfile_type == SUBFILE_APERIO_MACRO:
            ifd.role = Role.MACRO
            layout.macro.append(ifd)
        elif ifd.new_subfile_type == SUBFILE_REDUCED:
            ifd.role = Role.LABEL
            layout.label.append(ifd)
        elif is_reduced_copy(image.full, ifd):
            image.levels.append(ifd)
        else:
            logger.warning("Unrecognised Aperio IFD at offset %d: %s", ifd.offset, ifd.label())
            layout.other.append(ifd)
    _assign_roles(image)
    layout.images.append(image)


def _discover_generic(
    chain: list[ImageIFD], subs: dict[int, list[ImageIFD]], layout: Layout
) -> None:
    for ifd in chain:
        sub_levels = subs.get(ifd.offset, [])
        if sub_levels:
            image = PyramidImage([ifd] + sub_levels)
            _assign_roles(image)
            layout.images.append(image)
            continue
        if layout.images and is_reduced_copy(layout.images[-1].full, ifd):
            layout.images[-1].levels.append(ifd)
            _assign_roles(layout.images[-1])
            continue
        if ifd.new_subfile_type & SUBFILE_REDUCED:
            logger.warning(
                "Reduced-resolution IFD at offset %d matches no full image", ifd.offset
            )
            layout.other.append(ifd)
            continue
        image = PyramidImage([ifd])
        _assign_roles(image)
        layout.images.append(image)


def discover(info: dict[str, Any]) -> Layout:
    """Classify every IFD of a file into pyramids, thumbnails, label and macro."""
    if not info["ifds"]:
        raise LayoutError("File contains no IFDs")
    chain = [image_from_ifd(ifd) for ifd in info["ifds"]]
    subs: dict[int, list[ImageIFD]] = {}
    for ifd in info["ifds"]:
        sub_images = [image_from_ifd(sub) for sub in iter_sub_ifds(ifd)]
        if sub_images:
            subs[int(ifd["offset"])] = sub_images
    vendor = detect_vendor(chain[0].description)
    layout = Layout(
        path=str(info.get("path_or_fobj", "")),
        bigtiff=bool(info.get("bigtiff", False)),
        vendor=vendor,
    )
    if vendor == "aperio":
        _discover_aperio(chain, layout)
    else:
        _discover_generic(chain, subs, layout)
    return layout


def describe(layout: Layout) -> str:
    lines = [f"{layout.path}", f"format: {'BigTIFF' if layout.bigtiff else 'classic TIFF'}  vendor: {layout.vendor}"]
    for n, image in enumerate(layout.images):
        lines.append(f"image {n}:")
        for level in image.levels:
            scale = image.full.width / level.width
            lines.append(f"  ifd@{level.offset:<12d} {level.label()} downsample={scale:.4f}")
    for name, group in (("label", layout.label), ("macro", layout.macro), ("other", layout.other)):
        for ifd in group:
            lines.append(f"{name}: ifd@{ifd.offset:<12d} {ifd.label()}")
    return "\n".join(lines)
