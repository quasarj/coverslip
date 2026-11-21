"""Discover the pyramid structure of a slide from its TIFF IFDs.

The IFD tree comes from ``tifftools.read_tiff`` so that the very same structure
can later be handed back to ``tifftools.write_tiff``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from xml.etree import ElementTree

import tifftools

from coverslip.model import ImageIFD, Layout, PyramidImage, Role

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


TIFFTOOLS_DOUBLE_REFERENCE_MESSAGE = "SubIFDs are double referenced"


class _ExplainDoubleReferencedSubIFDs(logging.Filter):
    """Swallow tifftools' terse warning and remember that it fired.

    Bio-Formats (and some other writers) link each reduced-resolution IFD both
    through the ``SubIFDs`` tag and through the previous sub-IFD's next-IFD
    pointer. tifftools stops following the pointer when it sees the same IFD
    is already listed in the tag, so nothing is lost; the output simply links
    the levels through the tag alone, which every reader we know of accepts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() == TIFFTOOLS_DOUBLE_REFERENCE_MESSAGE:
            self.count += 1
            return False
        return True


def read_info(path: str) -> dict[str, Any]:
    """Read the IFD tree with tifftools and prune SubIFD chains that leak."""
    tifftools_logger = logging.getLogger("tifftools.tifftools")
    explain = _ExplainDoubleReferencedSubIFDs()
    tifftools_logger.addFilter(explain)
    try:
        info = tifftools.read_tiff(path)
    finally:
        tifftools_logger.removeFilter(explain)
    if explain.count:
        logger.info(
            "%d sub-IFDs are linked both by the SubIFDs tag and by next-IFD "
            "pointers; the output links them by the SubIFDs tag only",
            explain.count,
        )
    prune_leaked_subifd_chains(info)
    return info


def prune_leaked_subifd_chains(info: dict[str, Any]) -> int:
    """Cut SubIFD chains whose next-IFD pointer runs into the main chain.

    Some writers leave the "next IFD" pointer of the last sub-resolution IFD
    pointing at the following top-level IFD. Following it would attach the
    main image to the wrong pyramid and write it twice. Returns the number of
    chains truncated.
    """
    top_offsets = {int(ifd["offset"]) for ifd in info["ifds"]}
    truncated = 0
    for ifd in info["ifds"]:
        entry = ifd["tags"].get(TAG_SUB_IFDS)
        if entry is None:
            continue
        for index, chain in enumerate(entry.get("ifds", [])):
            for position, sub in enumerate(chain):
                if int(sub["offset"]) in top_offsets:
                    logger.warning(
                        "SubIFD chain of IFD at %d runs into top-level IFD at %d; truncating",
                        ifd["offset"],
                        sub["offset"],
                    )
                    entry["ifds"][index] = chain[:position]
                    truncated += 1
                    break
    return truncated


def _entry_data(ifd: dict[str, Any], tag: int) -> Any:
    entry = ifd["tags"][tag]
    if "data" not in entry:
        raise LayoutError(
            f"Tag {tag} of IFD at offset {ifd.get('offset')} has no readable data; "
            "the file may be truncated"
        )
    return entry["data"]


def tag_value(ifd: dict[str, Any], tag: int, default: Any = None) -> Any:
    if tag not in ifd["tags"]:
        return default
    data = _entry_data(ifd, tag)
    if isinstance(data, (bytes, str)):
        return data
    if len(data) == 1:
        return data[0]
    return data


def _tag_list(ifd: dict[str, Any], tag: int) -> list[int]:
    if tag not in ifd["tags"]:
        return []
    data = _entry_data(ifd, tag)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def _tag_bytes(ifd: dict[str, Any], tag: int) -> bytes | None:
    if tag not in ifd["tags"]:
        return None
    data = _entry_data(ifd, tag)
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
    # Like BitsPerSample, SampleFormat may carry one value per sample.
    sample_formats = _tag_list(ifd, TAG_SAMPLE_FORMAT) or [1]
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
        sample_format=int(sample_formats[0]),
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


def ome_image_names(description: str) -> dict[int, str]:
    """Map top-level IFD index -> OME Image name, from the OME-XML header."""
    if "<OME" not in description:
        return {}
    try:
        root = ElementTree.fromstring(description)
    except ElementTree.ParseError as exc:
        logger.warning("Could not parse OME-XML: %s", exc)
        return {}
    names: dict[int, str] = {}
    for image in root.iter():
        if not image.tag.endswith("}Image") and image.tag != "Image":
            continue
        name = image.get("Name", "")
        for tiffdata in image.iter():
            if not (tiffdata.tag.endswith("}TiffData") or tiffdata.tag == "TiffData"):
                continue
            first = int(tiffdata.get("IFD", "0"))
            count = int(tiffdata.get("PlaneCount", "1"))
            for index in range(first, first + count):
                names[index] = name
    return names


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
            logger.warning(
                "Unrecognised Aperio IFD at offset %d: %s", ifd.offset, ifd.label()
            )
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
        names = ome_image_names(chain[0].description) if vendor == "ome" else {}
        _separate_secondary_images(chain, layout, names)
    return layout


def _separate_secondary_images(
    chain: list[ImageIFD], layout: Layout, names: dict[int, str]
) -> None:
    """Keep only pyramids whose full image matches the primary (largest) size.

    OME-TIFF files may carry overview/macro/label images alongside the slide.
    Those live in their own coordinate space, so regions must not be mapped
    onto them. Multi-plane images (channels, z) share the primary size and
    all receive the redaction.
    """
    if not layout.images:
        return
    primary = max(layout.images, key=lambda img: img.full.width * img.full.height)
    size = (primary.full.width, primary.full.height)
    index_by_offset = {ifd.offset: i for i, ifd in enumerate(chain)}
    kept: list[PyramidImage] = []
    for image in layout.images:
        if (image.full.width, image.full.height) == size:
            kept.append(image)
            continue
        name = names.get(index_by_offset.get(image.full.offset, -1), "").lower()
        if "macro" in name or "overview" in name:
            group, role = layout.macro, Role.MACRO
        elif "label" in name:
            group, role = layout.label, Role.LABEL
        else:
            group, role = layout.other, Role.OTHER
        for level in image.levels:
            level.role = role
            group.append(level)
    layout.images = kept
