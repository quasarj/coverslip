"""Data model: regions, chunks (tiles or strips), image IFDs and pyramids."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    """What an IFD represents within a slide."""

    FULL = "full"
    LEVEL = "level"
    THUMBNAIL = "thumbnail"
    LABEL = "label"
    MACRO = "macro"
    OTHER = "other"


@dataclass(frozen=True)
class Region:
    """An axis-aligned rectangle in integer pixel coordinates; half-open."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Region must have positive size, got {self}")

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    def intersection(self, other: Region) -> Region | None:
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return Region(x0, y0, x1 - x0, y1 - y0)

    def translate(self, dx: int, dy: int) -> Region:
        return Region(self.x + dx, self.y + dy, self.width, self.height)

    def clip(self, width: int, height: int) -> Region | None:
        return self.intersection(Region(0, 0, width, height))

    @classmethod
    def parse(cls, text: str) -> Region:
        """Parse ``"x,y,width,height"``."""
        parts = text.replace(" ", "").split(",")
        if len(parts) != 4:
            raise ValueError(f"Expected x,y,width,height but got {text!r}")
        try:
            x, y, w, h = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Region values must be integers: {text!r}") from exc
        return cls(x, y, w, h)


@dataclass(frozen=True)
class Chunk:
    """One tile or strip of an IFD, with its nominal geometry in level pixels.

    Tiles at the right or bottom edge extend past the image bounds; the codestream
    still covers the full nominal tile. Strips are clipped to the image height.
    """

    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def region(self) -> Region:
        return Region(self.x, self.y, self.width, self.height)


@dataclass
class ImageIFD:
    """The subset of a TIFF IFD that redaction needs."""

    offset: int
    width: int
    height: int
    tiled: bool
    chunk_width: int
    chunk_height: int
    compression: int
    photometric: int
    bits_per_sample: int
    samples_per_pixel: int
    planar_config: int
    predictor: int
    sample_format: int
    subsampling: tuple[int, int] | None
    jpegtables: bytes | None
    new_subfile_type: int
    description: str
    data_offsets: list[int]
    data_bytecounts: list[int]
    big_endian: bool
    role: Role = Role.OTHER

    @property
    def chunks_across(self) -> int:
        return math.ceil(self.width / self.chunk_width)

    @property
    def chunks_down(self) -> int:
        return math.ceil(self.height / self.chunk_height)

    @property
    def chunk_count(self) -> int:
        return self.chunks_across * self.chunks_down

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def label(self) -> str:
        kind = "tiled" if self.tiled else "stripped"
        return (
            f"{self.role.value:9s} {self.width}x{self.height} {kind} "
            f"{self.chunk_width}x{self.chunk_height} comp={self.compression} "
            f"photometric={self.photometric} chunks={self.chunk_count}"
        )


@dataclass
class PyramidImage:
    """A full-resolution image and all reduced-resolution copies of it."""

    levels: list[ImageIFD] = field(default_factory=list)

    @property
    def full(self) -> ImageIFD:
        return self.levels[0]


@dataclass
class Layout:
    """Everything we discovered about a slide file."""

    path: str
    bigtiff: bool
    vendor: str
    images: list[PyramidImage] = field(default_factory=list)
    label: list[ImageIFD] = field(default_factory=list)
    macro: list[ImageIFD] = field(default_factory=list)
    other: list[ImageIFD] = field(default_factory=list)

    def all_ifds(self) -> list[ImageIFD]:
        found: list[ImageIFD] = []
        for image in self.images:
            found.extend(image.levels)
        found.extend(self.label)
        found.extend(self.macro)
        found.extend(self.other)
        return found
