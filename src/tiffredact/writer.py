"""Write the redacted file with tifftools, substituting only the replaced chunks.

tifftools copies every tile/strip opaquely from ``ifd['path_or_fobj']`` using the
offsets and byte counts in the IFD. We hand it a *virtual* source whose address
space is the original file followed by a scratch file holding the replacement
chunks, and point the replaced entries at the scratch region. The output is a
clean rewrite with no orphaned bytes.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO

import tifftools

from tiffredact.layout import (
    TAG_STRIP_BYTE_COUNTS,
    TAG_STRIP_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_TILE_OFFSETS,
    iter_all_ifds,
)

logger = logging.getLogger(__name__)


class WriterError(Exception):
    """Something went wrong assembling the output file."""


@dataclass(frozen=True)
class Placement:
    """Where a replacement chunk lives in the virtual address space."""

    virtual_offset: int
    length: int


class ScratchStore:
    """Append-only store of replacement chunks, addressed past the original file."""

    def __init__(self, base: int, directory: str | None = None) -> None:
        self.base = base
        fd, self.path = tempfile.mkstemp(prefix="tiffredact-scratch-", dir=directory)
        self._fh: BinaryIO = os.fdopen(fd, "w+b")
        self.size = 0

    def add(self, data: bytes) -> Placement:
        self._fh.seek(self.size)
        self._fh.write(data)
        placement = Placement(self.base + self.size, len(data))
        self.size += len(data)
        return placement

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def __enter__(self) -> ScratchStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VirtualSource(io.RawIOBase):
    """Read-only file view: original bytes first, scratch bytes after them."""

    def __init__(
        self, original_path: str, scratch_path: str, original_size: int
    ) -> None:
        super().__init__()
        self._orig = open(original_path, "rb")
        self._scratch = open(scratch_path, "rb")
        self._split = original_size
        self._pos = 0
        self.name = f"{original_path}+scratch"

    @property
    def size(self) -> int:
        self._scratch.seek(0, os.SEEK_END)
        return self._split + self._scratch.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence {whence}")
        return self._pos

    def truncate(self, size: int | None = None) -> int:
        raise io.UnsupportedOperation("VirtualSource is read-only")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        parts: list[bytes] = []
        remaining = n
        while remaining > 0:
            if self._pos < self._split:
                self._orig.seek(self._pos)
                piece = self._orig.read(min(remaining, self._split - self._pos))
            else:
                self._scratch.seek(self._pos - self._split)
                piece = self._scratch.read(remaining)
            if not piece:
                break
            parts.append(piece)
            self._pos += len(piece)
            remaining -= len(piece)
        return b"".join(parts)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self._orig.close()
            self._scratch.close()
        super().close()


def _offset_tags(ifd: dict[str, Any]) -> tuple[int, int]:
    if TAG_TILE_OFFSETS in ifd["tags"]:
        return TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS
    if TAG_STRIP_OFFSETS in ifd["tags"]:
        return TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS
    raise WriterError(f"IFD at offset {ifd['offset']} has neither tiles nor strips")


def apply_replacements(
    info: dict[str, Any],
    replacements: dict[int, dict[int, Placement]],
    source: VirtualSource,
) -> None:
    """Point every IFD at ``source`` and patch replaced chunk offsets/lengths."""
    total = source.size
    seen: set[int] = set()
    for ifd in iter_all_ifds(info):
        ifd["path_or_fobj"] = source
        ifd["size"] = total
        offset = int(ifd["offset"])
        seen.add(offset)
        per_chunk = replacements.get(offset)
        if not per_chunk:
            continue
        offsets_tag, counts_tag = _offset_tags(ifd)
        offsets = list(ifd["tags"][offsets_tag]["data"])
        counts = list(ifd["tags"][counts_tag]["data"])
        for index, placement in per_chunk.items():
            if index >= len(offsets):
                raise WriterError(f"Chunk {index} out of range for IFD at {offset}")
            offsets[index] = placement.virtual_offset
            counts[index] = placement.length
        ifd["tags"][offsets_tag]["data"] = offsets
        ifd["tags"][counts_tag]["data"] = counts
    missing = set(replacements) - seen
    if missing:
        raise WriterError(f"Replacements target IFDs not in file: {sorted(missing)}")


def drop_ifds(info: dict[str, Any], offsets: set[int]) -> list[int]:
    """Remove top-level IFDs by offset; returns the offsets actually removed."""
    before = [int(ifd["offset"]) for ifd in info["ifds"]]
    info["ifds"] = [ifd for ifd in info["ifds"] if int(ifd["offset"]) not in offsets]
    return [o for o in before if o in offsets]


def write_output(
    info: dict[str, Any], output_path: str, overwrite: bool = False
) -> None:
    if os.path.exists(output_path) and not overwrite:
        raise WriterError(f"Output {output_path} exists; pass overwrite to replace it")
    tifftools.write_tiff(info, output_path, allowExisting=overwrite)


def iter_chunk_placements(
    replacements: dict[int, dict[int, Placement]],
) -> Iterator[tuple[int, int, Placement]]:
    for ifd_offset, per_chunk in replacements.items():
        for index, placement in per_chunk.items():
            yield ifd_offset, index, placement
