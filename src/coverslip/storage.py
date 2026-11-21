"""Where a slide lives: ranged reads and patch application.

In-place redaction talks to storage only through :class:`Storage`, so another
backend (an S3 object assembled with multipart copy, say) needs to implement
these three methods and nothing else.
"""

from __future__ import annotations

import logging
import os
from typing import BinaryIO, Protocol

from coverslip.patches import PatchSet

logger = logging.getLogger(__name__)

ZERO_BLOCK = 1 << 20


class StorageError(Exception):
    """The stored file is not in the state the operation expects."""


class Storage(Protocol):
    def size(self) -> int:
        """Current length in bytes."""
        ...

    def read(self, offset: int, length: int) -> bytes:
        """Exactly ``length`` bytes at ``offset``; raises on a short read."""
        ...

    def apply(self, patch_set: PatchSet) -> None:
        """Apply every edit of ``patch_set``."""
        ...


class LocalFileStorage:
    """A file on a local (or range-aware mounted) filesystem."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._reader: BinaryIO = open(path, "rb")

    def size(self) -> int:
        return os.path.getsize(self.path)

    def read(self, offset: int, length: int) -> bytes:
        self._reader.seek(offset)
        data = self._reader.read(length)
        if len(data) != length:
            raise StorageError(
                f"Short read at {offset} of {self.path}: wanted {length} bytes, "
                f"got {len(data)}"
            )
        return data

    def apply(self, patch_set: PatchSet) -> None:
        """Write patches in order, then zeros, then flush to disk."""
        if self.size() != patch_set.source_size:
            raise StorageError(
                f"{self.path} is {self.size()} bytes but the patch set was computed "
                f"for {patch_set.source_size}; it changed since it was read"
            )
        with open(self.path, "r+b") as fh:
            for patch in patch_set.writes:
                fh.seek(patch.offset)
                fh.write(patch.data)
            for zero in patch_set.zeros:
                _write_zeros(fh, zero.offset, zero.length)
            fh.flush()
            os.fsync(fh.fileno())
        logger.debug(
            "applied %d patches and %d zero ranges to %s",
            len(patch_set.writes),
            len(patch_set.zeros),
            self.path,
        )

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> LocalFileStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _write_zeros(fh: BinaryIO, offset: int, length: int) -> None:
    fh.seek(offset)
    remaining = length
    while remaining > 0:
        block = min(remaining, ZERO_BLOCK)
        fh.write(bytes(block))
        remaining -= block
