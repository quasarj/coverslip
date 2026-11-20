"""Record of an in-place redaction, consumed by the range-aware verifier.

After an in-place run there is no source file to compare against; the manifest
says which byte ranges were written and zeroed and where the replaced chunks
used to live, so the audit reads only those ranges.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from tiffredact.model import Region
from tiffredact.patches import PatchSet, Range

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = ".tiffredact-manifest.json"


class ManifestError(Exception):
    """The manifest is missing, unreadable or from an unknown version."""


@dataclass(frozen=True)
class PatchedRange:
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class ChunkRecord:
    """One replaced tile/strip: where it was, where it is now."""

    ifd_offset: int
    index: int
    old_offset: int
    old_length: int
    old_sha256: str
    new_offset: int
    new_length: int
    # Another chunk still points at the old bytes (deduplicated tiles), so
    # they legitimately survive.
    old_slot_shared: bool = False


@dataclass
class Manifest:
    path: str
    source_size: int
    final_size: int
    regions: list[Region] = field(default_factory=list)
    fill: tuple[int, int, int] = (0, 0, 0)
    scrubbed: bool = False
    keep_tags: list[int] = field(default_factory=list)
    dropped_ifds: list[int] = field(default_factory=list)
    patched: list[PatchedRange] = field(default_factory=list)
    zeroed: list[Range] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    version: int = MANIFEST_VERSION

    def record_patches(self, patch_set: PatchSet) -> None:
        self.patched = [
            PatchedRange(patch.offset, len(patch.data), sha256_hex(patch.data))
            for patch in patch_set.writes
        ]
        self.zeroed = [(zero.offset, zero.end) for zero in patch_set.zeros]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=1)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> Manifest:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc
        if raw.get("version") != MANIFEST_VERSION:
            raise ManifestError(
                f"Manifest {path} has version {raw.get('version')!r}; "
                f"expected {MANIFEST_VERSION}"
            )
        try:
            return _manifest_from_json(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Manifest {path} is malformed: {exc}") from exc


def _manifest_from_json(raw: dict) -> Manifest:
    return Manifest(
        path=raw["path"],
        source_size=int(raw["source_size"]),
        final_size=int(raw["final_size"]),
        regions=[Region(**region) for region in raw["regions"]],
        fill=tuple(int(v) for v in raw["fill"]),  # type: ignore[arg-type]
        scrubbed=bool(raw["scrubbed"]),
        keep_tags=[int(tag) for tag in raw["keep_tags"]],
        dropped_ifds=[int(offset) for offset in raw["dropped_ifds"]],
        patched=[PatchedRange(**item) for item in raw["patched"]],
        zeroed=[(int(start), int(end)) for start, end in raw["zeroed"]],
        chunks=[ChunkRecord(**item) for item in raw["chunks"]],
    )


def default_manifest_path(slide_path: str) -> str:
    return f"{slide_path}{MANIFEST_SUFFIX}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
