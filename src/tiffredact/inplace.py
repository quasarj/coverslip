"""In-place redaction: compute byte edits instead of rewriting the whole file.

The planner compares the IFD tree as read (``before``) with the tree after
label/macro removal, scrubbing and chunk replacement (``after``) and emits a
:class:`~tiffredact.patches.PatchSet`:

* replaced chunks are written over their old slot when they fit, otherwise
  appended to the end of the file;
* changed tag values (chunk offset/byte count arrays, rewritten descriptions)
  are written over their old slot, inline, or appended if they grew;
* every IFD entry table that changed is rewritten at its own offset;
* every byte the ``before`` tree referenced and the ``after`` tree does not is
  zeroed. That single rule covers the old slots of moved chunks, data of
  removed tags, dropped IFDs with their strips, and the tails of anything that
  shrank. Bytes still referenced (deduplicated tiles, shared tag data) are
  never zeroed because they are still in the ``after`` tree.
"""

from __future__ import annotations

import copy
import logging
import os
import struct
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import tifftools

from tiffredact.codecs.fill import Rgb
from tiffredact.layout import (
    TAG_STRIP_BYTE_COUNTS,
    TAG_STRIP_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_TILE_OFFSETS,
)
from tiffredact.manifest import (
    ChunkRecord,
    Manifest,
    default_manifest_path,
    sha256_hex,
)
from tiffredact.model import ImageIFD, Layout, RegionInput
from tiffredact.patches import Patch, PatchSet, Range, Zero, subtract_ranges
from tiffredact.redact import (
    ChunkWork,
    ReadRange,
    RedactionReport,
    apply_metadata_changes,
    load_plan,
    rewrite_chunks,
)
from tiffredact.storage import LocalFileStorage, Storage

logger = logging.getLogger(__name__)

CLASSIC_TIFF_LIMIT = 1 << 32


class InPlaceError(Exception):
    """The file cannot be redacted in place; a full rewrite is needed."""


@dataclass(frozen=True)
class TiffFormat:
    """Field widths that differ between classic TIFF and BigTIFF."""

    bigtiff: bool
    bom: str

    @property
    def pointer_size(self) -> int:
        return 8 if self.bigtiff else 4

    @property
    def pointer_pack(self) -> str:
        return "Q" if self.bigtiff else "L"

    @property
    def count_pack(self) -> str:
        return "Q" if self.bigtiff else "H"

    @property
    def count_size(self) -> int:
        return 8 if self.bigtiff else 2

    @property
    def entry_size(self) -> int:
        return 20 if self.bigtiff else 12

    @property
    def header_length(self) -> int:
        return 16 if self.bigtiff else 8

    @property
    def first_ifd_position(self) -> int:
        return 8 if self.bigtiff else 4

    def table_length(self, entries: int) -> int:
        """Entry count field, the entries and the next-IFD pointer."""
        return self.count_size + entries * self.entry_size + self.pointer_size

    def pack_pointer(self, value: int) -> bytes:
        return struct.pack(self.bom + self.pointer_pack, value)

    def pack_entry_head(self, tag: int, datatype: int, count: int) -> bytes:
        return struct.pack(self.bom + "HH" + self.pointer_pack, tag, datatype, count)

    def split_entries(self, table: bytes, entries: int) -> list[tuple[int, bytes]]:
        """``(tag, raw entry bytes)`` for every entry of a raw IFD table."""
        found = []
        for n in range(entries):
            start = self.count_size + n * self.entry_size
            raw = table[start : start + self.entry_size]
            found.append((struct.unpack(self.bom + "H", raw[:2])[0], raw))
        return found

    def next_pointer(self, table: bytes) -> int:
        tail = table[-self.pointer_size :]
        return struct.unpack(self.bom + self.pointer_pack, tail)[0]


@dataclass(frozen=True)
class ChunkReplacement:
    ifd_offset: int
    index: int
    data: bytes
    old_sha256: str


def walk_ifds(ifds: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Every IFD reachable from ``ifds`` through any IFD-valued tag."""
    for ifd in ifds:
        yield ifd
        for entry in ifd["tags"].values():
            for chain in entry.get("ifds", []):
                yield from walk_ifds(chain)


def chunk_tags(ifd: dict[str, Any]) -> tuple[int, int] | None:
    if TAG_TILE_OFFSETS in ifd["tags"]:
        return TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS
    if TAG_STRIP_OFFSETS in ifd["tags"]:
        return TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS
    return None


def chunk_extents(ifd: dict[str, Any]) -> list[Range]:
    """Byte ranges of the non-empty tiles/strips an IFD points at."""
    tags = chunk_tags(ifd)
    if tags is None:
        return []
    offsets = ifd["tags"][tags[0]]["data"]
    counts = ifd["tags"][tags[1]]["data"]
    return [(o, o + c) for o, c in zip(offsets, counts) if c > 0]


def tag_data_extent(entry: dict[str, Any]) -> Range | None:
    """Byte range of a tag's out-of-line value as read; None if inline."""
    if "offset" not in entry:
        return None
    size = tifftools.Datatype[entry["datatype"]].size
    return entry["offset"], entry["offset"] + entry["count"] * size


def encode_value(entry: dict[str, Any], bom: str) -> tuple[bytes, int]:
    """Serialise a tag value the way tifftools does; returns (bytes, count)."""
    datatype = tifftools.Datatype[entry["datatype"]]
    data = entry["data"]
    if datatype.pack:
        count = len(data) // len(datatype.pack)
        try:
            return struct.pack(bom + datatype.pack * count, *data), count
        except struct.error as exc:
            raise InPlaceError(
                f"A new value does not fit datatype {datatype.name} ({exc}); "
                "use a full rewrite"
            ) from exc
    if datatype == tifftools.Datatype.ASCII:
        payload = (data if isinstance(data, bytes) else data.encode()) + b"\x00"
        return payload, len(payload)
    return bytes(data), len(data)


def _value_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return int(after["datatype"]) != int(before["datatype"]) or after.get(
        "data"
    ) != before.get("data")


def _is_offset_data_tag(tag: int) -> bool:
    return tag in tifftools.Tag and tifftools.Tag[tag].isOffsetData()


class PatchPlanner:
    """Turn the difference between two IFD trees of one file into a patch set."""

    def __init__(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        read: ReadRange,
        source_size: int,
    ) -> None:
        if before.get("ndpi"):
            raise InPlaceError("NDPI files cannot be redacted in place")
        if not after["ifds"]:
            raise InPlaceError("No IFD would remain in the file")
        self.before = before
        self.after = after
        self.read = read
        self.source_size = source_size
        self.fmt = TiffFormat(bool(before["bigtiff"]), before["endianPack"])
        self._end = source_size
        self._appends: list[Patch] = []
        self._chunk_overwrites: list[Patch] = []
        self._tag_writes: list[Patch] = []
        self._table_writes: list[Patch] = []
        self._live: list[Range] = [(0, self.fmt.header_length)]
        self._before_by_offset = {
            int(ifd["offset"]): ifd for ifd in walk_ifds(before["ifds"])
        }
        self._after_by_offset = {
            int(ifd["offset"]): ifd for ifd in walk_ifds(after["ifds"])
        }
        self._orphans = set(self._before_by_offset) - set(self._after_by_offset)

    # --- chunks -------------------------------------------------------------

    def place_chunks(self, replacements: list[ChunkReplacement]) -> list[ChunkRecord]:
        """Choose a position for every replaced chunk and update ``after``."""
        refcounts = self._slot_refcounts()
        by_ifd: dict[int, list[ChunkReplacement]] = {}
        for replacement in replacements:
            by_ifd.setdefault(replacement.ifd_offset, []).append(replacement)
        records: list[ChunkRecord] = []
        for ifd_offset, group in by_ifd.items():
            records.extend(self._place_ifd_chunks(ifd_offset, group, refcounts))
        return records

    def _slot_refcounts(self) -> Counter[int]:
        """How many chunks point at each data offset (deduplicated tiles > 1)."""
        refcounts: Counter[int] = Counter()
        for ifd in walk_ifds(self.before["ifds"]):
            refcounts.update(start for start, _ in chunk_extents(ifd))
        return refcounts

    def _place_ifd_chunks(
        self, ifd_offset: int, group: list[ChunkReplacement], refcounts: Counter[int]
    ) -> list[ChunkRecord]:
        ifd = self._after_by_offset.get(ifd_offset)
        tags = chunk_tags(ifd) if ifd is not None else None
        if ifd is None or tags is None:
            raise InPlaceError(f"Replaced chunks target unknown IFD at {ifd_offset}")
        offsets = list(ifd["tags"][tags[0]]["data"])
        counts = list(ifd["tags"][tags[1]]["data"])
        records = []
        for replacement in group:
            old_offset, old_length = (
                offsets[replacement.index],
                counts[replacement.index],
            )
            shared = old_length > 0 and refcounts[old_offset] > 1
            new_offset = self._place_chunk(
                replacement.data, old_offset, old_length, shared
            )
            offsets[replacement.index] = new_offset
            counts[replacement.index] = len(replacement.data)
            records.append(
                ChunkRecord(
                    ifd_offset=ifd_offset,
                    index=replacement.index,
                    old_offset=old_offset,
                    old_length=old_length,
                    old_sha256=replacement.old_sha256,
                    new_offset=new_offset,
                    new_length=len(replacement.data),
                    old_slot_shared=shared,
                )
            )
        ifd["tags"][tags[0]]["data"] = offsets
        ifd["tags"][tags[1]]["data"] = counts
        return records

    def _place_chunk(
        self, data: bytes, old_offset: int, old_length: int, shared: bool
    ) -> int:
        """Reuse the old slot when the chunk fits and nobody else reads it."""
        if len(data) <= old_length and not shared:
            self._chunk_overwrites.append(Patch(old_offset, data))
            return old_offset
        return self._append(data)

    def _append(self, data: bytes) -> int:
        if self._end % 2:  # TIFF values start on a word boundary
            self._appends.append(Patch(self._end, b"\x00"))
            self._end += 1
        offset = self._end
        if not self.fmt.bigtiff and offset + len(data) > CLASSIC_TIFF_LIMIT:
            raise InPlaceError(
                "Appending would grow this classic TIFF past 4 GB; use a full "
                "rewrite, which upgrades to BigTIFF"
            )
        self._appends.append(Patch(offset, data))
        self._end += len(data)
        return offset

    # --- IFDs ---------------------------------------------------------------

    def plan(self) -> PatchSet:
        """Emit the patch set; call after :meth:`place_chunks`."""
        top_next = self._top_level_next_pointers()
        planned: set[int] = set()
        for ifd in walk_ifds(self.after["ifds"]):
            offset = int(ifd["offset"])
            if offset in planned:
                continue
            planned.add(offset)
            self._plan_ifd(ifd, top_next)
            self._live.extend(chunk_extents(ifd))
        header = self._header_patch()
        writes = [
            *self._appends,
            *self._chunk_overwrites,
            *self._tag_writes,
            *self._table_writes,
            *header,
        ]
        zeros = [
            Zero(start, end - start)
            for start, end in subtract_ranges(self._before_extents(), self._live)
        ]
        return PatchSet(self.source_size, writes, zeros)

    def _top_level_next_pointers(self) -> dict[int, int]:
        """Each surviving top-level IFD points at the next survivor (or 0)."""
        offsets = [int(ifd["offset"]) for ifd in self.after["ifds"]]
        return dict(zip(offsets, [*offsets[1:], 0]))

    def _header_patch(self) -> list[Patch]:
        first = int(self.after["ifds"][0]["offset"])
        if first == int(self.before["firstifd"]):
            return []
        return [Patch(self.fmt.first_ifd_position, self.fmt.pack_pointer(first))]

    def _plan_ifd(self, ifd: dict[str, Any], top_next: dict[int, int]) -> None:
        """Rebuild the entry table of one surviving IFD; patch it if it changed."""
        offset = int(ifd["offset"])
        before = self._before_by_offset[offset]
        raw = self.read(offset, self.fmt.table_length(before["tagcount"]))
        entries = []
        for tag, raw_entry in self.fmt.split_entries(raw, before["tagcount"]):
            entry = self._plan_entry(tag, raw_entry, before, ifd)
            if entry is not None:
                entries.append(entry)
        table = (
            struct.pack(self.fmt.bom + self.fmt.count_pack, len(entries))
            + b"".join(entries)
            + self.fmt.pack_pointer(self._next_pointer(offset, raw, top_next))
        )
        self._live.append((offset, offset + len(table)))
        if table != raw[: len(table)]:
            self._table_writes.append(Patch(offset, table))

    def _next_pointer(self, offset: int, raw: bytes, top_next: dict[int, int]) -> int:
        if offset in top_next:
            return top_next[offset]
        # Sub-IFD chains keep their links, except one that ran into an IFD
        # that is being dropped.
        raw_next = self.fmt.next_pointer(raw)
        return 0 if raw_next in self._orphans else raw_next

    def _plan_entry(
        self, tag: int, raw_entry: bytes, before: dict[str, Any], after: dict[str, Any]
    ) -> bytes | None:
        """The entry's new bytes, or None when the tag is removed."""
        before_entry = before["tags"].get(tag)
        if before_entry is None:
            return self._keep_unparsed_entry(tag, raw_entry, before, after)
        after_entry = after["tags"].get(tag)
        if after_entry is None:
            if _is_offset_data_tag(tag):
                raise InPlaceError(
                    f"Removing tag {tag} would orphan the data it points at; "
                    "use a full rewrite"
                )
            return None
        if not _value_changed(before_entry, after_entry):
            extent = tag_data_extent(before_entry)
            if extent is not None:
                self._live.append(extent)
            return raw_entry
        return self._replace_value(tag, before_entry, after_entry)

    def _keep_unparsed_entry(
        self, tag: int, raw_entry: bytes, before: dict[str, Any], after: dict[str, Any]
    ) -> bytes:
        """tifftools skipped this entry (unknown datatype); we cannot size it."""
        if len(after["tags"]) < len(before["tags"]):
            raise InPlaceError(
                f"IFD at {before['offset']} has tag {tag} with an unknown datatype "
                "whose data cannot be located and zeroed; use a full rewrite"
            )
        return raw_entry

    def _replace_value(
        self, tag: int, before_entry: dict[str, Any], after_entry: dict[str, Any]
    ) -> bytes:
        """Place a changed value inline, in its old slot, or at end of file."""
        payload, count = encode_value(after_entry, self.fmt.bom)
        head = self.fmt.pack_entry_head(tag, int(after_entry["datatype"]), count)
        if len(payload) <= self.fmt.pointer_size:
            return head + payload.ljust(self.fmt.pointer_size, b"\x00")
        old_slot = tag_data_extent(before_entry)
        if old_slot is not None and len(payload) <= old_slot[1] - old_slot[0]:
            position = old_slot[0]
            self._tag_writes.append(Patch(position, payload))
        else:
            position = self._append(payload)
        self._live.append((position, position + len(payload)))
        return head + self.fmt.pack_pointer(position)

    def _before_extents(self) -> list[Range]:
        """Everything the original tree references: tables, tag data, chunks."""
        extents: list[Range] = []
        for ifd in walk_ifds(self.before["ifds"]):
            offset = int(ifd["offset"])
            extents.append((offset, offset + self.fmt.table_length(ifd["tagcount"])))
            for entry in ifd["tags"].values():
                extent = tag_data_extent(entry)
                if extent is not None:
                    extents.append(extent)
            extents.extend(chunk_extents(ifd))
        return extents


def collect_replacements(
    read: ReadRange,
    layout: Layout,
    work: dict[int, dict[int, ChunkWork]],
    fill: Rgb,
    use_dct: bool,
    report: RedactionReport,
) -> list[ChunkReplacement]:
    """Redact the planned chunks, keeping the new bytes in memory."""
    replacements: list[ChunkReplacement] = []

    def keep(level: ImageIFD, index: int, old: bytes, new: bytes) -> None:
        if new == old:  # already redacted (a repeated run); nothing to write
            return
        replacements.append(ChunkReplacement(level.offset, index, new, sha256_hex(old)))

    rewrite_chunks(read, layout, work, fill, use_dct, report, keep)
    return replacements


def redact_in_place(
    path: str,
    regions: Sequence[RegionInput],
    fill: Rgb = (0, 0, 0),
    drop_label: bool = False,
    drop_macro: bool = False,
    use_dct: bool = True,
    scrub: bool = False,
    keep_tags: Iterable[int] = (),
    manifest_path: str | None = None,
    overwrite_manifest: bool = False,
) -> RedactionReport:
    """Redact ``path`` itself, writing only the bytes that change.

    The manifest is saved before the file is touched, so an interrupted run
    can still be audited with :func:`tiffredact.verify.verify_in_place`.
    """
    manifest_path = manifest_path or default_manifest_path(path)
    if os.path.exists(manifest_path) and not overwrite_manifest:
        raise InPlaceError(
            f"Manifest {manifest_path} exists; pass overwrite to replace it"
        )
    before, layout, resolved, work = load_plan(
        path, regions, scrub, drop_label, drop_macro
    )
    after = copy.deepcopy(before)
    report = RedactionReport(path, path, resolved)
    kept_tags = sorted(set(keep_tags))
    with LocalFileStorage(path) as storage:
        replacements = collect_replacements(
            storage.read, layout, work, fill, use_dct, report
        )
        dropped = apply_metadata_changes(
            after, layout, report, drop_label, drop_macro, scrub, kept_tags
        )
        patch_set, records = plan_patches(storage, before, after, replacements)
        manifest = Manifest(
            path=path,
            source_size=patch_set.source_size,
            final_size=patch_set.final_size,
            regions=resolved,
            fill=fill,
            scrubbed=scrub,
            keep_tags=kept_tags,
            dropped_ifds=sorted(dropped),
            chunks=records,
        )
        manifest.record_patches(patch_set)
        manifest.save(manifest_path)
        storage.apply(patch_set)
    report.notes.append(
        f"in place: {patch_set.bytes_written} bytes written in "
        f"{len(patch_set.writes)} patches, {patch_set.bytes_zeroed} bytes zeroed, "
        f"file grew by {patch_set.bytes_appended} bytes"
    )
    report.notes.append(f"manifest: {manifest_path}")
    return report


def plan_patches(
    storage: Storage,
    before: dict[str, Any],
    after: dict[str, Any],
    replacements: list[ChunkReplacement],
) -> tuple[PatchSet, list[ChunkRecord]]:
    """Compute and validate the patch set that turns ``before`` into ``after``."""
    planner = PatchPlanner(before, after, storage.read, storage.size())
    records = planner.place_chunks(replacements)
    patch_set = planner.plan()
    patch_set.validate()
    return patch_set, records
