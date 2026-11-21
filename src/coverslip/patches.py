"""Byte-level edits that turn a slide into its redacted form without a rewrite.

A :class:`PatchSet` is storage-neutral: a local file applies it with seeks and
writes, an object store can assemble a new object from untouched ranges of the
old one plus the patched ranges.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Range = tuple[int, int]
"""Half-open byte range ``[start, end)``."""


class PatchError(Exception):
    """A patch set is inconsistent and must not be applied."""


@dataclass(frozen=True)
class Patch:
    """Overwrite (or, past the end of the source, append) bytes at ``offset``."""

    offset: int
    data: bytes

    @property
    def end(self) -> int:
        return self.offset + len(self.data)


@dataclass(frozen=True)
class Zero:
    """Overwrite ``length`` bytes at ``offset`` with zeros."""

    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class PatchSet:
    """Ordered edits against a file of ``source_size`` bytes.

    ``writes`` are applied in list order, then ``zeros``. The planner orders
    them so that an interrupted run leaves a readable file: appended data
    first, then the pointers to it, and the old bytes are zeroed last.
    """

    source_size: int
    writes: list[Patch] = field(default_factory=list)
    zeros: list[Zero] = field(default_factory=list)

    @property
    def final_size(self) -> int:
        return max([self.source_size, *(patch.end for patch in self.writes)])

    @property
    def bytes_appended(self) -> int:
        return self.final_size - self.source_size

    @property
    def bytes_written(self) -> int:
        return sum(len(patch.data) for patch in self.writes)

    @property
    def bytes_zeroed(self) -> int:
        return sum(zero.length for zero in self.zeros)

    def validate(self) -> None:
        """Refuse overlapping edits, holes past the end and out-of-file zeros."""
        self._check_no_overlap()
        self._check_appends_contiguous()

    def _check_no_overlap(self) -> None:
        ranges = [(patch.offset, patch.end) for patch in self.writes]
        ranges += [(zero.offset, zero.end) for zero in self.zeros]
        ranges.sort()
        for (_, previous_end), (start, _) in zip(ranges, ranges[1:]):
            if start < previous_end:
                raise PatchError(f"Edits overlap at offset {start}")
        if any(zero.end > self.source_size for zero in self.zeros):
            raise PatchError("A zeroed range extends past the end of the source")

    def _check_appends_contiguous(self) -> None:
        cursor = self.source_size
        for patch in sorted(self.writes, key=lambda p: p.offset):
            if patch.end <= self.source_size:
                continue
            if patch.offset != cursor:
                raise PatchError(
                    f"Appended data at {patch.offset} leaves a gap after {cursor}"
                )
            cursor = patch.end


def merge_ranges(ranges: Iterable[Range]) -> list[Range]:
    """Sort ranges and join the ones that touch or overlap; drop empty ones."""
    merged: list[Range] = []
    for start, end in sorted(r for r in ranges if r[1] > r[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_ranges(ranges: Iterable[Range], holes: Iterable[Range]) -> list[Range]:
    """Parts of ``ranges`` that no range in ``holes`` covers."""
    remaining: list[Range] = []
    cuts = merge_ranges(holes)
    first_cut = 0
    for start, end in merge_ranges(ranges):
        while first_cut < len(cuts) and cuts[first_cut][1] <= start:
            first_cut += 1
        cursor = start
        for cut_start, cut_end in cuts[first_cut:]:
            if cut_start >= end:
                break
            if cut_start > cursor:
                remaining.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < end:
            remaining.append((cursor, end))
    return remaining


def ranges_intersect(sorted_ranges: list[Range], candidate: Range) -> bool:
    """Whether ``candidate`` overlaps any of the merged, sorted ranges."""
    low, high = 0, len(sorted_ranges)
    while low < high:
        middle = (low + high) // 2
        if sorted_ranges[middle][1] <= candidate[0]:
            low = middle + 1
        else:
            high = middle
    return low < len(sorted_ranges) and sorted_ranges[low][0] < candidate[1]
