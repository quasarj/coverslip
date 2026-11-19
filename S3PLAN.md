# S3PLAN — in-place redaction for S3-resident slides

Status: idea, not started. Written 2026-11-18 after the metadata scrubbing
milestone. The current tool (see `PLAN.md`) is complete and verified; this
plan describes a second write backend for slides that live in S3 and are
mounted locally through a range-aware tool, where a full rewrite means a full
download plus a full upload.

## Motivation

Today's pipeline, per file:

| phase | bytes read | bytes written |
|---|---|---|
| discover layout (`tifftools.read_tiff`) | IFD tables + tag arrays, a few MB | 0 |
| rewrite touched chunks | only the touched tiles/strips | scratch file, small |
| write output (`tifftools.write_tiff`) | **entire source** | **entire output** |

The read side is already range-friendly. The write side streams every byte,
so over an S3 mount a 2 GB slide costs 2 GB down and 2 GB up even for a
single redacted tile. An in-place writer would make both directions a few
megabytes.

## Design: compute patches, then apply them

Split the work into two layers so the storage backend is a detail.

### 1. Patch computation (`inplace.py`)

Reuse everything up to and including chunk rewriting (`layout`, `geometry`,
`codecs`, `scrub`). Instead of handing the modified IFD tree to tifftools,
compute a **patch set**:

```
Patch(offset: int, data: bytes)          # overwrite bytes at offset
Zero(offset: int, length: int)           # overwrite with zeros
Append(data: bytes) -> offset            # extend the file; offset resolved at apply time
```

tifftools already records, for every tag, `datapos` (the entry's value field
inside the IFD) and `offset` (where out-of-line data lives), so every write
position is known without a second parser.

Operations, in the order that keeps the file readable if interrupted:

1. **Append first.** For each replaced chunk whose new bytes exceed the old
   byte count, append the new bytes at end of file and record the new offset.
   Chunks that fit are written over the old bytes in place, with the tail of
   the old slot zeroed.
2. **Patch pointer arrays.** Rewrite `TileOffsets`/`TileByteCounts` (or the
   strip equivalents) in place. They are fixed-width LONG/LONG8 arrays, so
   their size never changes.
3. **Metadata.** Removing tags shrinks an IFD: rewrite the entry table at the
   same offset with fewer entries, rewrite the next-IFD pointer immediately
   after it, zero the leftover bytes. Rewritten `ImageDescription` values
   (Aperio key filtering, OME-XML element removal) are always no longer than
   the original, so they fit in the old slot; zero the remainder and patch
   the count field. Inline values (≤ 4 / 8 bytes) are patched in the entry.
4. **Dropped IFDs** (label, macro): patch the previous IFD's next pointer to
   skip them, then zero the dropped IFD table, its tag data and all its strip
   bytes.
5. **Zero every orphaned range**: old slots of chunks that moved to the end,
   data of removed tags (XMP, EXIF/GPS sub-IFDs, ImageJ metadata), dropped
   IFDs. This is the step libtiff's own update mode skips and the reason the
   current tool rewrites the whole file; it must not be optional.

Output: an ordered list of patches plus a manifest (source size, source
IFD offsets, patched ranges, zeroed ranges, replaced-chunk hashes) that the
verifier consumes.

### 2. Patch application (backends)

* **Local file** (`--in-place`): `open(path, "r+b")`, apply in order.
  Works on a range-aware FUSE mount if the mount turns partial writes into
  server-side splices.
* **S3 direct** (`--s3 bucket/key`, no mount needed): assemble a new object
  with multipart upload using `UploadPartCopy` for untouched ranges of the
  existing object and `UploadPart` for patched ranges. Parts must be ≥ 5 MB
  except the last, so adjacent untouched ranges are merged and small patches
  are carried inside a part that also copies neighbouring bytes (read those
  neighbours with a ranged GET). Completing the upload replaces the object
  atomically; with bucket versioning the previous version is the rollback.
* **Patch file** (`--emit-patch out.json`): write the patch set for a
  separate applier, useful for auditing what will change before touching
  storage.

### 3. Range-aware verification

The current `verify` compares output to source chunk by chunk, which is a
full read. For in-place mode:

* read back every patched range and decode replaced chunks; check the
  region is the fill colour,
* read back every zeroed range and confirm it is zero,
* re-parse the IFD chain and SubIFDs and check consistency (counts, offsets
  within file, no pointer into a zeroed range),
* for `--scrubbed`, run the existing `scrub_problems` on the re-parsed IFDs,
* residue: the old bytes of replaced chunks are known from the manifest; the
  only places they could survive are their old slots, which were zeroed and
  are read back above. No whole-file search is needed.

Untouched bytes were never written, so they are not re-read. Traffic for the
audit is about the same as for the redaction.

## Constraints and caveats

* **Classic TIFF cannot grow past 4 GB.** Appends are refused when the
  resulting offset would not fit in 32 bits; fall back to the full rewrite,
  which upgrades to BigTIFF. (Both user samples are classic TIFF, including
  the 1.8 GB one.)
* **Partially covered JPEG tiles grow** by roughly 600 bytes because the
  rewritten tile embeds its own DQT/DHT tables. Those tiles always move to
  the end of the file. Fully covered tiles shrink. Lossless codecs shrink.
* **Interrupted runs** leave a file that is still readable after step 1 and
  2 but may contain residue until step 5 completes. On S3 direct this does
  not apply: the object changes only when the multipart upload completes.
* **No source for comparison** after an in-place run. The manifest replaces
  it; keep it next to the file (or in object metadata) until the audit has
  passed.
* **Mount semantics vary.** mountpoint-s3 does not support modifying existing
  objects; s3fs downloads the whole object to a cache before writing; only
  tools that implement range updates through server-side copy avoid the
  round trip. The S3-direct backend removes this dependency.
* **Multipart copy limits**: 10,000 parts per upload, 5 MB minimum part size
  (except the last), 5 GB maximum per `UploadPartCopy` range. Objects over
  50 GB need ≥ 5 MB average parts, which is natural here.

## Effort

Roughly 300–400 lines for `inplace.py` (patch computation + local applier),
~150 for the S3 applier (boto3, multipart copy with part planning), ~150 for
the range-aware verifier, plus tests that run the local backend against the
synthetic pyramids and the sample slides and compare the result byte-for-byte
with the full-rewrite output (same tiles, same metadata; only layout differs).

## Open questions

* Where does the manifest live for S3 objects: sidecar object, object
  metadata, or a database the batch driver already keeps?
* Should `--in-place` on a plain local file be allowed, or only the S3
  backend? Locally the full rewrite is fast and strictly safer.
* Part planning for the S3 backend: simplest is "one part per contiguous
  run, pad small patches with copied neighbours"; check it against the
  10,000-part limit for slides with thousands of scattered redacted tiles
  (a 20 GB slide with 3,000 scattered tiles is ~6,000 parts, close to the
  limit; merging runs that are < 5 MB apart fixes it).
