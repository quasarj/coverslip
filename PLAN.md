# tiffredact — plan

## Goal

Redact rectangular regions of the pixel data in a whole-slide image (WSI)
stored as tiled TIFF / BigTIFF, Aperio SVS, or OME-TIFF, by painting the
region a solid colour (black by default) on **every** resolution level of the
pyramid, while touching **only** the tiles/strips that intersect the region.
All other compressed tile bytes are copied verbatim, so the rest of the image
takes no generation loss and the run time is dominated by one sequential copy
of the file.

Coordinates are given on the full-resolution level; the tool maps them to each
lower-resolution level itself.

Non-goals for now: metadata / tag scrubbing (may come later), NDPI / MRXS /
DICOM, automatic PHI detection.

## Prior art (why we are building this)

| Tool | What it does | Why it does not fit |
|---|---|---|
| DSA-WSI-DeID (Kitware) | Redacts polygons on the main image of SVS/NDPI/Philips/OME-TIFF | Decodes the *whole* slide into pyvips and re-encodes the whole pyramid |
| John Snow Labs Visual NLP | Per-tile SVS redaction, scaled across levels | Commercial, closed |
| wsi-anon, anonymize-slide, paw | Metadata and label/macro wiping | No pixel redaction |
| `jpegtran -wipe` | Lossless DCT-domain rectangle wipe on a single JPEG | Single file, grey fill; but it is the right primitive per tile |
| tifftools (Kitware) | Pure-Python IFD rewrite, copies image data opaquely, no abandoned bytes | Does not touch pixels; we use it as the writer |

## Facts that drive the design

* **TIFF tiles/strips are independent codestreams.** Replacing one only requires
  patching its entry in `TileOffsets`/`TileByteCounts` (or the strip
  equivalents).
* **JPEG tiles can be edited losslessly for the untouched blocks.** Read the
  quantized DCT coefficients, overwrite the coefficients of the blocks inside
  the region with the fill colour (for black in YCbCr: Y DC = −1024/q₀, all AC
  = 0, Cb/Cr DC = 0), and re-entropy-code. Granularity is one MCU (8 px at
  4:4:4, 16 px at 4:2:0); we snap the rectangle **outward**. Over-redaction
  by up to 15 px is accepted.
* **Aperio SVS layout** (confirmed on the two sample files, both classic TIFF
  despite one being 1.8 GB):
  * IFD 0: full-res, tiled 240×240, JPEG, `ImageDescription` starts with
    `Aperio`, per-IFD `JPEGTables`.
  * IFD 1: thumbnail, **stripped** (16 rows/strip), JPEG. Must be redacted too.
  * IFD 2..n: pyramid levels, tiled, same compression. Downsamples are not
    exact powers of two (e.g. 4.0004, 8.0015).
  * Label: `NewSubfileType=1`, LZW strips. Macro: `NewSubfileType=9`, JPEG
    strips. Both optional; removable via CLI flag.
  * Tags say `Photometric=RGB`, `YCbCrSubsampling=2,2`, but the codestream in
    the samples is 3-component 4:4:4 with component ids 0,1,2 and no
    JFIF/Adobe marker, which libjpeg treats as YCbCr. All three components use
    quant table 0 and the standard *luminance* Huffman tables (DC+AC id 0), so
    any coefficient value is encodable with the shared tables.
  * Newer scanners emit JPEG 2000 (compression 33003/33005). No DCT trick
    there; decode → paint → re-encode the affected tiles with OpenJPEG.
* **OME-TIFF pyramids** live in `SubIFDs` (tag 330) of each full-res plane,
  largest to smallest, `NewSubfileType` bit 0 set. OME-XML is in IFD 0's
  `ImageDescription` and must be preserved byte-for-byte. A region must be
  applied to every plane (channel / z / t) of the image.
* **Generic pyramidal TIFF** uses either SubIFDs or a chain of
  reduced-resolution IFDs. Levels are recognised by matching aspect ratio.
* **libtiff's in-place rewrite** (`TIFFAppendToStrip`) overwrites in place when
  the new tile is ≤ the old size and otherwise appends at EOF, leaving the old
  bytes orphaned. Orphaned bytes are a redaction leak. Our default is therefore
  a full streaming rewrite with no abandoned data; an in-place mode, if added,
  must explicitly zero stale bytes.
* **Sizes almost always shrink** (black blocks entropy-code to a few bytes;
  constant data compresses to nothing under LZW/Deflate). The one growth case
  is embedding DQT/DHT in a redacted JPEG tile (~600 B) when the original
  relied on the shared `JPEGTables`. We accept this; the writer handles any
  size.

## Language and stack

Python 3.14, managed with `uv`. The heavy lifting is all C codecs with
mature Python bindings, and the TIFF plumbing exists already.

| Need | Library |
|---|---|
| Parse classic/BigTIFF, SubIFDs, tile offsets, `JPEGTables`, series/levels | `tifffile` |
| Decode/encode JPEG (with external tables), JPEG 2000, LZW, Deflate, PackBits | `imagecodecs` |
| Lossless DCT-domain edit of JPEG tiles | `jpeglib` (libjpeg `jpeg_read/write_coefficients`) |
| Rewrite the file: new IFDs, untouched data copied opaquely, no abandoned bytes | `tifftools` |
| Verification | `openslide-python` (dev dependency), `tiffinfo` |
| CLI | `argparse` (stdlib) |

## Architecture

```
src/tiffredact/
  model.py        dataclasses: Chunk (tile or strip geometry), Level, Image, SlideLayout
  layout.py       discover levels / thumbnail / label / macro / SubIFD pyramids (SVS, OME, generic)
  geometry.py     region mapping between levels, chunk intersection, MCU snapping
  codecs/
    jpeg_dct.py   lossless DCT-domain redaction via jpeglib (JPEGTables handling)
    pixel.py      decode → paint → re-encode for LZW / Deflate / PackBits / none / JPEG 2000
    dispatch.py   choose codec path per chunk, solid-fill tile cache
  writer.py       build new IFD set; virtual source file = original ∪ scratch; tifftools.write_tiff
  redact.py       orchestration: layout → regions → chunks → codec → writer
  cli.py          `tiffredact redact`, `tiffredact inspect`, `tiffredact verify`
tests/            mirrors src; synthetic TIFFs generated with tifffile; SVS tests skipped if samples absent
```

### Write-out mechanism

`tifftools.write_tiff` copies each IFD's offset data from `ifd['path_or_fobj']`
by seeking to `TileOffsets[i]` and reading `TileByteCounts[i]`. We exploit
this with a **virtual source file**: offsets below the original file size read
from the original; offsets at or above it read from a scratch file that holds
the replacement chunks. For every redacted chunk we set
`TileOffsets[i] = original_size + scratch_offset` and
`TileByteCounts[i] = len(new_bytes)`. tifftools then produces a clean file
with no orphaned data and all tags (including unknown ones, `JPEGTables`,
ICC, OME-XML) preserved.

Label and macro removal: drop those IFDs from the list before writing.

## Milestones

1. **Scaffold + inspection** — uv project, `inspect` command printing the
   discovered layout for SVS. *(commit)*
2. **End-to-end with pixel-domain re-encode** — region mapping, chunk
   selection, decode/paint/re-encode for every codec incl. JPEG (via
   imagecodecs with `JPEGTables`), writer, `redact` command. Verify with
   OpenSlide that the region is black on every level and the thumbnail.
   *(commit)*
3. **Lossless DCT path for JPEG** — jpeglib-based block wipe; property test
   that untouched blocks decode bit-identically to the original. *(commit)*
4. **Label/macro removal flags**, `verify` command (region all-black on every
   level, untouched chunk bytes identical, no residue of old tile bytes).
   *(commit)*
5. **OME-TIFF and generic pyramidal TIFF** — SubIFD pyramids, multi-plane
   application, synthetic test fixtures. *(commit)*
6. **BigTIFF and JPEG 2000 coverage**, performance pass (parallel chunk
   processing). *(commit)*

## Verification checklist

* Open output with OpenSlide; `read_region` over the mapped rectangle at every
  level is entirely the fill colour.
* For every chunk not intersecting a region, output bytes == input bytes.
* `tiffinfo -D` and `tifftools dump` succeed; tag sets match the input minus
  any dropped IFDs.
* No byte sequence from a replaced original tile is present in the output.
* OME-XML / `ImageDescription` unchanged.

## Risks / open items

* `jpeglib` requires file paths; per-tile temp files are acceptable initially.
* JPEG tiles whose shared Huffman tables are *optimized* (not standard) may
  lack symbols needed for the fill blocks; mitigated by always writing full
  DQT/DHT into redacted tiles.
* Fill colours other than black need a YCbCr conversion for the DCT path.
* Macro images cannot be mapped from full-res coordinates; they can only be
  dropped (flag) or, later, redacted with their own coordinates.
