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

Status as of 2026-11-18:

1. **Scaffold + inspection** — done. `tiffredact inspect` prints the discovered
   layout for SVS, OME-TIFF and generic pyramids.
2. **End-to-end with pixel-domain re-encode** — done (`--no-dct` path and all
   lossless codecs).
3. **Lossless DCT path for JPEG** — done; default for 8-bit JPEG chunks. Rects
   are snapped outward to whole MCUs so every component is painted over the
   same area.
4. **Label/macro removal flags, `verify` command** — done.
5. **OME-TIFF and generic pyramidal TIFF** — done for SubIFD pyramids,
   reduced-IFD chains and multi-plane OME files (synthetic fixtures); a public
   Leica OME-TIFF sample is used as a real-file check.
6. **BigTIFF and JPEG 2000** — done (synthetic BigTIFF fixtures; Aperio
   33005 sample verified with OpenSlide). No parallelism was needed: the
   1.8 GB sample slide takes ~6 s for ~2400 tiles.

Results on real files (all verified with `tiffredact verify`, OpenSlide and
tifffile; untouched areas byte/pixel identical):

| file | structure | chunks rewritten | time |
|---|---|---|---|
| C3L-00977-22.svs (75 MB) | Aperio JPEG RGB 4:4:4, classic TIFF | 171 | 0.4 s |
| C3N-04611-21.svs (1.8 GB) | Aperio JPEG RGB 4:4:4, classic TIFF | 2435 | 6 s |
| CMU-1-JP2K-33005.svs (133 MB) | Aperio JPEG 2000 | 172 | 1 s |
| CMU-1.tiff (204 MB) | generic YCbCr JPEG, 9-level IFD chain | 400 | 0.5 s |

Possible follow-ups: metadata scrubbing, `PlanarConfiguration=2`, NDPI,
an in-place mode that zeroes stale bytes, redacting the macro image by its
own coordinates.

## Verification checklist

* Open output with OpenSlide; `read_region` over the mapped rectangle at every
  level is entirely the fill colour.
* For every chunk not intersecting a region, output bytes == input bytes.
* `tiffinfo -D` and `tifftools dump` succeed; tag sets match the input minus
  any dropped IFDs.
* No byte sequence from a replaced original tile is present in the output.
* OME-XML / `ImageDescription` unchanged.

## Risks / open items

* `jpeglib` requires file paths; per-tile temp files are used and proved fast
  enough.
* Redacted JPEG tiles always embed their own DQT/DHT, so optimized shared
  Huffman tables cannot break them. The JFIF APP0 that libjpeg adds is
  stripped so the TIFF Photometric tag stays the only colour-space authority.
* With chroma-subsampled JPEG, decoder chroma upsampling blends a 1-2 px band
  on either side of the painted border. Luma and all coefficients outside the
  snapped rectangle are exact.
* Macro images cannot be mapped from full-res coordinates; they can only be
  dropped (flag) or, later, redacted with their own coordinates.
