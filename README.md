# coverslip

Inspect, view, redact and scrub whole-slide images (tiled TIFF / BigTIFF,
Aperio SVS, OME-TIFF) from the command line.

* **Browse.** `coverslip slide.svs` opens the slide's layout as a table in
  the terminal. From there you open any image in a window and flag the label
  and macro for removal, and coverslip prints the command that does it.
* **Inspect and view.** Coverslip lists the pyramid levels and associated
  images, opens any level in a window, and opens single tiles of a large level
  by clicking on an overview. A heatmap of stored bytes per tile shows tissue,
  background and empty tiles at a glance.
* **Draw regions.** Drag rectangles on a level to get the exact `-r` arguments
  `redact` takes, or preview regions you already have.
* **Redact.** Coverslip blacks out rectangular regions by rewriting **only the
  tiles that intersect them**. Every other tile is copied byte-for-byte, so the
  rest of the slide takes no generation loss and a multi-gigabyte slide is
  processed in seconds. You can work on a copy or modify the file in place.
* **Scrub.** Coverslip strips identifying metadata and keeps the slide
  readable.

Coverslip was formed from two earlier projects, tiffredact and tiffview.

## Install

```sh
uv sync                       # everything, including the viewer and test tools
uv pip install .              # headless: inspect, redact, scrub, verify
uv pip install '.[view]'      # plus the viewer (show, tile, grid)
uv pip install '.[view,tui]'  # plus the terminal browser
```

Runtime dependencies: `tifffile`, `imagecodecs`, `jpeglib`, `tifftools`,
`numpy`. The viewer also needs `matplotlib`, which is the optional `view`
extra. Without it, `show`, `tile` and `grid` exit with a hint to install it,
and everything else still works. The browser needs `textual`, the optional
`tui` extra, and exits with the same kind of hint without it. OpenSlide (`openslide-python` +
`openslide-bin`) is a dev dependency, used only by the test suite.

## Browsing

```sh
uv run coverslip slide.svs           # short for: coverslip browse slide.svs
```

The browser shows the `inspect` table and lets you act on its rows:

| key | action |
|---|---|
| up / down, `j` / `k` | move between images |
| Enter | open the image in a `show` window; the browser waits until you close it |
| `d` | flag or unflag a label or macro image for dropping |
| `s` | turn metadata scrubbing on or off |
| `p` | go back to the default profile |
| `u` | forget the last region drawn |
| `q` | quit and print the command |

**Default profile.** The browser starts with metadata scrubbing on and, for an
Aperio SVS, the label and macro flagged. The label usually shows the accession
number and the macro photographs the whole slide, label included. Unflag
either with `d`.

**It only prints a command.** The browser never modifies a file. The line at
the bottom shows the command your choices amount to, and `q` prints it:

```
redact with:
coverslip scrub slide.svs OUTPUT --drop-label --drop-macro
```

Regions you draw in a window opened from the browser (press `r` there, see
below) are kept, previewed in the next window you open, and added to the
command, which then becomes `coverslip redact ... -r ...`.

Limits:
- Only label and macro images can be dropped. Pyramid levels, thumbnail
  included, cannot, and neither can any image of an OME-TIFF.
- With scrubbing off and nothing flagged or drawn, there is no command to
  print.
- Without a terminal (output piped or redirected), `coverslip slide.svs`
  prints the `inspect` report instead.
- Opening an image needs the `view` extra as well.

## Inspecting

```sh
uv run coverslip inspect slide.svs          # levels and associated images
uv run coverslip inspect slide.svs --tags   # every tag, and what scrubbing would do
```

`inspect` prints one row for each pyramid level and each associated image
(label, macro, thumbnail). Each row gives:
- the size
- the tile (or strip) size and the tile grid
- the number of empty tiles (zero bytes stored)
- compression and photometric interpretation
- samples × bits
- the downsample factor from full resolution
- the bytes stored for that image
- the IFD offset

Level numbers in this table are the ones the viewer takes: `0` is full
resolution, and higher numbers are smaller.

## Viewing

```sh
# open an overview; click it to open full-resolution tiles
uv run coverslip show slide.svs

# a specific level, with clicks opening tiles of level 1 instead
uv run coverslip show slide.svs --level 3 --tile-level 1

# start with the tile byte-count map overlaid
uv run coverslip show slide.svs --grid

# associated images
uv run coverslip show slide.svs --label     # also --macro, --thumbnail

# one tile, by grid position or by a point in that level's pixels
uv run coverslip tile slide.svs --col 120 --row 45
uv run coverslip tile slide.svs --level 1 --at 9000,6100

# the byte-count map of a level on its own; click a cell to open that tile
uv run coverslip grid slide.svs --level 0
```

**Default level.** Without `--level`, `show` picks the smallest tiled level
whose long side is at least 1000 px and that fits within the pixel limit. It
uses a stripped thumbnail only when no tiled level qualifies.

**Pixel limit.** Levels over 8192 × 8192 pixels are refused unless you pass
`--force`. Assembling a full-resolution level can take many gigabytes of
memory.

In the `show` window:

| input | action |
|---|---|
| hover | level coordinates, plus the tile under the cursor (at `--tile-level`) and its stored bytes |
| left click | open that tile in a new window; opened tiles are outlined in yellow |
| `g` | toggle the byte-count map; empty tiles are red |
| `r` | toggle region drawing (see below) |
| `u` | undo the last region drawn |
| toolbar | pan and zoom; clicks don't open tiles while a tool is active |

**`tile` window.** The window shows one tile, cropped to the image bounds.
The title gives the tile's grid position, its pixel position on the level,
its file offset and its stored size.

**`grid` window.** The window shows the byte-count map as a heatmap. Clicking
a cell opens that tile.

### Drawing redaction regions

Press `r` in the `show` window, and dragging then draws rectangles instead of
opening tiles. Each rectangle is printed the moment you draw it, as a `-r`
argument in the shown level's own pixels:

```
-r 310,250,190,125@2875x2055
```

When you close the window, coverslip prints a complete command:

```
redact with:
coverslip redact slide.svs OUTPUT -r 310,250,190,125@2875x2055
```

The rectangle includes every pixel it partly covers. Because a stored
level's own pixels are exactly what `redact` accepts, nothing is lost between
drawing and redacting.

To check regions before redacting, pass them to `show`. The viewer draws them
dashed, in the area `redact` would actually paint on the shown level,
including its outward rounding:

```sh
uv run coverslip show slide.svs -r 5000,4000,3000,2000 -r 310,250,190,125@2875x2055
```

Regions you draw in that window are added to the printed command, after the
ones you passed in.

## Redacting

```sh
# black out two rectangles (x,y,width,height at level 0), drop label and macro
uv run coverslip redact slide.svs redacted.svs \
    -r 5000,4000,3000,2000 -r 100,100,500,900 --drop-label --drop-macro

# audit the result against the source
uv run coverslip verify slide.svs redacted.svs -r 5000,4000,3000,2000 -r 100,100,500,900

# a rectangle drawn on the 2875x2055 level (dimensions as printed by inspect)
uv run coverslip redact slide.svs redacted.svs -r 310,250,190,125@2875x2055

# metadata only: strip identifying tags and description fields, keep pixels
uv run coverslip scrub slide.svs clean.svs --drop-label --drop-macro
uv run coverslip verify slide.svs clean.svs --scrubbed

# both at once
uv run coverslip redact slide.svs clean.svs -r 5000,4000,3000,2000 --scrub

# only remove the label and macro images; pixels and metadata stay
uv run coverslip redact slide.svs nolabel.svs --drop-label --drop-macro
uv run coverslip verify slide.svs nolabel.svs
```

Options for `redact`:

| flag | meaning |
|---|---|
| `-r/--region X,Y,W,H[@WxH]` | rectangle at full resolution, or on the level that is `WxH` pixels; repeatable |
| `--fill R,G,B` | fill colour, default black |
| `--drop-label` / `--drop-macro` | remove the Aperio label / macro IFDs; enough on their own, without a region |
| `--no-dct` | re-encode JPEG chunks in the pixel domain instead of editing DCT blocks |
| `--overwrite` | replace an existing output file |
| `--scrub` | also scrub identifying metadata (see below) |
| `--keep-tag TAG` | exempt a TIFF tag (number or name) from scrubbing; repeatable |

A level is named by its pixel dimensions because those are the same however
the file lays out its pyramid (main IFD chain or SubIFDs). They must match a
stored level exactly; anything else is an error. The rectangle is scaled up to
full resolution with outward rounding and then applied to every level, so on
the other levels it can grow by up to one pixel of the level it was drawn on.
Reports and in-place manifests record the full-resolution rectangle.

A run that cannot be carried out prints one `coverslip <command>: error: ...`
line and exits with status 2. Examples:
- an unknown level
- an unreadable or unsupported file
- an existing output
- a level over the viewer's pixel limit

Add `-v` for the traceback. When a clicked tile cannot be opened (an empty
tile or an unsupported codec), the viewer logs a warning and keeps running.

`verify` exits with status 1 if any pixel in the mapped region on any level is not
the fill colour, if any untouched chunk differs from the source, or if the bytes
of a replaced source chunk can still be found in the output. With `--scrubbed`
it also fails if any non-allowlisted tag or identifying description field is
still present.

## In-place mode

`--in-place` (also spelled `--inplace`) modifies the input file itself instead
of writing a new one, touching only the bytes that change. It exists for
slides where a full rewrite is expensive, such as files behind a range-aware
network mount.

```sh
# redact and scrub slide.svs itself; no output argument
uv run coverslip redact slide.svs --in-place \
    -r 5000,4000,3000,2000 --drop-label --drop-macro --scrub
uv run coverslip scrub slide.svs --in-place

# audit it; regions, fill and scrub settings come from the manifest
uv run coverslip verify slide.svs --in-place
```

**There is no source left afterwards.** Keep a copy, or storage-level
versioning, until the audit has passed.

* A manifest is written next to the file before it is modified
  (`slide.svs.coverslip-manifest.json`, or `--manifest PATH`). It records the
  regions and settings, every byte range written or zeroed, and where each
  replaced chunk used to be. `verify --in-place` needs it. An existing
  manifest is not replaced without `--overwrite`.
* A replaced chunk is written over its old bytes when it fits; otherwise it is
  appended to the end of the file and its old bytes are zeroed. Removed tags,
  dropped label/macro IFDs with their strips, and the tails of shortened
  descriptions are zeroed as well, so nothing redacted remains in the file.
  The file never shrinks: it keeps zero-filled holes and may grow by the
  appended chunks.
* Pixels and tags are the same as the full rewrite produces; only the layout
  of the file differs. Next-IFD links between sub-IFDs are left as they are.
* `verify --in-place` reads back every written range, checks every zeroed
  range is zero, checks the old position of each replaced chunk no longer
  holds its bytes, re-parses the IFDs, decodes the chunks under each region
  and, if the run scrubbed, checks the metadata. It reads only those ranges,
  not the whole file.
* A classic TIFF cannot grow past 4 GB. If appending would cross that, the
  run stops before writing anything; use the full rewrite, which upgrades to
  BigTIFF. NDPI is refused, as is scrubbing a file with old-style JPEG tags
  (`JPEGInterchangeFormat`) or tags of an unknown datatype.
* Writes are ordered so that an interrupted run leaves a readable file:
  appended data first, then the pointers to it, zeroing last. Readable is not
  redacted: until zeroing has finished the old bytes can still be present,
  and `verify --in-place` with the manifest of that run reports it. There is
  no resume. Running again does not repair it, because the old bytes are no
  longer referenced by the file and a new run cannot find them; restore the
  copy and redact again.

## Metadata scrubbing

`scrub` (or `redact --scrub`) removes identifying metadata while keeping the
file readable by OpenSlide, Bio-Formats and tifffile:

* **TIFF tags** survive only if they are on a structural allowlist (image
  geometry, compression, colour, ICC profile, Software, Make/Model). Everything
  else is removed: DateTime, Artist, HostComputer, DocumentName, XMP, IPTC,
  EXIF/GPS sub-IFDs, ImageJ metadata, vendor private tags. `--keep-tag`
  exempts a tag; `inspect --tags` previews the decision for every tag.
* **Aperio `ImageDescription`**: the header line (dimensions, compression) is
  kept and `Key = Value` fields are filtered by an allowlist. `AppMag`, `MPP`,
  `StripeWidth`, stage offsets, exposure and ICC fields stay; `Filename`,
  `ImageID`, `Date`, `Time`, `Time Zone`, `User`, `ScanScope ID`, `DSR ID`
  and any unknown key are removed. OpenSlide still reports `mpp` and objective
  power.
* **OME-XML**: `Experimenter*`, `AcquisitionDate`, `Description`,
  `StructuredAnnotations` (vendor "original metadata"), `AnnotationRef`,
  `Project`/`Dataset`/`Folder`/`Plate`/`Screen`, `ROI`, `SerialNumber` and
  `LotNumber` attributes and `Image`/`Instrument` names are removed. `Pixels`,
  `Channel`, `TiffData` and the file UUID stay, so the pyramid and multi-file
  references remain valid.
* **Other descriptions** are removed, except tifffile's `{"shape": ...}` JSON.

Scrubbing does not touch pixel data; label and macro images still need
`--drop-label`/`--drop-macro` (SVS) or pixel redaction.

## How it works

1. **Layout discovery** (`layout.py`): the IFD tree is read with `tifftools`.
   Aperio files are classified by position and `NewSubfileType` (label = 1,
   macro = 9); OME-TIFF and generic pyramids are discovered through `SubIFDs`
   or reduced-resolution IFDs with matching aspect ratio. Every reduced copy of
   the main image, including a stripped thumbnail, is a level to redact.
2. **Region mapping** (`geometry.py`): per-axis downsample factors (vendor
   pyramids are not exact powers of two), rounded outward, then intersected
   with the tile/strip grid.
3. **Chunk rewriting** (`codecs/`):
   * **JPEG** chunks are edited in the DCT domain with `jpeglib`: the quantized
     coefficients of the blocks inside the rectangle are replaced by a flat
     fill, everything else is re-entropy-coded unchanged. The rectangle is
     snapped outward to whole MCUs (8 px, or 16 px with chroma subsampling).
     The rewritten chunk embeds its own tables and still decodes with the
     shared `JPEGTables` prepended, which is what OpenSlide, libtiff and
     tifffile do.
   * **LZW, Deflate, PackBits, ZSTD, uncompressed** chunks are decoded, painted
     and re-encoded; lossless, so unaffected pixels are exact.
   * **JPEG 2000** (Aperio 33003/33005, 34712) chunks are decoded with
     OpenJPEG, painted and re-encoded losslessly.
4. **Writing** (`writer.py`): `tifftools.write_tiff` rebuilds the file, copying
   untouched chunk bytes from the source and replaced chunks from a scratch
   file through a virtual address space. The output has no orphaned bytes, all
   tags (ICC, OME-XML, vendor description, unknown tags) are preserved, and
   classic/BigTIFF is kept as in the input.
5. **Viewing** (`chunks.py`, `viewer.py`): tiles are read and decoded one at a
   time with the same codecs, converting YCbCr JPEG to RGB for display, and a
   level is assembled from its tiles. Decoded tiles match OpenSlide on the
   Aperio JPEG, Aperio JPEG 2000 and generic YCbCr samples. Regions drawn
   in the viewer (`regions.py`) go through the same geometry as `redact`.

## Guarantees and limits

* Untouched chunks are byte-identical to the source.
* Inside a rewritten JPEG chunk, the DCT coefficients of every block outside
  the snapped rectangle are unchanged. With chroma-subsampled JPEG the decoder's
  chroma upsampling blends a 1-2 px band on either side of the painted border;
  luma is exact.
* Over-redaction of up to one MCU (15 px) around each rectangle for JPEG.
* `PlanarConfiguration=2` (separate planes, as written by some OME-TIFF
  exporters) is supported; each plane's chunks are rewritten with that plane's
  fill value.
* OME-TIFF files may carry macro/label/overview images in their own
  coordinate space. They are detected from the OME-XML image names and left
  untouched. `--drop-label`/`--drop-macro` are refused for OME-TIFF because
  the OME-XML references IFDs by index.
* Not supported yet: sub-byte samples, CMYK/YCCK JPEG, NDPI/MRXS/DICOM. SVS
  macro images cannot be mapped from slide coordinates; drop them with
  `--drop-macro`.

### Reader notes

* Bio-Formats-written OME-TIFFs link each pyramid level both through the
  `SubIFDs` tag and through the previous level's next-IFD pointer. The output
  keeps the `SubIFDs` tag only (logged at INFO level); Bio-Formats, tifffile
  and OpenSlide read the result exactly as they read the source.
* The ImageJ Bio-Formats importer cannot open any plane with more than
  2^31 pixels (for example 44961 x 52939) and fails with a
  `NegativeArraySizeException` in `MinMaxCalculator.openBytes`. This happens
  on the source file as well; import a lower resolution or use "Crop on
  import". Bio-Formats itself refuses planes above that size with
  "ImageWidth x ImageLength > 2147483647 is not supported".
* OpenSlide's generic TIFF driver ignores `SubIFDs`, so it reports a single
  level for these OME-TIFFs, before and after redaction.

## Verified on

| file | structure | chunks rewritten | time |
|---|---|---|---|
| Aperio SVS, 75 MB | JPEG RGB 4:4:4, classic TIFF, stripped thumbnail | 171 | 0.4 s |
| Aperio SVS, 1.8 GB | JPEG RGB 4:4:4, classic TIFF, 4 levels | 2435 | 6 s |
| CMU-1-JP2K-33005.svs, 133 MB | Aperio JPEG 2000 | 172 | 1 s |
| CMU-1.tiff, 204 MB | generic YCbCr 4:2:0 JPEG, 9-level IFD chain | 400 | 0.5 s |
| Leica-1.ome.tiff, 723 MB | OME BigTIFF, planar JPEG, SubIFD pyramid + macro | 1314 | 2 s |
| Zeiss Axio Scan OME-TIFF (Bio-Formats 8.5), 3.9 GB | OME BigTIFF, YCbCr JPEG, 9-level SubIFD pyramid | 27 | 1 s |

All outputs pass `coverslip verify`; untouched areas were also confirmed
identical with OpenSlide and tifffile. The Zeiss output was also read with
Bio-Formats 8.5 (`showinf`, `bfconvert -crop`) with the same core metadata
as the source.

## Tests

```sh
uv run pytest
```

Synthetic pyramids are generated with `tifffile`. Tests on real Aperio files run
when `*.svs` files are present in the repository root, and some tests compare
decoded tiles with OpenSlide on slides in `samples/`. Either kind is skipped
when its files are missing.

Viewer tests run headless on matplotlib's Agg backend. They drive the windows
with synthetic clicks, key presses and drags.
