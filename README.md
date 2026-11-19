# tiffredact

Redact rectangular pixel regions in whole-slide images (tiled TIFF / BigTIFF,
Aperio SVS, OME-TIFF) by rewriting **only the tiles that intersect the
region**. Every other tile is copied byte-for-byte, so the rest of the slide
takes no generation loss and a multi-gigabyte slide is processed in seconds.

Regions are given in full-resolution pixel coordinates; the tool maps them onto
every pyramid level and onto the SVS thumbnail itself.

## Install

```sh
uv sync
```

Runtime dependencies: `tifffile`, `imagecodecs`, `jpeglib`, `tifftools`,
`numpy`. OpenSlide (`openslide-python` + `openslide-bin`) is a dev dependency
used only by the test suite.

## Usage

```sh
# what is in the file?
uv run tiffredact inspect slide.svs

# black out two rectangles (x,y,width,height at level 0), drop label and macro
uv run tiffredact redact slide.svs redacted.svs \
    -r 5000,4000,3000,2000 -r 100,100,500,900 --drop-label --drop-macro

# audit the result against the source
uv run tiffredact verify slide.svs redacted.svs -r 5000,4000,3000,2000 -r 100,100,500,900
```

Options for `redact`:

| flag | meaning |
|---|---|
| `-r/--region X,Y,W,H` | rectangle at full resolution; repeatable |
| `--fill R,G,B` | fill colour, default black |
| `--drop-label` / `--drop-macro` | remove the Aperio label / macro IFDs |
| `--no-dct` | re-encode JPEG chunks in the pixel domain instead of editing DCT blocks |
| `--overwrite` | replace an existing output file |

`verify` exits non-zero if any pixel in the mapped region on any level is not
the fill colour, if any untouched chunk differs from the source, or if the bytes
of a replaced source chunk can still be found in the output.

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

## Guarantees and limits

* Untouched chunks are byte-identical to the source.
* Inside a rewritten JPEG chunk, the DCT coefficients of every block outside
  the snapped rectangle are unchanged. With chroma-subsampled JPEG the decoder's
  chroma upsampling blends a 1-2 px band on either side of the painted border;
  luma is exact.
* Over-redaction of up to one MCU (15 px) around each rectangle for JPEG.
* Not supported yet: `PlanarConfiguration=2`, sub-byte samples, CMYK/YCCK
  JPEG, NDPI/MRXS/DICOM, metadata scrubbing. Macro images cannot be mapped from
  slide coordinates; drop them with `--drop-macro`.

## Tests

```sh
uv run pytest
```

Synthetic pyramids are generated with `tifffile`. Tests on real Aperio files run
when `*.svs` files are present in the repository root and are skipped otherwise.
