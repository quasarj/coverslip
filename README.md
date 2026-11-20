# tiffview

Inspect and view whole-slide images (tiled TIFF / BigTIFF, Aperio SVS,
OME-TIFF) from the command line. There is no tile server and no smooth
zooming: you pick one pyramid level and view it whole, or you pick single
tiles of a large level and open each in its own window.

## Install

```sh
uv sync
```

Runtime dependencies: `tifffile`, `tifftools`, `imagecodecs`, `numpy`,
`matplotlib`. OpenSlide (`openslide-python` + `openslide-bin`) is a dev
dependency, used only by the test suite.

## Usage

```sh
# what is in the file?
uv run tiffview info slide.svs

# ... including every tag of every IFD
uv run tiffview info slide.svs --tags

# open an overview; click it to open full-resolution tiles
uv run tiffview show slide.svs

# a specific level, with clicks opening tiles of level 1 instead of level 0
uv run tiffview show slide.svs --level 3 --tile-level 1

# start with the tile byte-count map overlaid
uv run tiffview show slide.svs --grid

# associated images
uv run tiffview show slide.svs --label
uv run tiffview show slide.svs --macro
uv run tiffview show slide.svs --thumbnail

# one tile, by grid position or by a point in that level's pixels
uv run tiffview tile slide.svs --col 120 --row 45
uv run tiffview tile slide.svs --level 1 --at 9000,6100

# the byte-count map of a level on its own; click a cell to open that tile
uv run tiffview grid slide.svs --level 0
```

Level numbers are the ones `info` prints: `0` is full resolution, and higher
numbers are smaller.

### `info`

`info` prints one row for each pyramid level and each associated image. Each
row gives:
- the size
- the tile (or strip) size and the tile grid
- the number of empty tiles (zero bytes stored)
- compression and photometric interpretation
- samples × bits
- the downsample factor from full resolution
- the bytes stored for that level

### `show`

`show` assembles one whole level from its tiles and opens it in a window.

| option | meaning |
|---|---|
| `--level N` | level to show |
| `--label` / `--macro` / `--thumbnail` | show an associated image instead |
| `--tile-level N` | level whose tiles a click opens (default 0) |
| `--grid` | start with the byte-count map shown |
| `--force` | show a level larger than the pixel limit |
| `--image M` | which pyramid, when the file holds several (default 0) |

**Default level.** Without `--level`, `show` picks the smallest tiled level
whose long side is at least 1000 px and that fits within the pixel limit. It
uses a stripped thumbnail only when no tiled level qualifies.

**Pixel limit.** Levels over 8192 × 8192 pixels are refused unless you pass
`--force`. Assembling a full-resolution level can take many gigabytes of
memory.

In the window:

- **Hover** to see the level coordinates, the tile under the cursor (at the
  tile level) and the bytes stored for that tile.
- **Left-click** to open that tile in a new window. Each click opens another
  window, so tiles can be compared side by side, and each opened tile is
  outlined in yellow on the level.
- **`g`** toggles the byte-count map. It colours each tile by the bytes
  stored for it, which shows tissue versus background at a glance. Empty
  tiles are red.
- Matplotlib's toolbar pans and zooms. Clicks don't open tiles while the pan
  or zoom tool is active.

### `tile`

`tile` opens one tile, cropped to the image bounds. The title and the
terminal output give the tile's grid position, its pixel position on the
level, its file offset and its stored size. Hovering shows coordinates on the
level the tile belongs to.

### `grid`

`grid` shows the byte-count map of one level as a heatmap with a colour bar.
Each cell is one tile, and empty tiles are red. Clicking a cell opens that
tile.

## Supported files

tiffview recognises the following layouts:
- Aperio SVS, including the label, macro and thumbnail images
- OME-TIFF and other SubIFD pyramids
- pyramids stored as a chain of reduced-resolution IFDs

Supported compressions:
- none, LZW, Deflate, PackBits, Zstd
- JPEG, including YCbCr (shown as RGB) and Aperio's RGB JPEG
- JPEG 2000, including Aperio 33003 / 33005
- WebP

Images that aren't 8-bit are contrast-stretched for display. Images with one
sample are shown in grey.

## Errors

A run that cannot be carried out prints one `tiffview <command>: error: ...`
line and exits with status 2. Examples:
- an unknown level
- a point outside the image
- a level over the pixel limit
- an unreadable file

Add `-v` for the traceback and debug logging.

When a clicked tile cannot be opened (an empty tile or an unsupported codec),
tiffview logs a warning and the viewer keeps running.

## Tests

```sh
uv run pytest
```

The tests build small synthetic pyramids with `tifffile`, and they run
headless on matplotlib's Agg backend.

A few tests compare decoded tiles against OpenSlide on real slides in
`samples/`, which is git-ignored. Those tests are skipped when the slides are
missing.

## Relationship to tiffredact

tiffview copies its read-side code from
[tiffredact](https://github.com/quasarj/tiffredact):
- `model.py` and `layout.py`, unchanged apart from the package name
- the decode half of `codecs/pixel.py`

The copy adds a `to_rgb` option to `decode_chunk`. tiffredact keeps samples
exactly as stored because it re-encodes them, while a viewer needs RGB. The
two projects may later share these modules.
