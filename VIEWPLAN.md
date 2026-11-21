# tiffview plan

> Historical: the plan for tiffview, written before it was merged into
> coverslip. Module and command names below are tiffview's; `info` is now
> `coverslip inspect`.

A CLI tool for local viewing and inspection of whole-slide TIFF files (tiled
TIFF / BigTIFF, Aperio SVS, OME-TIFF). No tile server, no smooth zooming: pick a
pyramid level and view it whole, or pick individual tiles of a large level and
view them one at a time.

## Decisions

- **Display:** matplotlib windows (option A). Inline terminal images (iTerm2 /
  kitty / Ghostty protocols) may be added later.
- **Reading code:** copied from `../tiffredact` (read side only). If this works
  out, both projects may later be refactored onto shared modules, so copied
  modules stay as close to the originals as possible.
- **Tile clicks:** open a single tile, at a level chosen with `--tile-level N`
  (default 0). Every click opens a new window so tiles can be compared.
- **Grid map:** a heatmap of tile byte counts, shown on its own or overlaid on
  a level view.

## Project

uv project, `src/` layout, Python >= 3.14, Black (88), pytest.

Runtime dependencies: `tifffile`, `tifftools`, `imagecodecs`, `numpy` (as in
tiffredact) plus `matplotlib`. Dev: `pytest`, `openslide-python`,
`openslide-bin`.

## Modules

Copied from tiffredact:

- `model.py` — `Region`, `Chunk`, `ImageIFD`, `PyramidImage`, `Layout`, `Role`.
  Kept as-is.
- `layout.py` — `read_info`, `discover`, `describe`.
- `codecs/pixel.py` — decode half only (`decode_chunk` and helpers).

New:

- `chunks.py` — read a chunk's bytes (seek/read via `data_offsets` /
  `data_bytecounts`), decode a single tile, assemble a whole level from its
  tiles. Assembly refuses levels above a size cap (e.g. 8k x 8k) without
  `--force`.
- `info.py` — text report: per level dimensions, tile size, tile count,
  compression (by name), photometric, downsample, total bytes, empty
  (zero-bytecount) tiles; label / macro / thumbnail; optional `--tags` dump.
- `gridmap.py` — `chunks_down x chunks_across` array of tile byte counts for a
  level; mapping from a point on one level to a tile on another.
- `viewer.py` — matplotlib windows:
  - level view with built-in pan/zoom; optional grid-map overlay of the tile
    level (`g` toggles it); click opens the tile under the cursor at the tile
    level; status line shows coordinates and tile `(col,row)`.
  - tile view: one tile, titled with col/row, offset, bytecount and level
    coordinates.
  - grid map view: byte-count heatmap with colorbar; click opens that tile.
- `cli.py` — argparse; expected failures print one
  `tiffview <command>: error: ...` line and exit 2 (`-v` adds a traceback).

## CLI

```
tiffview info  slide.svs [--tags]
tiffview show  slide.svs [--level N | --label | --macro | --thumbnail]
                         [--grid] [--tile-level N] [--force]
tiffview tile  slide.svs (--col C --row R | --at X,Y) [--level N]
tiffview grid  slide.svs [--level N]
```

`show` without a level picks the smallest pyramid level whose long side is at
least ~1000 px. `--at X,Y` is in the coordinates of the chosen level.

## Tests

- Synthetic pyramid fixtures copied from tiffredact's `conftest.py`.
- Tile decode and level assembly equal the source array, including edge tiles.
- Point-to-tile mapping across levels.
- Grid map values against known bytecounts.
- CLI `info` output and error paths.
- Viewer logic kept separate from GUI calls; figures are built under the `Agg`
  backend so tests run headless.

## Commits

1. Plan (this file).
2. Scaffold + copied modules.
3. Chunk reading/assembly + info report.
4. Grid map.
5. Viewer.
6. CLI.
