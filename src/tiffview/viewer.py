"""Matplotlib windows: a whole level, single tiles and the tile byte-count map.

Figures are built from arrays and layout objects, and clicks call an injected
``open_tile(col, row)`` callback, so everything here also works (and is
tested) under the non-interactive Agg backend. Only :func:`run` blocks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle

from tiffview import chunks, gridmap
from tiffview.codecs.pixel import UnsupportedCodec
from tiffview.model import ImageIFD

logger = logging.getLogger(__name__)

OpenTile = Callable[[int, int], None]

MAX_FIGURE_INCHES = 12.0
GRID_ALPHA = 0.45
GRID_CMAP = "viridis"
EMPTY_TILE_COLOUR = "red"
OPENED_TILE_COLOUR = "yellow"
TOGGLE_GRID_KEY = "g"


def configure_matplotlib() -> None:
    """Free ``g`` (matplotlib toggles axis gridlines with it) for the grid map."""
    plt.rcParams["keymap.grid"] = [
        key for key in plt.rcParams["keymap.grid"] if key != TOGGLE_GRID_KEY
    ]


def run() -> None:
    """Show every open figure and block until all windows are closed."""
    plt.show()


def figsize_for(width: int, height: int) -> tuple[float, float]:
    scale = MAX_FIGURE_INCHES / max(width, height)
    return max(width * scale, 4.0), max(height * scale, 3.0)


def to_display(array: np.ndarray) -> np.ndarray:
    """Make a decoded (height, width, samples) array something imshow can draw.

    One-sample images become 2-D (drawn in grey); images with more than four
    samples show their first three. Anything but 8-bit is stretched to 0..1.
    """
    samples = array.shape[2]
    if samples in (1, 2):
        array = array[:, :, 0]
    elif samples > 4:
        array = array[:, :, :3]
    if array.dtype == np.uint8:
        return array
    return stretch(array)


def stretch(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _gray_or_none(display: np.ndarray) -> str | None:
    return "gray" if display.ndim == 2 else None


def masked_byte_grid(ifd: ImageIFD) -> np.ma.MaskedArray:
    """Byte grid with empty tiles masked, so they draw in the 'bad' colour."""
    grid = gridmap.byte_grid(ifd)
    return np.ma.masked_equal(grid, 0)


def grid_colormap() -> Colormap:
    return plt.get_cmap(GRID_CMAP).with_extremes(bad=EMPTY_TILE_COLOUR)


def is_plain_click(event: Any, ax: Axes) -> bool:
    """A left click inside ``ax`` while no pan/zoom tool is active."""
    if event.inaxes is not ax or event.button != 1:
        return False
    if event.xdata is None or event.ydata is None:
        return False
    toolbar = getattr(event.canvas, "toolbar", None)
    return toolbar is None or not toolbar.mode


@dataclass
class LevelView:
    """A whole level, optionally clickable to open tiles of a ``tiles`` level."""

    figure: Figure
    ax: Axes
    view: ImageIFD
    tiles: ImageIFD | None
    open_tile: OpenTile | None
    byte_counts: np.ndarray | None = None
    overlay: AxesImage | None = None
    opened: set[tuple[int, int]] = field(default_factory=set)

    def toggle_grid(self) -> None:
        if self.overlay is None:
            return
        self.overlay.set_visible(not self.overlay.get_visible())
        self.figure.canvas.draw_idle()

    def mark_opened(self, col: int, row: int) -> None:
        if self.tiles is None or (col, row) in self.opened:
            return
        self.opened.add((col, row))
        x0, y0, x1, y1 = gridmap.tile_box_on_view(self.view, self.tiles, col, row)
        box = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=OPENED_TILE_COLOUR,
            linewidth=1.5,
        )
        self.ax.add_patch(box)
        self.figure.canvas.draw_idle()

    def on_click(self, event: Any) -> None:
        if self.tiles is None or self.open_tile is None:
            return
        if not is_plain_click(event, self.ax):
            return
        tile = gridmap.tile_under(self.view, self.tiles, event.xdata, event.ydata)
        if tile is None:
            return
        self.mark_opened(*tile)
        self.open_tile(*tile)

    def on_key(self, event: Any) -> None:
        if event.key == TOGGLE_GRID_KEY:
            self.toggle_grid()

    def format_coord(self, x: float, y: float) -> str:
        text = f"x={x:.0f} y={y:.0f}"
        if self.tiles is None or self.byte_counts is None:
            return text
        tile = gridmap.tile_under(self.view, self.tiles, x, y)
        if tile is None:
            return text
        col, row = tile
        stored = self.byte_counts[row, col]
        return f"{text}  tile ({col},{row})  {stored:,} bytes"


def _add_grid_overlay(ax: Axes, view: ImageIFD, tiles: ImageIFD) -> AxesImage:
    overlay = ax.imshow(
        masked_byte_grid(tiles),
        extent=gridmap.grid_extent(view, tiles),
        cmap=grid_colormap(),
        alpha=GRID_ALPHA,
        interpolation="nearest",
    )
    # the grid covers whole nominal tiles; keep the view on the image itself
    ax.set_xlim(0, view.width)
    ax.set_ylim(view.height, 0)
    return overlay


def level_figure(
    array: np.ndarray,
    view: ImageIFD,
    title: str,
    *,
    tiles: ImageIFD | None = None,
    open_tile: OpenTile | None = None,
    show_grid: bool = False,
) -> LevelView:
    """A window showing a whole level.

    With ``tiles``, hovering reports the tile of that level under the cursor,
    ``g`` toggles its byte-count map and a left click calls ``open_tile``.
    """
    figure = plt.figure(figsize=figsize_for(view.width, view.height))
    figure.canvas.manager.set_window_title(title)
    ax = figure.add_subplot()
    display = to_display(array)
    # extent puts pixel (i, j) at [i, i+1) x [j, j+1), so event coordinates
    # are level pixel coordinates without a half-pixel shift
    ax.imshow(
        display, cmap=_gray_or_none(display), extent=(0, view.width, view.height, 0)
    )
    ax.set_title(title)
    level = LevelView(figure, ax, view, tiles, open_tile)
    if tiles is not None:
        level.byte_counts = gridmap.byte_grid(tiles)
        level.overlay = _add_grid_overlay(ax, view, tiles)
        level.overlay.set_visible(show_grid)
    ax.format_coord = level.format_coord
    figure.canvas.mpl_connect("button_press_event", level.on_click)
    figure.canvas.mpl_connect("key_press_event", level.on_key)
    return level


def tile_title(level_name: str, ifd: ImageIFD, col: int, row: int) -> str:
    index = chunks.tile_index(ifd, col, row)
    chunk = chunks.chunk_at(ifd, index)
    return (
        f"level {level_name} tile ({col},{row}) at {chunk.x},{chunk.y}  "
        f"offset {ifd.data_offsets[index]}  {ifd.data_bytecounts[index]:,} bytes"
    )


def tile_figure(
    array: np.ndarray, ifd: ImageIFD, col: int, row: int, title: str
) -> Figure:
    """A window showing one tile; hover reports level coordinates."""
    chunk = chunks.chunk_at(ifd, chunks.tile_index(ifd, col, row))
    height, width = array.shape[:2]
    figure = plt.figure(figsize=figsize_for(width, height))
    figure.canvas.manager.set_window_title(title)
    ax = figure.add_subplot()
    display = to_display(array)
    ax.imshow(
        display,
        cmap=_gray_or_none(display),
        extent=(chunk.x, chunk.x + width, chunk.y + height, chunk.y),
        interpolation="nearest",
    )
    ax.set_title(title, fontsize="small")
    ax.format_coord = lambda x, y: f"x={x:.0f} y={y:.0f}"
    return figure


@dataclass
class GridView:
    """The byte-count map of a level on its own; a click opens a tile."""

    figure: Figure
    ax: Axes
    ifd: ImageIFD
    open_tile: OpenTile
    byte_counts: np.ndarray

    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        col, row = int(x), int(y)
        if 0 <= col < self.ifd.chunks_across and 0 <= row < self.ifd.chunks_down:
            return col, row
        return None

    def on_click(self, event: Any) -> None:
        if not is_plain_click(event, self.ax):
            return
        cell = self.cell_at(event.xdata, event.ydata)
        if cell is not None:
            self.open_tile(*cell)

    def format_coord(self, x: float, y: float) -> str:
        cell = self.cell_at(x, y)
        if cell is None:
            return ""
        col, row = cell
        stored = self.byte_counts[row, col]
        return f"tile ({col},{row})  {stored:,} bytes"


def grid_figure(ifd: ImageIFD, title: str, open_tile: OpenTile) -> GridView:
    """A window showing stored bytes per tile, with empty tiles in red."""
    across, down = ifd.chunks_across, ifd.chunks_down
    figure = plt.figure(figsize=figsize_for(across, down))
    figure.canvas.manager.set_window_title(title)
    ax = figure.add_subplot()
    # extent makes cell (col, row) span [col, col+1) x [row, row+1)
    image = ax.imshow(
        masked_byte_grid(ifd),
        cmap=grid_colormap(),
        extent=(0, across, down, 0),
        interpolation="nearest",
    )
    figure.colorbar(image, ax=ax, label="stored bytes per tile")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    grid = GridView(figure, ax, ifd, open_tile, gridmap.byte_grid(ifd))
    ax.format_coord = grid.format_coord
    figure.canvas.mpl_connect("button_press_event", grid.on_click)
    return grid


TILE_ERRORS = (chunks.EmptyChunkError, UnsupportedCodec, OSError, ValueError)
"""Failures opening one tile that should not take the whole viewer down."""


def tile_opener(path: str, ifd: ImageIFD, level_name: str) -> OpenTile:
    """An ``open_tile`` callback that decodes the tile and shows a new window."""

    def open_tile(col: int, row: int) -> None:
        try:
            array = chunks.read_tile(path, ifd, col, row)
        except TILE_ERRORS as exc:
            logger.warning("Cannot open tile (%d,%d): %s", col, row, exc)
            return
        title = tile_title(level_name, ifd, col, row)
        tile_figure(array, ifd, col, row, title).show()

    return open_tile
