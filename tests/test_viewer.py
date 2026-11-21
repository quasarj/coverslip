from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.widgets import click_and_drag

from coverslip import chunks, viewer
from coverslip.model import Region

from conftest import load_layout


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def click(ax: Any, x: float, y: float, button: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        inaxes=ax, button=button, xdata=x, ydata=y, canvas=ax.figure.canvas
    )


def test_to_display_shapes_and_types() -> None:
    rgb = np.zeros((4, 5, 3), np.uint8)
    assert viewer.to_display(rgb) is rgb
    assert viewer.to_display(np.zeros((4, 5, 1), np.uint8)).shape == (4, 5)
    assert viewer.to_display(np.zeros((4, 5, 6), np.uint8)).shape == (4, 5, 3)
    wide = np.array([[[100], [300]]], np.uint16)
    assert viewer.to_display(wide).tolist() == [[0.0, 1.0]]
    flat = viewer.to_display(np.full((2, 2, 1), 7, np.uint16))
    assert flat.max() == 0


def test_level_click_opens_tile_of_chosen_level(lzw_pyramid: Path) -> None:
    full, half, quarter = load_layout(lzw_pyramid).images[0].levels
    opened: list[tuple[int, int]] = []
    array = chunks.read_level(str(lzw_pyramid), quarter)
    view = viewer.level_figure(
        array, quarter, "q", tiles=full, open_tile=lambda c, r: opened.append((c, r))
    )
    view.on_click(click(view.ax, 40, 20))
    assert opened == [(2, 1)]
    assert len(view.ax.patches) == 1
    view.on_click(click(view.ax, 40, 20))
    assert opened == [(2, 1), (2, 1)]
    assert len(view.ax.patches) == 1, "a tile is outlined once"


@pytest.mark.parametrize(
    "event_args", [(-5.0, 10.0, 1), (40.0, 20.0, 3), (None, None, 1)]
)
def test_level_click_ignored(lzw_pyramid: Path, event_args: tuple) -> None:
    full, _, quarter = load_layout(lzw_pyramid).images[0].levels
    opened: list[tuple[int, int]] = []
    array = chunks.read_level(str(lzw_pyramid), quarter)
    view = viewer.level_figure(
        array, quarter, "q", tiles=full, open_tile=lambda c, r: opened.append((c, r))
    )
    view.on_click(click(view.ax, *event_args))
    view.on_click(SimpleNamespace(inaxes=None, button=1, xdata=1, ydata=1))
    assert opened == []


def test_level_without_tiles_is_not_clickable(lzw_pyramid: Path) -> None:
    quarter = load_layout(lzw_pyramid).images[0].levels[-1]
    array = chunks.read_level(str(lzw_pyramid), quarter)
    view = viewer.level_figure(array, quarter, "q")
    view.on_click(click(view.ax, 40, 20))
    assert view.overlay is None
    assert view.format_coord(40, 20) == "x=40 y=20"


def test_grid_overlay_toggle_and_extent(lzw_pyramid: Path) -> None:
    full, _, quarter = load_layout(lzw_pyramid).images[0].levels
    array = chunks.read_level(str(lzw_pyramid), quarter)
    view = viewer.level_figure(array, quarter, "q", tiles=full, show_grid=True)
    assert view.overlay is not None and view.overlay.get_visible()
    assert tuple(view.overlay.get_extent()) == (0, 112, 80, 0)
    assert view.ax.get_xlim() == (0, 105) and view.ax.get_ylim() == (75, 0)
    view.on_key(SimpleNamespace(key="g"))
    assert not view.overlay.get_visible()
    view.on_key(SimpleNamespace(key="x"))
    assert not view.overlay.get_visible()


def test_level_hover_reports_tile(lzw_pyramid: Path) -> None:
    full, _, quarter = load_layout(lzw_pyramid).images[0].levels
    array = chunks.read_level(str(lzw_pyramid), quarter)
    view = viewer.level_figure(array, quarter, "q", tiles=full)
    stored = full.data_bytecounts[chunks.tile_index(full, 2, 1)]
    assert view.format_coord(40.2, 20.7) == f"x=40 y=21  tile (2,1)  {stored:,} bytes"
    assert view.format_coord(200, 20) == "x=200 y=20"


def test_grid_view_click_and_hover(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    opened: list[tuple[int, int]] = []
    grid = viewer.grid_figure(full, "g", lambda c, r: opened.append((c, r)))
    grid.on_click(click(grid.ax, 6.5, 4.2))
    grid.on_click(click(grid.ax, 7.0, 0.5))
    assert opened == [(6, 4)]
    assert grid.format_coord(0.5, 0.5).startswith("tile (0,0)  ")
    assert grid.format_coord(-1, 0) == ""


def test_empty_tiles_are_masked(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    full.data_bytecounts[3] = 0
    grid = viewer.masked_byte_grid(full)
    assert grid.mask[0, 3] and grid.mask.sum() == 1


def test_tile_figure_uses_level_coordinates(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    tile = chunks.read_tile(str(lzw_pyramid), full, 6, 4)
    figure = viewer.tile_figure(tile, full, 6, 4, "t")
    image = figure.axes[0].images[0]
    assert tuple(image.get_extent()) == (384, 420, 300, 256)


def test_tile_title(lzw_pyramid: Path) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    index = chunks.tile_index(full, 2, 1)
    title = viewer.tile_title("0", full, 2, 1)
    assert title.startswith("level 0 tile (2,1) at 128,64  ")
    assert f"offset {full.data_offsets[index]}" in title


def test_tile_opener_logs_empty_tile(
    lzw_pyramid: Path, caplog: pytest.LogCaptureFixture
) -> None:
    full = load_layout(lzw_pyramid).images[0].full
    full.data_bytecounts[0] = 0
    open_tile = viewer.tile_opener(str(lzw_pyramid), full, "0")
    open_tile(0, 0)
    assert "Cannot open tile (0,0)" in caplog.text
    assert plt.get_fignums() == []


def test_configure_matplotlib_frees_g() -> None:
    viewer.configure_matplotlib()
    assert "g" not in plt.rcParams["keymap.grid"]


def drawing_view(path: Path, messages: list) -> viewer.LevelView:
    full, half, _ = load_layout(path).images[0].levels
    array = chunks.read_level(str(path), half)
    level = viewer.level_figure(
        array,
        half,
        "half",
        tiles=full,
        open_tile=lambda c, r: messages.append(("tile", c, r)),
        draw_regions=True,
        report=messages.append,
    )
    level.figure.canvas.draw()
    return level


def test_drag_draws_region_only_in_region_mode(lzw_pyramid: Path) -> None:
    messages: list = []
    level = drawing_view(lzw_pyramid, messages)
    click_and_drag(level.selector, start=(10.2, 20.5), end=(50.7, 60))
    assert level.regions == []
    level.on_key(SimpleNamespace(key="r"))
    assert level.region_mode and "drawing regions" in level.ax.get_title()
    click_and_drag(level.selector, start=(10.2, 20.5), end=(50.7, 60))
    assert level.regions == [Region(10, 20, 41, 40)]
    assert messages[-1] == "-r 10,20,41,40@210x150"
    assert level.region_args() == ["10,20,41,40@210x150"]
    assert len(level.region_patches) == 1


def test_clicks_do_not_open_tiles_in_region_mode(lzw_pyramid: Path) -> None:
    messages: list = []
    level = drawing_view(lzw_pyramid, messages)
    level.toggle_region_mode()
    level.on_click(click(level.ax, 40, 20))
    assert messages == []
    level.toggle_region_mode()
    assert level.ax.get_title() == "half"
    level.on_click(click(level.ax, 40, 20))
    assert messages == [("tile", 1, 0)]


def test_undo_removes_last_region(lzw_pyramid: Path) -> None:
    messages: list = []
    level = drawing_view(lzw_pyramid, messages)
    level.add_region(Region(1, 2, 3, 4))
    level.add_region(Region(5, 6, 7, 8))
    patch = level.region_patches[-1]
    level.on_key(SimpleNamespace(key="u"))
    assert level.regions == [Region(1, 2, 3, 4)]
    assert patch not in level.ax.patches
    assert messages[-1] == "removed -r 5,6,7,8@210x150"
    level.undo_region()
    level.undo_region()
    assert level.regions == []


def test_region_keys_ignored_without_drawing(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    level = viewer.level_figure(chunks.read_level(str(lzw_pyramid), half), half, "h")
    level.on_key(SimpleNamespace(key="r"))
    assert level.selector is None and not level.region_mode


def test_preview_regions_are_drawn(lzw_pyramid: Path) -> None:
    half = load_layout(lzw_pyramid).images[0].levels[1]
    level = viewer.level_figure(
        chunks.read_level(str(lzw_pyramid), half),
        half,
        "h",
        preview=[Region(1, 2, 3, 4), Region(10, 10, 5, 5)],
    )
    assert [p.get_linestyle() for p in level.ax.patches] == ["--", "--"]
    assert level.regions == []


def test_configure_matplotlib_frees_r_and_u() -> None:
    viewer.configure_matplotlib()
    for name, keys in plt.rcParams.items():
        if name.startswith("keymap."):
            assert not {"g", "r", "u"} & set(keys), name
