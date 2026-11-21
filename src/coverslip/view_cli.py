"""The viewing commands: show, tile and grid.

The viewer needs matplotlib, an optional dependency (``coverslip[view]``), so
it is imported only when one of these commands runs; redaction stays usable
on a headless install.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from types import ModuleType

from coverslip import chunks, gridmap, regions
from coverslip import layout as layout_mod
from coverslip.model import ImageIFD, Layout, PyramidImage, RegionSpec, Role

DEFAULT_VIEW_SIDE = 1000
"""``show`` picks the smallest level whose long side is at least this."""


class SelectionError(Exception):
    """The requested level, image or tile does not exist in this file."""


class ViewerUnavailableError(Exception):
    """matplotlib is not installed."""


VIEW_ERRORS: tuple[type[Exception], ...] = (
    chunks.EmptyChunkError,
    chunks.LevelTooLargeError,
    chunks.TileIndexError,
    SelectionError,
    ViewerUnavailableError,
)
"""Expected failures of the viewing commands, on top of the main CLI's."""


def parse_point(text: str) -> tuple[float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be X,Y")
    try:
        x, y = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point values must be numbers") from exc
    return x, y


def checked_region(text: str) -> str:
    """Validate a ``-r`` style region but keep its text, to echo it back."""
    try:
        RegionSpec.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return text


def _add_image(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        type=int,
        default=0,
        metavar="M",
        help="pyramid to use when the file holds several (default 0)",
    )


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register show, tile and grid on the main CLI's subcommands."""
    p_show = sub.add_parser("show", help="open a whole level (or label/macro)")
    p_show.add_argument("path")
    which = p_show.add_mutually_exclusive_group()
    which.add_argument(
        "--level",
        type=int,
        metavar="N",
        help=f"pyramid level (default: smallest with a side >= {DEFAULT_VIEW_SIDE}px)",
    )
    which.add_argument("--label", action="store_true", help="show the label image")
    which.add_argument("--macro", action="store_true", help="show the macro image")
    which.add_argument(
        "--thumbnail", action="store_true", help="show the thumbnail image"
    )
    p_show.add_argument(
        "--tile-level",
        type=int,
        default=0,
        metavar="N",
        help="level whose tiles a click opens (default 0)",
    )
    p_show.add_argument(
        "--grid", action="store_true", help="start with the byte-count map shown"
    )
    p_show.add_argument("--force", action="store_true", help="show a level of any size")
    p_show.add_argument(
        "-r",
        "--region",
        action="append",
        default=[],
        type=checked_region,
        metavar="X,Y,W,H[@WxH]",
        help="preview a redaction region, as redact takes it; repeatable",
    )
    _add_image(p_show)

    p_tile = sub.add_parser("tile", help="open one tile")
    p_tile.add_argument("path")
    p_tile.add_argument("--level", type=int, default=0, metavar="N")
    p_tile.add_argument("--col", type=int, metavar="C")
    p_tile.add_argument("--row", type=int, metavar="R")
    p_tile.add_argument(
        "--at",
        type=parse_point,
        metavar="X,Y",
        help="open the tile containing this point (pixels of --level)",
    )
    _add_image(p_tile)

    p_grid = sub.add_parser("grid", help="open the tile byte-count map of a level")
    p_grid.add_argument("path")
    p_grid.add_argument("--level", type=int, default=0, metavar="N")
    _add_image(p_grid)


def load_viewer() -> ModuleType:
    """Import :mod:`coverslip.viewer`, explaining a missing matplotlib."""
    try:
        return importlib.import_module("coverslip.viewer")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "matplotlib":
            raise
        raise ViewerUnavailableError(
            "viewing needs matplotlib; install coverslip[view]"
        ) from exc


def load(path: str, verbose: bool) -> Layout:
    if not verbose:
        # layout reports at INFO how a redaction would write the file back
        # out; irrelevant when only viewing it
        logging.getLogger(layout_mod.__name__).setLevel(logging.WARNING)
    return layout_mod.discover(layout_mod.read_info(path))


def select_image(layout: Layout, index: int) -> PyramidImage:
    if not 0 <= index < len(layout.images):
        raise SelectionError(
            f"No image {index}; this file has {len(layout.images)} pyramid(s)"
        )
    return layout.images[index]


def select_level(image: PyramidImage, index: int) -> ImageIFD:
    if not 0 <= index < len(image.levels):
        sizes = ", ".join(
            f"{n}: {lvl.width}x{lvl.height}" for n, lvl in enumerate(image.levels)
        )
        raise SelectionError(f"No level {index}; levels are {sizes}")
    return image.levels[index]


def _smallest_big_enough(image: PyramidImage, candidates: list[int]) -> int | None:
    """The smallest candidate level that is at least ``DEFAULT_VIEW_SIDE`` on
    its long side and within the pixel limit."""
    chosen = None
    for n in candidates:
        level = image.levels[n]
        big_enough = max(level.width, level.height) >= DEFAULT_VIEW_SIDE
        within_limit = level.width * level.height <= chunks.DEFAULT_MAX_PIXELS
        if big_enough and within_limit:
            chosen = n
    return chosen


def default_view_level(image: PyramidImage) -> int:
    """Prefer a real (tiled) pyramid level over a stripped thumbnail.

    Thumbnails are considered only when no tiled level qualifies, as in files
    whose small levels are all stored in strips. Small images show level 0.
    """
    every = list(range(len(image.levels)))
    tiled = [n for n in every if image.levels[n].role != Role.THUMBNAIL]
    for candidates in (tiled, every):
        chosen = _smallest_big_enough(image, candidates)
        if chosen is not None:
            return chosen
    return 0


def select_associated(layout: Layout, role: Role) -> ImageIFD:
    candidates = {
        Role.LABEL: layout.label,
        Role.MACRO: layout.macro,
        Role.THUMBNAIL: [
            lvl for img in layout.images for lvl in img.levels if lvl.role == role
        ],
    }[role]
    if not candidates:
        raise SelectionError(f"This file has no {role.value} image")
    return candidates[0]


def tile_from_args(args: argparse.Namespace, ifd: ImageIFD) -> tuple[int, int]:
    if args.at is not None:
        if args.col is not None or args.row is not None:
            raise SelectionError("give either --at or --col/--row, not both")
        tile = gridmap.tile_under(ifd, ifd, *args.at)
        if tile is None:
            raise SelectionError(
                f"Point {args.at[0]:g},{args.at[1]:g} is outside the "
                f"{ifd.width}x{ifd.height} level"
            )
        return tile
    if args.col is None or args.row is None:
        raise SelectionError("give --col and --row, or --at X,Y")
    chunks.tile_index(ifd, args.col, args.row)
    return args.col, args.row


def window_title(path: str, what: str, ifd: ImageIFD) -> str:
    return f"{Path(path).name}  {what}  {ifd.width}x{ifd.height}"


def _show_associated(args: argparse.Namespace, layout: Layout, role: Role) -> None:
    viewer = load_viewer()
    ifd = select_associated(layout, role)
    max_pixels = None if args.force else chunks.DEFAULT_MAX_PIXELS
    array = chunks.read_level(args.path, ifd, max_pixels)
    viewer.level_figure(array, ifd, window_title(args.path, role.value, ifd))


def _show_level(args: argparse.Namespace, layout: Layout) -> list[str]:
    """Show a pyramid level; return the regions drawn, as ``-r`` arguments."""
    viewer = load_viewer()
    image = select_image(layout, args.image)
    n = args.level if args.level is not None else default_view_level(image)
    view = select_level(image, n)
    tiles = select_level(image, args.tile_level)
    specs = [RegionSpec.parse(text) for text in args.region]
    preview = regions.preview_on_level(layout, image.full, view, specs)
    max_pixels = None if args.force else chunks.DEFAULT_MAX_PIXELS
    array = chunks.read_level(args.path, view, max_pixels)
    print(
        f"showing level {n} ({view.width}x{view.height}); click to open level "
        f"{args.tile_level} tiles, g toggles the byte-count map, r draws "
        "redaction regions"
    )
    level = viewer.level_figure(
        array,
        view,
        window_title(args.path, f"level {n}", view),
        tiles=tiles,
        open_tile=viewer.tile_opener(args.path, tiles, str(args.tile_level)),
        show_grid=args.grid,
        draw_regions=True,
        preview=preview,
    )
    viewer.run()
    return level.region_args()


def cmd_show(args: argparse.Namespace) -> int:
    viewer = load_viewer()
    layout = load(args.path, args.verbose)
    viewer.configure_matplotlib()
    associated = _associated_role(args)
    if associated is not None:
        if args.region:
            raise SelectionError(
                f"--region previews apply to pyramid levels, not --{associated.value}"
            )
        _show_associated(args, layout, associated)
        viewer.run()
        return 0
    drawn = _show_level(args, layout)
    if drawn:
        print("redact with:")
        print(regions.redact_command(args.path, args.region + drawn))
    return 0


def _associated_role(args: argparse.Namespace) -> Role | None:
    if args.label:
        return Role.LABEL
    if args.macro:
        return Role.MACRO
    if args.thumbnail:
        return Role.THUMBNAIL
    return None


def cmd_tile(args: argparse.Namespace) -> int:
    viewer = load_viewer()
    layout = load(args.path, args.verbose)
    ifd = select_level(select_image(layout, args.image), args.level)
    col, row = tile_from_args(args, ifd)
    array = chunks.read_tile(args.path, ifd, col, row)
    title = viewer.tile_title(str(args.level), ifd, col, row)
    print(title)
    viewer.configure_matplotlib()
    viewer.tile_figure(array, ifd, col, row, title)
    viewer.run()
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    viewer = load_viewer()
    layout = load(args.path, args.verbose)
    ifd = select_level(select_image(layout, args.image), args.level)
    title = (
        f"{Path(args.path).name}  level {args.level}  "
        f"{ifd.chunks_across}x{ifd.chunks_down} tiles"
    )
    viewer.configure_matplotlib()
    viewer.grid_figure(ifd, title, viewer.tile_opener(args.path, ifd, str(args.level)))
    viewer.run()
    return 0


HANDLERS = {"show": cmd_show, "tile": cmd_tile, "grid": cmd_grid}
