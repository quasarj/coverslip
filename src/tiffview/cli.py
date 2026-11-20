"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import tifftools

from tiffview import __version__, chunks, gridmap, info, viewer
from tiffview import layout as layout_mod
from tiffview.codecs.pixel import UnsupportedCodec
from tiffview.model import ImageIFD, Layout, PyramidImage, Role

logger = logging.getLogger(__name__)

EXIT_ERROR = 2
"""Exit status for a run that could not be carried out."""

DEFAULT_VIEW_SIDE = 1000
"""``show`` picks the smallest level whose long side is at least this."""


class SelectionError(Exception):
    """The requested level, image or tile does not exist in this file."""


EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    chunks.EmptyChunkError,
    chunks.LevelTooLargeError,
    chunks.TileIndexError,
    layout_mod.LayoutError,
    OSError,
    SelectionError,
    tifftools.TifftoolsError,
    UnsupportedCodec,
)
"""Failures caused by the input or the request rather than by a bug."""


def parse_point(text: str) -> tuple[float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be X,Y")
    try:
        x, y = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point values must be numbers") from exc
    return x, y


def _add_image(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        type=int,
        default=0,
        metavar="M",
        help="pyramid to use when the file holds several (default 0)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiffview",
        description="Inspect and view whole-slide TIFF images, level by level "
        "and tile by tile.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="list pyramid levels and associated images")
    p_info.add_argument("path")
    p_info.add_argument("--tags", action="store_true", help="also dump every tag")

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
    return parser


def load(path: str) -> tuple[dict[str, Any], Layout]:
    raw = layout_mod.read_info(path)
    return raw, layout_mod.discover(raw)


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


def cmd_info(args: argparse.Namespace) -> int:
    raw, layout = load(args.path)
    print(info.format_report(layout))
    if args.tags:
        print()
        print(info.format_tags(raw, layout))
    return 0


def _show_associated(args: argparse.Namespace, layout: Layout, role: Role) -> None:
    ifd = select_associated(layout, role)
    max_pixels = None if args.force else chunks.DEFAULT_MAX_PIXELS
    array = chunks.read_level(args.path, ifd, max_pixels)
    viewer.level_figure(array, ifd, window_title(args.path, role.value, ifd))


def _show_level(args: argparse.Namespace, layout: Layout) -> None:
    image = select_image(layout, args.image)
    n = args.level if args.level is not None else default_view_level(image)
    view = select_level(image, n)
    tiles = select_level(image, args.tile_level)
    max_pixels = None if args.force else chunks.DEFAULT_MAX_PIXELS
    array = chunks.read_level(args.path, view, max_pixels)
    print(
        f"showing level {n} ({view.width}x{view.height}); click to open level "
        f"{args.tile_level} tiles, g toggles the byte-count map"
    )
    viewer.level_figure(
        array,
        view,
        window_title(args.path, f"level {n}", view),
        tiles=tiles,
        open_tile=viewer.tile_opener(args.path, tiles, str(args.tile_level)),
        show_grid=args.grid,
    )


def cmd_show(args: argparse.Namespace) -> int:
    _, layout = load(args.path)
    viewer.configure_matplotlib()
    if args.label:
        _show_associated(args, layout, Role.LABEL)
    elif args.macro:
        _show_associated(args, layout, Role.MACRO)
    elif args.thumbnail:
        _show_associated(args, layout, Role.THUMBNAIL)
    else:
        _show_level(args, layout)
    viewer.run()
    return 0


def cmd_tile(args: argparse.Namespace) -> int:
    _, layout = load(args.path)
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
    _, layout = load(args.path)
    ifd = select_level(select_image(layout, args.image), args.level)
    title = (
        f"{Path(args.path).name}  level {args.level}  "
        f"{ifd.chunks_across}x{ifd.chunks_down} tiles"
    )
    viewer.configure_matplotlib()
    viewer.grid_figure(ifd, title, viewer.tile_opener(args.path, ifd, str(args.level)))
    viewer.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # the copied layout code reports, at INFO, how tiffredact would write
        # the file back out; irrelevant for viewing
        logging.getLogger("tiffview.layout").setLevel(logging.WARNING)
    handlers = {
        "info": cmd_info,
        "show": cmd_show,
        "tile": cmd_tile,
        "grid": cmd_grid,
    }
    try:
        return handlers[args.command](args)
    except EXPECTED_ERRORS as exc:
        return _report_error(args.command, exc)


def _report_error(command: str, exc: Exception) -> int:
    """Print an expected failure as one line; ``--verbose`` adds the traceback."""
    logger.debug("%s failed", command, exc_info=exc)
    print(f"tiffview {command}: error: {exc}", file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
