"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from tiffredact import __version__
from tiffredact import layout as layout_mod
from tiffredact.model import Region
from tiffredact.redact import redact_file
from tiffredact.scrub import describe_tags, parse_keep_tag
from tiffredact.verify import verify_file


def parse_fill(text: str) -> tuple[int, int, int]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("fill must be R,G,B")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fill values must be integers") from exc
    if any(v < 0 or v > 255 for v in values):
        raise argparse.ArgumentTypeError("fill values must be in 0..255")
    return values  # type: ignore[return-value]


def parse_region(text: str) -> Region:
    try:
        return Region.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_tag(text: str) -> int:
    try:
        return parse_keep_tag(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_region(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r",
        "--region",
        action="append",
        default=[],
        type=parse_region,
        metavar="X,Y,W,H",
        help="full-resolution rectangle; repeatable",
    )


def _add_fill(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fill",
        type=parse_fill,
        default=(0, 0, 0),
        metavar="R,G,B",
        help="fill colour",
    )


def _add_keep_tag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-tag",
        action="append",
        default=[],
        type=parse_tag,
        metavar="TAG",
        help="TIFF tag (number or name) to keep despite scrubbing; repeatable",
    )


def _add_drop_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--drop-label", action="store_true", help="remove label image(s)"
    )
    parser.add_argument(
        "--drop-macro", action="store_true", help="remove macro image(s)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiffredact",
        description="Redact rectangular pixel regions and identifying metadata "
        "in whole-slide TIFF images.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="show the pyramid layout of a file")
    p_inspect.add_argument("path")
    p_inspect.add_argument(
        "--tags", action="store_true", help="list every tag and what scrubbing would do"
    )

    p_redact = sub.add_parser("redact", help="redact regions into a new file")
    p_redact.add_argument("input")
    p_redact.add_argument("output")
    _add_region(p_redact)
    _add_fill(p_redact)
    _add_drop_flags(p_redact)
    p_redact.add_argument(
        "--no-dct",
        action="store_true",
        help="re-encode JPEG chunks in the pixel domain instead of editing DCT blocks",
    )
    p_redact.add_argument(
        "--scrub", action="store_true", help="also scrub identifying metadata"
    )
    _add_keep_tag(p_redact)

    p_scrub = sub.add_parser(
        "scrub", help="scrub identifying metadata without touching pixels"
    )
    p_scrub.add_argument("input")
    p_scrub.add_argument("output")
    _add_drop_flags(p_scrub)
    _add_keep_tag(p_scrub)

    p_verify = sub.add_parser("verify", help="audit a redacted file against its source")
    p_verify.add_argument("input")
    p_verify.add_argument("output")
    _add_region(p_verify)
    _add_fill(p_verify)
    p_verify.add_argument(
        "--scrubbed",
        action="store_true",
        help="also check that identifying metadata is gone",
    )
    _add_keep_tag(p_verify)
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    info = layout_mod.read_info(args.path)
    layout = layout_mod.discover(info)
    print(layout_mod.describe(layout))
    if args.tags:
        print()
        print(describe_tags(info))
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    if not args.region and not args.scrub:
        raise SystemExit("redact: at least one --region is required (or use --scrub)")
    report = redact_file(
        args.input,
        args.output,
        args.region,
        fill=args.fill,
        drop_label=args.drop_label,
        drop_macro=args.drop_macro,
        use_dct=not args.no_dct,
        overwrite=args.overwrite,
        scrub=args.scrub,
        keep_tags=args.keep_tag,
    )
    print(report.summary())
    return 0


def cmd_scrub(args: argparse.Namespace) -> int:
    report = redact_file(
        args.input,
        args.output,
        [],
        drop_label=args.drop_label,
        drop_macro=args.drop_macro,
        overwrite=args.overwrite,
        scrub=True,
        keep_tags=args.keep_tag,
    )
    print(report.summary())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_file(
        args.input,
        args.output,
        args.region,
        fill=args.fill,
        scrubbed=args.scrubbed,
        keep_tags=args.keep_tag,
    )
    print(report.summary())
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    handlers = {
        "inspect": cmd_inspect,
        "redact": cmd_redact,
        "scrub": cmd_scrub,
        "verify": cmd_verify,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
