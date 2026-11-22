"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from typing import Any

import tifftools

from coverslip import __version__, browse, info, view_cli
from coverslip import layout as layout_mod
from coverslip.codecs.fill import UnsupportedFill
from coverslip.codecs.jpeg_dct import JpegDctError
from coverslip.codecs.pixel import UnsupportedCodec
from coverslip.geometry import RegionFrameError
from coverslip.inplace import InPlaceError, redact_in_place
from coverslip.manifest import ManifestError
from coverslip.model import RegionSpec
from coverslip.redact import RedactionError, RedactionReport, redact_file
from coverslip.scrub import ScrubError, describe_tags, parse_keep_tag
from coverslip.storage import StorageError
from coverslip.verify import VerifyError, verify_file, verify_in_place
from coverslip.writer import WriterError

logger = logging.getLogger(__name__)

EXIT_ERROR = 2
"""Exit status for a run that could not be carried out; 1 is a failed audit."""

EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    InPlaceError,
    JpegDctError,
    layout_mod.LayoutError,
    ManifestError,
    OSError,
    RedactionError,
    RegionFrameError,
    ScrubError,
    StorageError,
    tifftools.TifftoolsError,
    browse.TuiUnavailableError,
    UnsupportedCodec,
    UnsupportedFill,
    VerifyError,
    WriterError,
    *view_cli.VIEW_ERRORS,
)
"""Failures caused by the input or the request rather than by a bug.

Anything else (``PatchError`` included: it means the planner broke its own
invariants) keeps its traceback.
"""


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


def parse_region(text: str) -> RegionSpec:
    try:
        return RegionSpec.parse(text)
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
        metavar="X,Y,W,H[@WxH]",
        help="rectangle to redact on every level; coordinates are full-resolution "
        "pixels, or pixels of the WxH pyramid level when @WxH is given "
        "(inspect lists the levels); repeatable",
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
        "--overwrite",
        action="store_true",
        help="replace an existing output file (or, in place, an existing manifest)",
    )


def _add_in_place(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--in-place", "--inplace", dest="in_place", action="store_true", help=help_text
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="in-place manifest (default: <input>.coverslip-manifest.json)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coverslip",
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

    p_browse = sub.add_parser(
        "browse",
        help="browse the layout interactively: open images, flag label and "
        "macro for dropping (the default when only a file is given)",
    )
    p_browse.add_argument("path")

    p_redact = sub.add_parser("redact", help="redact regions into a new file")
    p_redact.add_argument("input")
    p_redact.add_argument("output", nargs="?")
    _add_in_place(p_redact, "modify INPUT itself, writing only the changed bytes")
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
    p_scrub.add_argument("output", nargs="?")
    _add_in_place(p_scrub, "modify INPUT itself, writing only the changed bytes")
    _add_drop_flags(p_scrub)
    _add_keep_tag(p_scrub)

    p_verify = sub.add_parser("verify", help="audit a redacted file against its source")
    p_verify.add_argument("input")
    p_verify.add_argument("output", nargs="?")
    _add_in_place(
        p_verify,
        "audit INPUT, which was redacted in place, from its manifest "
        "(regions, fill and scrub settings are read from the manifest)",
    )
    _add_region(p_verify)
    _add_fill(p_verify)
    p_verify.add_argument(
        "--scrubbed",
        action="store_true",
        help="also check that identifying metadata is gone",
    )
    _add_keep_tag(p_verify)

    view_cli.add_parsers(sub)
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    raw = layout_mod.read_info(args.path)
    layout = layout_mod.discover(raw)
    print(info.format_report(layout))
    if args.tags:
        print()
        print(describe_tags(raw))
    return 0


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_browse(args: argparse.Namespace) -> int:
    """Open the browser; without a terminal, print the inspect report."""
    layout = view_cli.load(args.path, args.verbose)
    if not _is_interactive():
        print(info.format_report(layout))
        return 0
    show = partial(view_cli.show_entry, args.path, layout)
    plan = browse.load_tui().run_browser(layout, show, EXPECTED_ERRORS)
    line = browse.command(plan) if plan is not None else None
    if line is not None:
        print("redact with:")
        print(line)
    return 0


def _check_destination(args: argparse.Namespace) -> None:
    """Exactly one of OUTPUT and --in-place says where the result goes."""
    if args.in_place and args.output is not None:
        raise SystemExit(f"{args.command}: give either OUTPUT or --in-place, not both")
    if not args.in_place and args.output is None:
        raise SystemExit(f"{args.command}: OUTPUT is required unless --in-place")
    if args.manifest and not args.in_place:
        raise SystemExit(f"{args.command}: --manifest only applies with --in-place")


def _run_redaction(
    args: argparse.Namespace, regions: list[RegionSpec], scrub: bool, **options: Any
) -> RedactionReport:
    """Dispatch to the full rewrite or the in-place writer."""
    common = {
        "drop_label": args.drop_label,
        "drop_macro": args.drop_macro,
        "scrub": scrub,
        "keep_tags": args.keep_tag,
        **options,
    }
    if args.in_place:
        return redact_in_place(
            args.input,
            regions,
            manifest_path=args.manifest,
            overwrite_manifest=args.overwrite,
            **common,
        )
    return redact_file(
        args.input, args.output, regions, overwrite=args.overwrite, **common
    )


def cmd_redact(args: argparse.Namespace) -> int:
    _check_destination(args)
    if not args.region and not args.scrub:
        raise SystemExit("redact: at least one --region is required (or use --scrub)")
    report = _run_redaction(
        args, args.region, args.scrub, fill=args.fill, use_dct=not args.no_dct
    )
    print(report.summary())
    return 0


def cmd_scrub(args: argparse.Namespace) -> int:
    _check_destination(args)
    report = _run_redaction(args, [], True)
    print(report.summary())
    return 0


def cmd_verify_in_place(args: argparse.Namespace) -> int:
    if args.region or args.scrubbed or args.keep_tag:
        raise SystemExit(
            "verify: --in-place reads regions and scrub settings from the manifest"
        )
    report = verify_in_place(args.input, args.manifest)
    print(report.summary())
    return 0 if report.ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    _check_destination(args)
    if args.in_place:
        return cmd_verify_in_place(args)
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


HANDLERS = {
    "browse": cmd_browse,
    "inspect": cmd_inspect,
    "redact": cmd_redact,
    "scrub": cmd_scrub,
    "verify": cmd_verify,
    **view_cli.HANDLERS,
}


def with_default_command(argv: list[str]) -> list[str]:
    """``coverslip FILE`` is short for ``coverslip browse FILE``.

    The first word that is not an option names the command; anything that is
    not a command is taken to be the file.
    """
    for position, word in enumerate(argv):
        if word.startswith("-"):
            continue
        if word in HANDLERS:
            return argv
        return [*argv[:position], "browse", *argv[position:]]
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    words = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(with_default_command(words))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return HANDLERS[args.command](args)
    except EXPECTED_ERRORS as exc:
        return _report_error(args.command, exc)


def _report_error(command: str, exc: Exception) -> int:
    """Print an expected failure as one line; ``--verbose`` adds the traceback."""
    logger.debug("%s failed", command, exc_info=exc)
    print(f"coverslip {command}: error: {exc}", file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
