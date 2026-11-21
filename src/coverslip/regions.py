"""Regions drawn on a displayed pyramid level, as ``redact`` arguments.

A rectangle dragged on screen becomes ``X,Y,W,H@WxH`` in the pixels of the
level it was drawn on: exactly what ``redact -r`` accepts, so no precision is
lost on the way. Previews go the other way, from ``-r`` arguments to the
level on screen, through the same geometry ``redact`` uses.
"""

from __future__ import annotations

import math
import shlex

from coverslip.geometry import resolve_region, scale_region
from coverslip.model import ImageIFD, Layout, Region, RegionSpec


def region_from_drag(
    level: ImageIFD, x0: float, y0: float, x1: float, y1: float
) -> Region | None:
    """The pixels of ``level`` touched by a drag between two points.

    Rounds outward so every pixel the rectangle covers even partly is
    included, and clips to the level. ``None`` if nothing is left.
    """
    left = max(0, math.floor(min(x0, x1)))
    top = max(0, math.floor(min(y0, y1)))
    right = min(level.width, math.ceil(max(x0, x1)))
    bottom = min(level.height, math.ceil(max(y0, y1)))
    if right <= left or bottom <= top:
        return None
    return Region(left, top, right - left, bottom - top)


def region_arg(region: Region, level: ImageIFD) -> str:
    """``X,Y,W,H@WxH`` for a region in ``level`` pixels."""
    return (
        f"{region.x},{region.y},{region.width},{region.height}"
        f"@{level.width}x{level.height}"
    )


def redact_command(path: str, region_args: list[str]) -> str:
    """A ready-to-run ``coverslip redact`` line; OUTPUT is left to fill in."""
    words = ["coverslip", "redact", path, "OUTPUT"]
    for arg in region_args:
        words += ["-r", arg]
    return shlex.join(words)


def preview_on_level(
    layout: Layout, full: ImageIFD, level: ImageIFD, specs: list[RegionSpec]
) -> list[Region]:
    """Where ``redact`` would paint each spec, in the pixels of ``level``.

    Regions are resolved to full resolution and scaled back down exactly as
    ``redact`` does, so the preview shows outward rounding too. Regions that
    miss the level entirely are left out.
    """
    shown = []
    for spec in specs:
        region = scale_region(resolve_region(layout, spec), full, level)
        if region is not None:
            shown.append(region)
    return shown
