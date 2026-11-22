"""What the browser decides: which images to drop, and the command that does it.

The browser never modifies a file. It collects a :class:`Plan` (images
flagged for dropping, metadata scrubbing, regions drawn in the viewer) and
turns it into the ``coverslip`` command line that carries it out, the same
way ``show`` prints a command for the regions drawn in it.

Nothing here needs Textual, so the decisions can be tested without a
terminal.
"""

from __future__ import annotations

import importlib
import shlex
from dataclasses import dataclass, field
from types import ModuleType

from coverslip.model import ImageIFD, Layout, Role

DROP_FLAGS: dict[Role, str] = {Role.LABEL: "--drop-label", Role.MACRO: "--drop-macro"}
"""The images ``redact`` and ``scrub`` can remove, and the flag for each."""

NOTHING_TO_DO = "nothing flagged"
DROP_NEEDS_MORE = (
    "dropping images needs metadata scrubbing (s) or a drawn region as well"
)


class TuiUnavailableError(Exception):
    """Textual is not installed."""


def load_tui() -> ModuleType:
    """Import :mod:`coverslip.tui`, explaining a missing Textual."""
    try:
        return importlib.import_module("coverslip.tui")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in ("textual", "rich"):
            raise
        raise TuiUnavailableError(
            "browsing needs textual; install coverslip[tui]"
        ) from exc


@dataclass
class Plan:
    """A redaction the browser has been asked to prepare."""

    path: str
    drop: set[Role] = field(default_factory=set)
    scrub: bool = False
    regions: list[str] = field(default_factory=list)


def undroppable_reason(layout: Layout, ifd: ImageIFD) -> str | None:
    """Why ``ifd`` cannot be flagged for dropping; ``None`` if it can."""
    if ifd.role not in DROP_FLAGS:
        return (
            f"a {ifd.role.value} image cannot be dropped; "
            "only label and macro images can"
        )
    if layout.vendor == "ome":
        return (
            "label and macro images cannot be dropped from OME-TIFF: "
            "the OME-XML references IFDs by index"
        )
    return None


def droppable_roles(layout: Layout) -> set[Role]:
    """The roles this file has an image of that can be dropped."""
    associated = [*layout.label, *layout.macro]
    return {ifd.role for ifd in associated if undroppable_reason(layout, ifd) is None}


def default_plan(layout: Layout) -> Plan:
    """The default profile: scrub metadata and drop every label and macro.

    Both images can identify a patient: the label usually shows the
    accession number, and the macro photographs the whole slide, label
    included. Files they cannot be dropped from get scrubbing only.
    """
    return Plan(layout.path, drop=droppable_roles(layout), scrub=True)


def toggle_drop(plan: Plan, role: Role) -> None:
    """Flag or unflag every image of ``role``: drops are all-or-none by role."""
    plan.drop.symmetric_difference_update({role})


def _drop_words(plan: Plan) -> list[str]:
    return [flag for role, flag in DROP_FLAGS.items() if role in plan.drop]


def _region_words(plan: Plan) -> list[str]:
    return [word for region in plan.regions for word in ("-r", region)]


def command(plan: Plan) -> str | None:
    """The command that carries out ``plan``; OUTPUT is left to fill in.

    ``None`` when there is nothing to run, or nothing ``redact`` accepts:
    it refuses to drop images without also scrubbing or redacting a region.
    """
    if plan.regions:
        words = ["redact", plan.path, "OUTPUT", *_region_words(plan)]
        words += _drop_words(plan) + (["--scrub"] if plan.scrub else [])
    elif plan.scrub:
        words = ["scrub", plan.path, "OUTPUT", *_drop_words(plan)]
    else:
        return None
    return shlex.join(["coverslip", *words])


def describe(plan: Plan) -> str:
    """One line saying what quitting the browser will print."""
    line = command(plan)
    if line is not None:
        return line
    return DROP_NEEDS_MORE if plan.drop else NOTHING_TO_DO
