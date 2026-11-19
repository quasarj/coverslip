"""Metadata scrubbing: remove identifying information from tags and descriptions.

Policy: a TIFF tag survives only if it is on the structural allowlist (needed
to decode or colour-manage the image) or explicitly kept by the caller.
``ImageDescription`` is rewritten per vendor so readers keep working:

* Aperio: the header line is kept, ``Key = Value`` pairs are filtered by an
  allowlist (``AppMag``, ``MPP``, ... survive; ``Filename``, ``Date``,
  ``User``, ``ScanScope ID`` ... are removed).
* OME-TIFF: experimenters, acquisition dates, descriptions, image names,
  structured annotations, ROIs, serial numbers and project/plate/screen
  containers are removed from the OME-XML; pixel structure is untouched.
* Anything else: removed, except tifffile's ``{"shape": ...}`` JSON and
  ImageJ hyperstack headers made only of numeric key=value lines.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

import tifftools

from tiffredact.layout import TAG_IMAGE_DESCRIPTION, iter_all_ifds

logger = logging.getLogger(__name__)

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# Tags that describe how to decode or display the pixels. Everything else is
# removed unless the caller keeps it explicitly.
ALLOWED_TAGS: frozenset[int] = frozenset(
    {
        254,  # NewSubfileType
        255,  # SubfileType
        256,  # ImageWidth
        257,  # ImageLength
        258,  # BitsPerSample
        259,  # Compression
        262,  # Photometric
        263,  # Threshholding
        266,  # FillOrder
        270,  # ImageDescription (rewritten, see scrub_description)
        271,  # Make
        272,  # Model
        273,  # StripOffsets
        274,  # Orientation
        277,  # SamplesPerPixel
        278,  # RowsPerStrip
        279,  # StripByteCounts
        280,  # MinSampleValue
        281,  # MaxSampleValue
        282,  # XResolution
        283,  # YResolution
        284,  # PlanarConfig
        296,  # ResolutionUnit
        297,  # PageNumber
        305,  # Software
        317,  # Predictor
        318,  # WhitePoint
        319,  # PrimaryChromaticities
        320,  # ColorMap
        322,  # TileWidth
        323,  # TileLength
        324,  # TileOffsets
        325,  # TileByteCounts
        330,  # SubIFDs
        332,  # InkSet
        336,  # DotRange
        338,  # ExtraSamples
        339,  # SampleFormat
        340,  # SMinSampleValue
        341,  # SMaxSampleValue
        347,  # JPEGTables
        530,  # YCbCrSubsampling
        531,  # YCbCrPositioning
        532,  # ReferenceBlackWhite
        32997,  # ImageDepth
        32998,  # TileDepth
        34675,  # ICC Profile
    }
)

APERIO_ALLOWED_KEYS: frozenset[str] = frozenset(
    key.lower()
    for key in (
        "AppMag",
        "MPP",
        "StripeWidth",
        "Left",
        "Top",
        "LineCameraSkew",
        "LineAreaXOffset",
        "LineAreaYOffset",
        "Focus Offset",
        "Exposure Time",
        "Exposure Scale",
        "DisplayColor",
        "Gamma",
        "ImageQuality",
        "OriginalWidth",
        "OriginalHeight",
        "Filtered",
        "Parmset",
        "ICC Profile",
        "SessonMode",
        "SessionMode",
    )
)

OME_REMOVED_ELEMENTS: frozenset[str] = frozenset(
    {
        "Experimenter",
        "ExperimenterGroup",
        "ExperimenterRef",
        "ExperimenterGroupRef",
        "StructuredAnnotations",
        "AnnotationRef",
        "AcquisitionDate",
        "Description",
        "Project",
        "ProjectRef",
        "Dataset",
        "DatasetRef",
        "Folder",
        "FolderRef",
        "Plate",
        "PlateRef",
        "Screen",
        "ScreenRef",
        "ROI",
        "ROIRef",
    }
)
OME_REMOVED_ATTRIBUTES: frozenset[str] = frozenset({"SerialNumber", "LotNumber"})
OME_NAMED_ELEMENTS_TO_UNNAME: frozenset[str] = frozenset({"Image", "Instrument"})


class ScrubError(Exception):
    """Metadata could not be scrubbed safely."""


@dataclass
class ScrubReport:
    removed_tags: Counter[int] = field(default_factory=Counter)
    removed_description_keys: Counter[str] = field(default_factory=Counter)
    removed_ome_items: Counter[str] = field(default_factory=Counter)
    descriptions_removed: int = 0

    def summary(self) -> str:
        lines = []
        if self.removed_tags:
            names = ", ".join(
                f"{tag_name(tag)} x{count}"
                for tag, count in sorted(self.removed_tags.items())
            )
            lines.append(f"  removed tags: {names}")
        if self.removed_description_keys:
            keys = ", ".join(sorted(self.removed_description_keys))
            lines.append(f"  removed description keys: {keys}")
        if self.removed_ome_items:
            items = ", ".join(
                f"{name} x{count}"
                for name, count in sorted(self.removed_ome_items.items())
            )
            lines.append(f"  removed OME-XML items: {items}")
        if self.descriptions_removed:
            lines.append(
                f"  ImageDescription tags removed: {self.descriptions_removed}"
            )
        return "\n".join(lines) if lines else "  nothing to scrub"


def tag_name(tag: int) -> str:
    try:
        return f"{tag}:{tifftools.Tag[tag].name}"
    except KeyError:
        return str(tag)


def parse_keep_tag(text: str) -> int:
    """Accept a tag number or a tifftools tag name."""
    if text.isdigit():
        return int(text)
    try:
        return int(tifftools.Tag[text])
    except KeyError as exc:
        raise ValueError(f"Unknown TIFF tag {text!r}") from exc


# --- Aperio -----------------------------------------------------------------


def scrub_aperio_description(text: str) -> tuple[str, list[str]]:
    """Keep the header and allowlisted ``Key = Value`` pairs."""
    parts = text.split("|")
    kept = [parts[0]]
    removed: list[str] = []
    for part in parts[1:]:
        key = part.split("=", 1)[0].strip()
        if key.lower() in APERIO_ALLOWED_KEYS:
            kept.append(part)
        else:
            removed.append(key)
    return "|".join(kept), removed


def aperio_description_keys(text: str) -> list[str]:
    return [part.split("=", 1)[0].strip() for part in text.split("|")[1:]]


# --- OME-XML ----------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def scrub_ome_xml(xml: str) -> tuple[str, Counter[str]]:
    """Remove identifying elements/attributes; keep pixel structure intact."""
    ElementTree.register_namespace("", OME_NS)
    ElementTree.register_namespace("xsi", XSI_NS)
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ScrubError(f"OME-XML could not be parsed: {exc}") from exc
    removed: Counter[str] = Counter()
    _scrub_ome_element(root, removed)
    body = ElementTree.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>' + body, removed


def _scrub_ome_element(element: ElementTree.Element, removed: Counter[str]) -> None:
    for child in list(element):
        name = _local(child.tag)
        if name in OME_REMOVED_ELEMENTS:
            element.remove(child)
            removed[name] += 1
            continue
        _scrub_ome_element(child, removed)
    for attr in list(element.attrib):
        local = _local(attr)
        if local in OME_REMOVED_ATTRIBUTES or (
            local == "Name" and _local(element.tag) in OME_NAMED_ELEMENTS_TO_UNNAME
        ):
            del element.attrib[attr]
            removed[f"{_local(element.tag)}@{local}"] += 1


def ome_xml_problems(xml: str) -> list[str]:
    """Names of identifying items still present in an OME-XML document."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        return [f"OME-XML does not parse: {exc}"]
    problems: list[str] = []
    for element in root.iter():
        name = _local(element.tag)
        if name in OME_REMOVED_ELEMENTS:
            problems.append(f"OME-XML still contains <{name}>")
        for attr in element.attrib:
            local = _local(attr)
            if local in OME_REMOVED_ATTRIBUTES or (
                local == "Name" and name in OME_NAMED_ELEMENTS_TO_UNNAME
            ):
                problems.append(f"OME-XML still contains {name}@{local}")
    return problems


# --- descriptions -----------------------------------------------------------


IMAGEJ_SAFE_KEYS: frozenset[str] = frozenset(
    {
        "ImageJ",
        "hyperstack",
        "images",
        "channels",
        "slices",
        "frames",
        "unit",
        "spacing",
        "loop",
        "min",
        "max",
        "finterval",
        "fps",
        "mode",
        "xorigin",
        "yorigin",
        "zorigin",
        "vunit",
        "yunit",
        "zunit",
        "cf",
    }
)


def _is_shaped_json(text: str) -> bool:
    return text.lstrip().startswith('{"shape"')


def _is_imagej_structural(text: str) -> bool:
    """ImageJ hyperstack headers are numeric key=value lines; keep those only."""
    if not text.startswith("ImageJ="):
        return False
    for line in text.splitlines():
        if not line.strip():
            continue
        key = line.split("=", 1)[0]
        if key not in IMAGEJ_SAFE_KEYS and not (
            key.startswith("c") and key[1:].isdigit()
        ):
            return False
    return True


def _is_structural_description(text: str) -> bool:
    return _is_shaped_json(text) or _is_imagej_structural(text)


def scrub_description(text: str, vendor: str, report: ScrubReport) -> str | None:
    """Return the scrubbed description, or None to drop the tag."""
    if vendor == "aperio" and text.startswith("Aperio"):
        cleaned, removed = scrub_aperio_description(text)
        report.removed_description_keys.update(removed)
        return cleaned
    if "<OME" in text:
        cleaned, removed = scrub_ome_xml(text)
        report.removed_ome_items.update(removed)
        return cleaned
    if _is_structural_description(text):
        return text
    report.descriptions_removed += 1
    return None


def description_problems(text: str, vendor: str) -> list[str]:
    if vendor == "aperio" and text.startswith("Aperio"):
        bad = [
            k
            for k in aperio_description_keys(text)
            if k.lower() not in APERIO_ALLOWED_KEYS
        ]
        return [f"Aperio description still has key {k!r}" for k in bad]
    if "<OME" in text:
        return ome_xml_problems(text)
    if _is_structural_description(text):
        return []
    return ["ImageDescription still present"]


# --- whole file -------------------------------------------------------------


def _description_text(entry: dict[str, Any]) -> str:
    data = entry["data"]
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def scrub_ifd(
    ifd: dict[str, Any], vendor: str, keep: set[int], report: ScrubReport
) -> None:
    tags = ifd["tags"]
    for tag in list(tags):
        if tag == TAG_IMAGE_DESCRIPTION and tag not in keep:
            cleaned = scrub_description(_description_text(tags[tag]), vendor, report)
            if cleaned is None:
                del tags[tag]
            else:
                tags[tag] = {"datatype": int(tifftools.Datatype.ASCII), "data": cleaned}
            continue
        if tag in ALLOWED_TAGS or tag in keep:
            continue
        del tags[tag]
        report.removed_tags[tag] += 1


def scrub_info(
    info: dict[str, Any], vendor: str, keep: Iterable[int] = ()
) -> ScrubReport:
    """Scrub every IFD (main chain and SubIFDs) in place."""
    report = ScrubReport()
    keep_set = set(keep)
    for ifd in iter_all_ifds(info):
        scrub_ifd(ifd, vendor, keep_set, report)
    return report


def scrub_problems(
    info: dict[str, Any], vendor: str, keep: Iterable[int] = ()
) -> list[str]:
    """Audit: identifying metadata that is still present."""
    keep_set = set(keep)
    problems: list[str] = []
    for index, ifd in enumerate(iter_all_ifds(info)):
        for tag, entry in ifd["tags"].items():
            if tag in keep_set:
                continue
            if tag == TAG_IMAGE_DESCRIPTION:
                for problem in description_problems(_description_text(entry), vendor):
                    problems.append(f"IFD {index}: {problem}")
            elif tag not in ALLOWED_TAGS:
                problems.append(f"IFD {index}: tag {tag_name(tag)} still present")
    return problems


def describe_tags(info: dict[str, Any], max_value: int = 100) -> str:
    """Human-readable listing of every tag, marking what scrubbing removes."""
    lines = []
    for index, ifd in enumerate(iter_all_ifds(info)):
        lines.append(f"IFD {index} @ {ifd['offset']}")
        for tag, entry in sorted(ifd["tags"].items()):
            if "ifds" in entry:
                value = f"<{len(entry['ifds'])} sub-IFD chain(s)>"
            else:
                raw = entry.get("data", "")
                value = repr(raw)
                if len(value) > max_value:
                    value = value[:max_value] + "..."
            mark = " " if tag in ALLOWED_TAGS else "-"
            if tag == TAG_IMAGE_DESCRIPTION:
                mark = "~"
            lines.append(f"  {mark} {tag_name(tag):28s} {value}")
    lines.append("legend: ' ' kept, '-' removed by scrub, '~' rewritten by scrub")
    return "\n".join(lines)
