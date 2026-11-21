from pathlib import Path

import numpy as np
import pytest
import tifffile

from coverslip import layout as layout_mod
from coverslip import scrub
from coverslip.model import Region
from coverslip.redact import RedactionError, redact_file
from coverslip.verify import verify_file

from conftest import make_image, requires_openslide, requires_svs, sample_svs_paths

APERIO_DESC = (
    "Aperio Image Library v12.0.11 \r\n22352x18229 [0,100 21911x18129] (240x240) "
    "JPEG/RGB Q=70|AppMag = 20|StripeWidth = 2032|ScanScope ID = SS1553|"
    "Filename = 120098|Date = 12/14/17|Time = 13:23:18|Time Zone = GMT-05:00|"
    "User = 759bed0e|MPP = 0.4942|Left = 19.301331|DSR ID = aperio04|"
    "ImageID = 120098|OriginalWidth = 22352|ICC Profile = ScanScope v1"
)

OME_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" Creator="test" '
    'UUID="urn:uuid:1" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 x.xsd">'
    '<Experimenter ID="Experimenter:0" UserName="jdoe"/>'
    '<Instrument ID="Instrument:0"><Microscope SerialNumber="SN123" Model="M"/></Instrument>'
    '<Image ID="Image:0" Name="patient slide"><AcquisitionDate>2011-05-31T09:33:14</AcquisitionDate>'
    "<Description>case 42</Description>"
    '<ExperimenterRef ID="Experimenter:0"/>'
    '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" SizeC="3" SizeT="1" SizeX="10" SizeY="10" SizeZ="1" Type="uint8">'
    '<Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="3"/>'
    '<TiffData IFD="0" PlaneCount="1"><UUID FileName="a.ome.tiff">urn:uuid:1</UUID></TiffData></Pixels>'
    '<AnnotationRef ID="Annotation:0"/></Image>'
    '<StructuredAnnotations><XMLAnnotation ID="Annotation:0"><Value><OriginalMetadata>'
    "<Key>Filename</Key><Value>secret.svs</Value></OriginalMetadata></Value></XMLAnnotation>"
    "</StructuredAnnotations></OME>"
)


def test_aperio_description_keeps_header_and_allowed_keys() -> None:
    cleaned, removed = scrub.scrub_aperio_description(APERIO_DESC)
    assert cleaned.startswith("Aperio Image Library v12.0.11 \r\n22352x18229")
    assert "|AppMag = 20|" in cleaned and "|MPP = 0.4942|" in cleaned
    assert cleaned.endswith("|ICC Profile = ScanScope v1")
    for key in (
        "Filename",
        "Date",
        "Time",
        "User",
        "ScanScope ID",
        "DSR ID",
        "ImageID",
    ):
        assert key in removed
        assert f"{key} =" not in cleaned
    assert scrub.description_problems(cleaned, "aperio") == []
    assert scrub.description_problems(APERIO_DESC, "aperio")


def test_ome_xml_scrub_removes_identifiers_and_keeps_structure() -> None:
    cleaned, removed = scrub.scrub_ome_xml(OME_XML)
    for word in (
        "jdoe",
        "SN123",
        "patient slide",
        "2011-05-31",
        "case 42",
        "secret.svs",
    ):
        assert word not in cleaned
    assert removed["Experimenter"] == 1 and removed["StructuredAnnotations"] == 1
    assert removed["Image@Name"] == 1 and removed["Microscope@SerialNumber"] == 1
    assert (
        "<TiffData" in cleaned and 'Name="DAPI"' in cleaned and "urn:uuid:1" in cleaned
    )
    assert "ns0:" not in cleaned and 'xmlns="http://www.openmicroscopy.org' in cleaned
    assert "xsi:schemaLocation" in cleaned
    assert cleaned.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert scrub.ome_xml_problems(cleaned) == []
    assert scrub.ome_xml_problems(OME_XML)


def test_scrub_removes_non_structural_tags(tmp_path: Path) -> None:
    src = tmp_path / "src.tif"
    image = make_image(64, 96)
    extratags = [
        (315, "s", 0, "Dr. Who", False),  # Artist
        (316, "s", 0, "workstation-7", False),  # HostComputer
        (700, "B", 0, b"<x:xmpmeta/>", False),  # XMP
        (269, "s", 0, "case-42.tif", False),  # DocumentName
    ]
    tifffile.imwrite(
        src,
        image,
        compression="lzw",
        software="unit-test",
        datetime=True,
        description="free text about a patient",
        metadata=None,
        extratags=extratags,
    )
    out = tmp_path / "out.tif"
    report = redact_file(str(src), str(out), [], scrub=True)
    assert report.scrub is not None
    assert set(report.scrub.removed_tags) == {306, 315, 316, 700, 269}
    assert report.scrub.descriptions_removed == 1
    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert 305 in page.tags and page.tags[305].value == "unit-test"
        for code in (306, 315, 316, 700, 269, 270):
            assert code not in page.tags
        assert np.array_equal(page.asarray(), image)
    assert verify_file(str(src), str(out), [], scrubbed=True).ok
    assert not verify_file(str(src), str(src), [], scrubbed=True).ok


def test_keep_tag_exempts_a_tag(tmp_path: Path) -> None:
    src = tmp_path / "src.tif"
    tifffile.imwrite(src, make_image(32, 32), datetime=True)
    out = tmp_path / "out.tif"
    redact_file(str(src), str(out), [], scrub=True, keep_tags=[306])
    with tifffile.TiffFile(out) as tf:
        assert 306 in tf.pages[0].tags
    assert verify_file(str(src), str(out), [], scrubbed=True, keep_tags=[306]).ok
    assert not verify_file(str(src), str(out), [], scrubbed=True).ok


def test_parse_keep_tag_accepts_names() -> None:
    assert scrub.parse_keep_tag("DateTime") == 306
    assert scrub.parse_keep_tag("306") == 306
    with pytest.raises(ValueError):
        scrub.parse_keep_tag("NoSuchTag")


def test_shaped_description_is_kept(tmp_path: Path) -> None:
    src = tmp_path / "src.tif"
    tifffile.imwrite(src, make_image(32, 32))  # tifffile writes {"shape": ...}
    out = tmp_path / "out.tif"
    redact_file(str(src), str(out), [], scrub=True)
    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].description.startswith('{"shape"')


def test_regions_required_without_scrub(tmp_path: Path) -> None:
    src = tmp_path / "src.tif"
    tifffile.imwrite(src, make_image(32, 32))
    with pytest.raises(RedactionError):
        redact_file(str(src), str(tmp_path / "out.tif"), [])


def test_ome_file_scrub_keeps_it_readable(tmp_path: Path) -> None:
    src = tmp_path / "src.ome.tif"
    image = make_image(128, 160)
    with tifffile.TiffWriter(src, ome=True) as writer:
        writer.write(
            image,
            tile=(64, 64),
            compression="lzw",
            subifds=1,
            metadata={
                "axes": "YXS",
                "Name": "patient 42",
                "Description": "confidential",
            },
        )
        writer.write(image[::2, ::2], subfiletype=1, tile=(64, 64), compression="lzw")
    out = tmp_path / "out.ome.tif"
    region = Region(10, 10, 40, 40)
    report = redact_file(str(src), str(out), [region], scrub=True)
    assert (
        report.scrub is not None and report.scrub.removed_ome_items["Image@Name"] == 1
    )
    with tifffile.TiffFile(out) as tf:
        assert tf.is_ome
        assert (
            "patient 42" not in tf.ome_metadata
            and "confidential" not in tf.ome_metadata
        )
        series = tf.series[0]
        assert series.shape == (128, 160, 3) and len(series.levels) == 2
        assert series.asarray()[10:50, 10:50].max() == 0
    assert verify_file(str(src), str(out), [region], scrubbed=True).ok


@requires_svs
@requires_openslide
def test_aperio_scrub_keeps_openslide_properties(tmp_path: Path) -> None:
    import openslide

    src = min(sample_svs_paths(), key=lambda p: Path(p).stat().st_size)
    out = tmp_path / "out.svs"
    redact_file(src, str(out), [], scrub=True, drop_label=True, drop_macro=True)
    audit = verify_file(src, str(out), [], scrubbed=True)
    assert audit.ok, audit.summary()
    with openslide.OpenSlide(src) as a, openslide.OpenSlide(str(out)) as b:
        for key in ("openslide.mpp-x", "openslide.objective-power", "aperio.AppMag"):
            assert a.properties[key] == b.properties[key]
        for key in (
            "aperio.Filename",
            "aperio.Date",
            "aperio.User",
            "aperio.ScanScope ID",
        ):
            assert key in a.properties and key not in b.properties
        assert b.level_dimensions == a.level_dimensions
    layout = layout_mod.discover(layout_mod.read_info(str(out)))
    assert not layout.label and not layout.macro


def test_imagej_structural_description_is_kept_but_free_text_is_not() -> None:
    report = scrub.ScrubReport()
    header = "ImageJ=1.53\nhyperstack=true\nimages=3\nchannels=3\nslices=1\nframes=1\nunit=micron"
    assert scrub.scrub_description(header, "generic", report) == header
    assert (
        scrub.scrub_description("ImageJ=1.53\nInfo=patient 42", "generic", report)
        is None
    )
    assert report.descriptions_removed == 1
    assert scrub.description_problems(header, "generic") == []
