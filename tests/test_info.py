from pathlib import Path

from coverslip import info, layout as layout_mod

from conftest import load_layout


def test_summary_rows(lzw_pyramid: Path) -> None:
    rows = info.summarize_layout(load_layout(lzw_pyramid))
    assert [r.name for r in rows] == ["0", "1", "2"]
    assert [(r.width, r.height) for r in rows] == [(420, 300), (210, 150), (105, 75)]
    assert [r.downsample for r in rows] == [1.0, 2.0, 4.0]
    full = rows[0]
    assert full.role == "full"
    assert (full.chunks_across, full.chunks_down, full.chunk_count) == (7, 5, 35)
    assert full.compression == "LZW"
    assert full.photometric == "RGB"
    assert full.empty_chunks == 0
    assert full.stored_bytes > 0


def test_report_lists_every_level(jpeg_pyramid: Path) -> None:
    report = info.format_report(load_layout(jpeg_pyramid))
    lines = report.splitlines()
    assert lines[1] == "format: classic TIFF  vendor: generic"
    assert lines[3].split()[:3] == ["level", "role", "size"]
    assert lines[4].split()[:3] == ["0", "full", "420x300"]
    assert "JPEG" in lines[4] and "YCbCr" in lines[4]
    assert len(lines) == 7


def test_tags_dump(lzw_pyramid: Path) -> None:
    raw = layout_mod.read_info(str(lzw_pyramid))
    text = info.format_tags(raw, layout_mod.discover(raw))
    assert text.count("ifd@") == 3
    assert "(full):" in text.splitlines()[0]
    assert "ImageWidth" in text
    assert "TileOffsets" in text and "values)" in text


def test_tag_value_formatting() -> None:
    assert info.format_tag_value("a\nb") == "a\\nb"
    assert info.format_tag_value("x" * 100).endswith("... (100 chars)")
    assert info.format_tag_value([1, 2]) == "1, 2"
    assert info.format_tag_value(5) == "5"


def test_human_bytes() -> None:
    assert info.human_bytes(512) == "512 B"
    assert info.human_bytes(2048) == "2.0 KiB"
    assert info.human_bytes(3 * 1024**3) == "3.0 GiB"
