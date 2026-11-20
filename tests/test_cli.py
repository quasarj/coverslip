import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import tifffile

from tiffview import cli, viewer
from tiffview.cli import EXIT_ERROR, main

from conftest import make_image, requires_sample, sample_path, write_pyramid


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(viewer, "run", lambda: None)
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def restore_layout_log_level() -> Iterator[None]:
    """main() quiets tiffview.layout; don't let that leak into other tests."""
    layout_logger = logging.getLogger("tiffview.layout")
    level = layout_logger.level
    yield
    layout_logger.setLevel(level)


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1, lines
    return lines[0]


def figure_titles() -> list[str]:
    return [plt.figure(n).axes[0].get_title() for n in plt.get_fignums()]


def test_info(lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(lzw_pyramid)]) == 0
    out = capsys.readouterr().out
    assert "420x300" in out and "105x75" in out
    assert "ImageWidth" not in out
    assert main(["info", str(lzw_pyramid), "--tags"]) == 0
    assert "ImageWidth" in capsys.readouterr().out


def test_show_default_level_is_smallest_big_enough(tmp_path: Path) -> None:
    path = write_pyramid(tmp_path / "big.tif", make_image(1200, 2400), levels=3)
    assert main(["show", str(path)]) == 0
    assert figure_titles() == ["big.tif  level 1  1200x600"]


def test_show_small_image_defaults_to_full(lzw_pyramid: Path) -> None:
    assert main(["show", str(lzw_pyramid)]) == 0
    assert figure_titles() == ["lzw.tif  level 0  420x300"]


def test_show_level_and_tile_level(lzw_pyramid: Path) -> None:
    args = ["show", str(lzw_pyramid), "--level", "2", "--tile-level", "1", "--grid"]
    assert main(args) == 0
    assert figure_titles() == ["lzw.tif  level 2  105x75"]


def test_show_too_large_needs_force(
    lzw_pyramid: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.chunks, "DEFAULT_MAX_PIXELS", 1000)
    assert main(["show", str(lzw_pyramid), "--level", "0"]) == EXIT_ERROR
    assert "more than the limit" in error_line(capsys)
    assert main(["show", str(lzw_pyramid), "--level", "0", "--force"]) == 0


def test_show_missing_label(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "--label"]) == EXIT_ERROR
    assert error_line(capsys) == "tiffview show: error: This file has no label image"


def test_show_unknown_level(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "--tile-level", "5"]) == EXIT_ERROR
    assert error_line(capsys) == (
        "tiffview show: error: No level 5; levels are "
        "0: 420x300, 1: 210x150, 2: 105x75"
    )


def test_tile_by_col_row(lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tile", str(lzw_pyramid), "--col", "2", "--row", "1"]) == 0
    assert capsys.readouterr().out.startswith("level 0 tile (2,1) at 128,64")
    assert len(plt.get_fignums()) == 1


def test_tile_at_point_on_level(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["tile", str(lzw_pyramid), "--level", "1", "--at", "130,70"]) == 0
    assert capsys.readouterr().out.startswith("level 1 tile (2,1) at 128,64")


@pytest.mark.parametrize(
    "extra,message",
    [
        ([], "give --col and --row, or --at X,Y"),
        (["--col", "1"], "give --col and --row, or --at X,Y"),
        (["--at", "1,1", "--col", "1"], "give either --at or --col/--row, not both"),
        (["--at", "500,1"], "Point 500,1 is outside the 420x300 level"),
        (["--col", "7", "--row", "0"], "Tile (7,0) is outside the 7x5 tile grid"),
    ],
)
def test_tile_bad_selection(
    lzw_pyramid: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    assert main(["tile", str(lzw_pyramid), *extra]) == EXIT_ERROR
    assert error_line(capsys) == f"tiffview tile: error: {message}"


def test_grid(lzw_pyramid: Path) -> None:
    assert main(["grid", str(lzw_pyramid), "--level", "1"]) == 0
    assert figure_titles() == ["lzw.tif  level 1  4x3 tiles"]


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(tmp_path / "nope.tif")]) == EXIT_ERROR
    assert error_line(capsys).startswith("tiffview info: error: ")


@requires_sample("CMU-1-Small-Region.svs")
@pytest.mark.parametrize(
    "flag,role",
    [("--label", "label"), ("--macro", "macro"), ("--thumbnail", "thumbnail")],
)
def test_show_associated_images(flag: str, role: str) -> None:
    path = sample_path("CMU-1-Small-Region.svs")
    assert main(["show", path, flag]) == 0
    [title] = figure_titles()
    assert title.startswith(f"CMU-1-Small-Region.svs  {role}  ")


def test_default_level_prefers_tiled_over_thumbnail(tmp_path: Path) -> None:
    path = tmp_path / "with_thumb.tif"
    image = make_image(1500, 3000)
    with tifffile.TiffWriter(path) as writer:
        writer.write(image, tile=(256, 256), compression="lzw")
        writer.write(image[::2, ::2], subfiletype=1, tile=(256, 256))
        writer.write(image[::2, ::2][::2, ::2], subfiletype=1)
    _, layout = cli.load(str(path))
    image_layout = layout.images[0]
    roles = [lvl.role.value for lvl in image_layout.levels]
    assert roles == ["full", "level", "thumbnail"]
    assert cli.default_view_level(image_layout) == 1


def test_default_level_falls_back_to_thumbnail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "strips.tif"
    image = make_image(1500, 3000)
    with tifffile.TiffWriter(path) as writer:
        writer.write(image, tile=(256, 256), compression="lzw")
        writer.write(image[::2, ::2], subfiletype=1)
    monkeypatch.setattr(cli.chunks, "DEFAULT_MAX_PIXELS", 2000 * 2000)
    _, layout = cli.load(str(path))
    assert cli.default_view_level(layout.images[0]) == 1
