import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import tifffile

from coverslip import info, view_cli, viewer
from coverslip.cli import EXIT_ERROR, main
from coverslip.model import Region

from conftest import (
    load_layout,
    make_image,
    requires_sample,
    sample_path,
    write_pyramid,
)
from test_inplace import write_aperio_like


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(viewer, "run", lambda: None)
    yield
    plt.close("all")


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1, lines
    return lines[0]


def figure_titles() -> list[str]:
    return [plt.figure(n).axes[0].get_title() for n in plt.get_fignums()]


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
    monkeypatch.setattr(view_cli.chunks, "DEFAULT_MAX_PIXELS", 1000)
    assert main(["show", str(lzw_pyramid), "--level", "0"]) == EXIT_ERROR
    assert "more than the limit" in error_line(capsys)
    assert main(["show", str(lzw_pyramid), "--level", "0", "--force"]) == 0


def test_show_missing_label(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "--label"]) == EXIT_ERROR
    assert error_line(capsys) == "coverslip show: error: This file has no label image"


def test_show_unknown_level(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "--tile-level", "5"]) == EXIT_ERROR
    assert error_line(capsys) == (
        "coverslip show: error: No level 5; levels are "
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
    assert error_line(capsys) == f"coverslip tile: error: {message}"


def test_grid(lzw_pyramid: Path) -> None:
    assert main(["grid", str(lzw_pyramid), "--level", "1"]) == 0
    assert figure_titles() == ["lzw.tif  level 1  4x3 tiles"]


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", str(tmp_path / "nope.tif")]) == EXIT_ERROR
    assert error_line(capsys).startswith("coverslip show: error: ")


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
    layout = view_cli.load(str(path), verbose=False)
    image_layout = layout.images[0]
    roles = [lvl.role.value for lvl in image_layout.levels]
    assert roles == ["full", "level", "thumbnail"]
    assert view_cli.default_view_level(image_layout) == 1


def test_default_level_falls_back_to_thumbnail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "strips.tif"
    image = make_image(1500, 3000)
    with tifffile.TiffWriter(path) as writer:
        writer.write(image, tile=(256, 256), compression="lzw")
        writer.write(image[::2, ::2], subfiletype=1)
    monkeypatch.setattr(view_cli.chunks, "DEFAULT_MAX_PIXELS", 2000 * 2000)
    layout = view_cli.load(str(path), verbose=False)
    assert view_cli.default_view_level(layout.images[0]) == 1


def test_cli_import_does_not_need_matplotlib() -> None:
    code = "import sys; import coverslip.cli; " "sys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_view_without_matplotlib_is_a_one_line_error(
    lzw_pyramid: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "coverslip.viewer")
    assert main(["show", str(lzw_pyramid)]) == EXIT_ERROR
    assert error_line(capsys) == (
        "coverslip show: error: viewing needs matplotlib; install coverslip[view]"
    )


def capture_level_view(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the LevelView that show builds, and let a fake run() use it."""
    views: list = []
    original = viewer.level_figure

    def recording(*args, **kwargs):
        views.append(original(*args, **kwargs))
        return views[-1]

    monkeypatch.setattr(viewer, "level_figure", recording)
    return views


def test_show_prints_redact_command_for_drawn_regions(
    lzw_pyramid: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    views = capture_level_view(monkeypatch)

    def draw_then_close() -> None:
        views[0].add_region(Region(10, 20, 41, 40))

    monkeypatch.setattr(viewer, "run", draw_then_close)
    args = ["show", str(lzw_pyramid), "--level", "1", "-r", "1,1,5,5"]
    assert main(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-3:] == [
        "-r 10,20,41,40@210x150",
        "redact with:",
        f"coverslip redact {lzw_pyramid} OUTPUT -r 1,1,5,5 -r 10,20,41,40@210x150",
    ]


def test_show_without_drawing_prints_no_command(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid)]) == 0
    assert "redact with:" not in capsys.readouterr().out


def test_show_previews_regions(
    lzw_pyramid: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    views = capture_level_view(monkeypatch)
    args = ["show", str(lzw_pyramid), "--level", "2", "-r", "10,20,41,40@210x150"]
    assert main(args) == 0
    # the region selector owns a (hidden) rectangle too; previews are dashed
    [patch] = [p for p in views[0].ax.patches if p.get_linestyle() == "--"]
    assert patch.get_xy() == (5, 10) and patch.get_width() == 21


def test_show_region_outside_level_frame_is_an_error(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "-r", "1,1,5,5@99x99"]) == EXIT_ERROR
    assert "No pyramid level is 99x99" in error_line(capsys)


def test_show_region_with_label_is_an_error(
    lzw_pyramid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", str(lzw_pyramid), "--label", "-r", "1,1,5,5"]) == EXIT_ERROR
    assert error_line(capsys) == (
        "coverslip show: error: --region previews apply to pyramid levels, "
        "not --label"
    )


def test_show_rejects_malformed_region(lzw_pyramid: Path) -> None:
    with pytest.raises(SystemExit):
        main(["show", str(lzw_pyramid), "-r", "1,2,3"])


def test_show_entry_opens_the_level_of_that_row(lzw_pyramid: Path) -> None:
    layout = load_layout(lzw_pyramid)
    entry = info.layout_entries(layout)[1]
    drawn = view_cli.show_entry(str(lzw_pyramid), layout, entry, ["10,10,40,40"])
    assert drawn == []
    assert figure_titles() == ["lzw.tif  level 1  210x150"]


def test_show_entry_opens_an_associated_image(tmp_path: Path) -> None:
    path = write_aperio_like(tmp_path / "src.svs")
    layout = load_layout(path)
    entries = {entry.name: entry for entry in info.layout_entries(layout)}
    assert view_cli.show_entry(str(path), layout, entries["macro"]) == []
    assert figure_titles() == ["src.svs  macro  120x50"]
