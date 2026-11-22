from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coverslip import browse, cli
from coverslip.cli import EXIT_ERROR, main, with_default_command
from coverslip.patches import PatchError

from conftest import write_pyramid
from test_inplace import write_aperio_like


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1, lines
    return lines[0]


def test_unknown_level_is_a_one_line_error(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    out = tmp_path / "out.tif"
    status = main(["redact", str(src), str(out), "-r", "25,12,30,22@105x76"])
    assert status == EXIT_ERROR
    assert error_line(capsys) == (
        "coverslip redact: error: No pyramid level is 105x76; "
        "this file has 420x300, 210x150, 105x75"
    )
    assert not out.exists()


def test_region_outside_image_is_a_one_line_error(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    args = ["redact", str(src), str(tmp_path / "out.tif"), "-r", "5000,5000,9,9"]
    assert main(args) == EXIT_ERROR
    assert "lies outside the 420x300 image" in error_line(capsys)


def test_missing_and_non_tiff_inputs_are_one_line_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(tmp_path / "missing.svs")]) == EXIT_ERROR
    assert "No such file" in error_line(capsys)
    text = tmp_path / "notes.txt"
    text.write_text("not a slide")
    assert main(["inspect", str(text)]) == EXIT_ERROR
    assert error_line(capsys).startswith("coverslip inspect: error: ")


def test_existing_output_is_a_one_line_error(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    out = tmp_path / "out.tif"
    out.write_bytes(b"keep me")
    assert main(["redact", str(src), str(out), "-r", "1,1,9,9"]) == EXIT_ERROR
    assert "exists" in error_line(capsys)
    assert out.read_bytes() == b"keep me"


def test_unexpected_errors_keep_their_traceback(
    tmp_path: Path, rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_planner(*args: object, **kwargs: object) -> None:
        raise PatchError("Edits overlap at offset 8")

    monkeypatch.setattr("coverslip.cli.redact_in_place", broken_planner)
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    with pytest.raises(PatchError):
        main(["redact", str(src), "--in-place", "-r", "1,1,9,9"])


def test_inspect_lists_levels_and_tags(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    assert main(["inspect", str(src)]) == 0
    out = capsys.readouterr().out
    assert "420x300" in out and "105x75" in out
    assert "ImageWidth" not in out
    assert main(["inspect", str(src), "--tags"]) == 0
    assert "ImageWidth" in capsys.readouterr().out


def test_a_bare_file_means_browse() -> None:
    assert with_default_command(["slide.svs"]) == ["browse", "slide.svs"]
    assert with_default_command(["-v", "slide.svs"]) == ["-v", "browse", "slide.svs"]


def test_commands_and_options_are_left_alone() -> None:
    assert with_default_command(["inspect", "slide.svs"]) == ["inspect", "slide.svs"]
    assert with_default_command(["-v", "show", "a.svs"]) == ["-v", "show", "a.svs"]
    assert with_default_command(["--version"]) == ["--version"]
    assert with_default_command([]) == []


def test_browse_without_a_terminal_prints_the_inspect_report(
    tmp_path: Path, rgb_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    assert main([str(src)]) == 0
    browsed = capsys.readouterr().out
    assert main(["inspect", str(src)]) == 0
    assert browsed == capsys.readouterr().out


def test_browse_prints_the_command_for_the_plan(
    tmp_path: Path,
    rgb_image: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    plan = browse.Plan(str(src), scrub=True, regions=["1,2,3,4"])
    fake_tui = SimpleNamespace(run_browser=lambda layout, show, errors: plan)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr(browse, "load_tui", lambda: fake_tui)
    assert main(["browse", str(src)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "redact with:",
        f"coverslip redact {src} OUTPUT -r 1,2,3,4 --scrub",
    ]


def test_browse_left_without_quitting_prints_nothing(
    tmp_path: Path,
    rgb_image: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    fake_tui = SimpleNamespace(run_browser=lambda layout, show, errors: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr(browse, "load_tui", lambda: fake_tui)
    assert main(["browse", str(src)]) == 0
    assert capsys.readouterr().out == ""


def test_redact_with_nothing_to_do_is_refused(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    src = write_pyramid(tmp_path / "src.tif", rgb_image)
    with pytest.raises(SystemExit, match="nothing to do"):
        main(["redact", str(src), str(tmp_path / "out.tif")])


def test_redact_accepts_a_drop_flag_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write_aperio_like(tmp_path / "src.svs")
    out = tmp_path / "out.svs"
    assert main(["redact", str(src), str(out), "--drop-macro"]) == 0
    assert "dropped macro" in capsys.readouterr().out
    assert main(["verify", str(src), str(out)]) == 0
