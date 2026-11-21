from pathlib import Path

import numpy as np
import pytest

from coverslip.cli import EXIT_ERROR, main
from coverslip.patches import PatchError

from conftest import write_pyramid


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
