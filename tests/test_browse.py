from pathlib import Path

import pytest
import tifffile

from coverslip import browse
from coverslip.browse import Plan
from coverslip.model import Role

from conftest import load_layout, make_image
from test_inplace import write_aperio_like


def write_ome_with_macro(path: Path) -> Path:
    main = make_image(256, 320)
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(
            make_image(120, 80, seed=3),
            compression="lzw",
            metadata={"axes": "YXS", "Name": "macro"},
        )
        writer.write(main, tile=(64, 64), compression="lzw", metadata={"axes": "YXS"})
    return path


def test_default_profile_drops_label_and_macro_of_aperio(tmp_path: Path) -> None:
    layout = load_layout(write_aperio_like(tmp_path / "src.svs"))
    plan = browse.default_plan(layout)
    assert plan.drop == {Role.LABEL, Role.MACRO}
    assert plan.scrub
    assert plan.regions == []


def test_default_profile_only_scrubs_a_file_without_associated_images(
    lzw_pyramid: Path,
) -> None:
    plan = browse.default_plan(load_layout(lzw_pyramid))
    assert plan.drop == set()
    assert browse.command(plan) == f"coverslip scrub {lzw_pyramid} OUTPUT"


def test_ome_macro_cannot_be_dropped(tmp_path: Path) -> None:
    layout = load_layout(write_ome_with_macro(tmp_path / "macro.ome.tif"))
    assert [ifd.role for ifd in layout.macro] == [Role.MACRO]
    assert "OME-TIFF" in browse.undroppable_reason(layout, layout.macro[0])
    assert browse.default_plan(layout).drop == set()


def test_pyramid_levels_cannot_be_dropped(tmp_path: Path) -> None:
    layout = load_layout(write_aperio_like(tmp_path / "src.svs"))
    full, thumbnail = layout.images[0].levels
    assert "full image cannot be dropped" in browse.undroppable_reason(layout, full)
    assert "thumbnail image cannot be dropped" in browse.undroppable_reason(
        layout, thumbnail
    )
    assert browse.undroppable_reason(layout, layout.label[0]) is None
    assert browse.undroppable_reason(layout, layout.macro[0]) is None


def test_toggle_drop_flags_and_unflags_a_role() -> None:
    plan = Plan("slide.svs", drop={Role.LABEL, Role.MACRO})
    browse.toggle_drop(plan, Role.LABEL)
    assert plan.drop == {Role.MACRO}
    browse.toggle_drop(plan, Role.LABEL)
    assert plan.drop == {Role.LABEL, Role.MACRO}


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (
            Plan("slide.svs", drop={Role.LABEL, Role.MACRO}, scrub=True),
            "coverslip scrub slide.svs OUTPUT --drop-label --drop-macro",
        ),
        (
            Plan("slide.svs", drop={Role.MACRO}, scrub=True),
            "coverslip scrub slide.svs OUTPUT --drop-macro",
        ),
        (
            Plan("slide.svs", drop={Role.MACRO}, scrub=True, regions=["1,2,3,4@9x9"]),
            "coverslip redact slide.svs OUTPUT -r 1,2,3,4@9x9 --drop-macro --scrub",
        ),
        (
            Plan("slide.svs", regions=["1,2,3,4", "5,6,7,8"]),
            "coverslip redact slide.svs OUTPUT -r 1,2,3,4 -r 5,6,7,8",
        ),
        (
            Plan("my slide.svs", scrub=True),
            "coverslip scrub 'my slide.svs' OUTPUT",
        ),
    ],
)
def test_command(plan: Plan, expected: str) -> None:
    assert browse.command(plan) == expected
    assert browse.describe(plan) == expected


def test_nothing_flagged_has_no_command() -> None:
    plan = Plan("slide.svs")
    assert browse.command(plan) is None
    assert browse.describe(plan) == browse.NOTHING_TO_DO


def test_drops_alone_have_no_command() -> None:
    """redact refuses to drop images without scrubbing or a region."""
    plan = Plan("slide.svs", drop={Role.MACRO})
    assert browse.command(plan) is None
    assert browse.describe(plan) == browse.DROP_NEEDS_MORE


def test_missing_textual_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> None:
        raise ModuleNotFoundError("No module named 'textual'", name="textual")

    monkeypatch.setattr(browse.importlib, "import_module", fail)
    with pytest.raises(browse.TuiUnavailableError, match=r"coverslip\[tui\]"):
        browse.load_tui()
