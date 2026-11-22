"""The browser, driven headless through Textual's pilot.

The pilot is asynchronous; each test wraps its session in ``asyncio.run`` so
the suite needs no async pytest plugin.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from textual.pilot import Pilot
from textual.widgets import Static

from coverslip import browse
from coverslip.chunks import LevelTooLargeError
from coverslip.info import LayoutEntry
from coverslip.model import Role
from coverslip.tui import DROP_COLUMN, LevelTable, SlideBrowser

from conftest import load_layout
from test_inplace import write_aperio_like

Session = Callable[[Pilot[browse.Plan]], Awaitable[None]]

LABEL_ROW = 2
"""Rows of the Aperio-like fixture: full, thumbnail, label, macro."""


class RecordingViewer:
    """Stands in for the matplotlib viewer."""

    def __init__(self, drawn: Sequence[str] = ()) -> None:
        self.drawn = list(drawn)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, entry: LayoutEntry, regions: Sequence[str]) -> list[str]:
        self.calls.append((entry.name, list(regions)))
        return list(self.drawn)


def too_large(entry: LayoutEntry, regions: Sequence[str]) -> list[str]:
    raise LevelTooLargeError("level is more than the limit")


def browser(path: Path, show: Callable[..., list[str]] | None = None) -> SlideBrowser:
    viewer = show if show is not None else RecordingViewer()
    return SlideBrowser(load_layout(path), viewer, (LevelTooLargeError,))


def drive(app: SlideBrowser, session: Session) -> browse.Plan | None:
    async def run() -> None:
        async with app.run_test(size=(160, 30)) as pilot:
            await session(pilot)

    asyncio.run(run())
    return app.return_value


async def move_to(pilot: Pilot[browse.Plan], row: int) -> None:
    await pilot.press(*["down"] * row)


def drop_marks(app: SlideBrowser) -> list[str]:
    table = app.query_one(LevelTable)
    return [
        str(table.get_cell(str(index), DROP_COLUMN))
        for index in range(len(app.entries))
    ]


def plan_text(app: SlideBrowser) -> str:
    return str(app.query_one("#plan", Static).content)


def test_table_lists_the_inspect_rows_with_default_flags(tmp_path: Path) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"))

    async def session(pilot: Pilot[browse.Plan]) -> None:
        assert [entry.name for entry in app.entries] == ["0", "1", "label", "macro"]
        assert app.query_one(LevelTable).row_count == 4
        assert drop_marks(app) == ["", "", "drop", "drop"]
        assert "--drop-label --drop-macro" in plan_text(app)

    drive(app, session)


def test_quit_returns_the_default_plan(tmp_path: Path) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"))

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await pilot.press("q")

    plan = drive(app, session)
    assert plan is not None
    assert plan.drop == {Role.LABEL, Role.MACRO}
    assert plan.scrub


def test_d_unflags_the_label(tmp_path: Path) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"))

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await move_to(pilot, LABEL_ROW)
        await pilot.press("d")
        assert drop_marks(app) == ["", "", "", "drop"]
        assert "--drop-label" not in plan_text(app)
        await pilot.press("q")

    plan = drive(app, session)
    assert plan is not None and plan.drop == {Role.MACRO}


def test_d_on_a_pyramid_level_changes_nothing(tmp_path: Path) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"))

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await pilot.press("d")
        assert drop_marks(app) == ["", "", "drop", "drop"]

    drive(app, session)


def test_s_toggles_scrub_and_p_restores_the_profile(tmp_path: Path) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"))

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await pilot.press("s")
        assert not app.plan.scrub
        assert browse.DROP_NEEDS_MORE in plan_text(app)
        await move_to(pilot, LABEL_ROW)
        await pilot.press("d", "p")
        assert app.plan.scrub
        assert app.plan.drop == {Role.LABEL, Role.MACRO}

    drive(app, session)


def test_enter_shows_the_row_and_keeps_drawn_regions(tmp_path: Path) -> None:
    viewer = RecordingViewer(drawn=["1,2,3,4@96x64"])
    app = browser(write_aperio_like(tmp_path / "src.svs"), viewer)

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await pilot.press("j", "enter")
        assert viewer.calls == [("1", [])]
        assert "regions drawn: 1" in plan_text(app)
        await pilot.press("enter")
        assert viewer.calls[1] == ("1", ["1,2,3,4@96x64"])
        await pilot.press("u", "q")

    plan = drive(app, session)
    assert plan is not None and plan.regions == ["1,2,3,4@96x64"]


def test_a_viewer_error_is_reported_and_the_browser_keeps_running(
    tmp_path: Path,
) -> None:
    app = browser(write_aperio_like(tmp_path / "src.svs"), too_large)

    async def session(pilot: Pilot[browse.Plan]) -> None:
        await pilot.press("enter")
        assert [note.message for note in app._notifications] == [
            "level is more than the limit"
        ]
        await pilot.press("q")

    assert drive(app, session) is not None
