"""The browser: the inspect report as a table you can scroll, open and flag.

Needs Textual, an optional dependency (``coverslip[tui]``), so this module is
imported only when the browser runs. What the keys decide lives in
:mod:`coverslip.browse`; this module only draws it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import DataTable, Footer, Static

from coverslip import browse, info
from coverslip.info import LayoutEntry
from coverslip.model import Layout

ShowEntry = Callable[[LayoutEntry, Sequence[str]], list[str]]
"""Opens one row in the viewer, previewing the regions given, and returns the
regions drawn in it once its windows are closed."""

DROP_COLUMN = "drop"
DROP_MARK = Text("drop", style="bold red")
KEEP_MARK = Text("")


class LevelTable(DataTable[Text | str]):
    """The inspect table; Enter is advertised in the footer."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select_cursor", "Show image"),
        Binding("j", "cursor_down", "Cursor down", show=False),
        Binding("k", "cursor_up", "Cursor up", show=False),
    ]


class SlideBrowser(App[browse.Plan]):
    """Browse one slide; exits with the plan the user settled on."""

    CSS = """
    #heading { padding: 0 1; height: auto; }
    #levels { height: 1fr; }
    #plan { padding: 0 1; height: auto; border-top: solid $primary; }
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("d", "toggle_drop", "Drop/keep"),
        Binding("s", "toggle_scrub", "Scrub on/off"),
        Binding("p", "default_profile", "Default profile"),
        Binding("u", "undo_region", "Undo region"),
        Binding("q", "finish", "Quit"),
    ]

    def __init__(
        self,
        layout: Layout,
        show: ShowEntry,
        expected_errors: tuple[type[Exception], ...],
    ) -> None:
        super().__init__()
        self.layout = layout
        self.entries = info.layout_entries(layout)
        self.plan = browse.default_plan(layout)
        self._show = show
        self._expected_errors = expected_errors

    def compose(self) -> ComposeResult:
        heading = f"{self.layout.path}\n{info.format_line(self.layout)}"
        yield Static(heading, id="heading", markup=False)
        yield LevelTable(id="levels", cursor_type="row")
        yield Static(id="plan", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(LevelTable)
        table.add_column(DROP_COLUMN, key=DROP_COLUMN)
        for title in info.HEADER:
            table.add_column(title)
        for index, entry in enumerate(self.entries):
            cells = info.row_cells(info.summarize_entry(entry))
            table.add_row(KEEP_MARK, *cells, key=str(index))
        table.focus()
        self._refresh_plan()

    # --- drawing the plan ---------------------------------------------------

    def _refresh_plan(self) -> None:
        table = self.query_one(LevelTable)
        for index, entry in enumerate(self.entries):
            mark = DROP_MARK if entry.ifd.role in self.plan.drop else KEEP_MARK
            table.update_cell(str(index), DROP_COLUMN, mark)
        self.query_one("#plan", Static).update(self._plan_text())

    def _plan_text(self) -> str:
        scrub = "on" if self.plan.scrub else "off"
        return (
            f"metadata scrub: {scrub}    regions drawn: {len(self.plan.regions)}\n"
            f"on quit: {browse.describe(self.plan)}"
        )

    def _current_entry(self) -> LayoutEntry:
        return self.entries[self.query_one(LevelTable).cursor_row]

    # --- keys ---------------------------------------------------------------

    def action_toggle_drop(self) -> None:
        ifd = self._current_entry().ifd
        reason = browse.undroppable_reason(self.layout, ifd)
        if reason is not None:
            self.notify(reason, severity="warning")
            return
        browse.toggle_drop(self.plan, ifd.role)
        self._refresh_plan()

    def action_toggle_scrub(self) -> None:
        self.plan.scrub = not self.plan.scrub
        self._refresh_plan()

    def action_default_profile(self) -> None:
        """Back to the default flags; regions already drawn are kept."""
        default = browse.default_plan(self.layout)
        self.plan.drop, self.plan.scrub = default.drop, default.scrub
        self._refresh_plan()

    def action_undo_region(self) -> None:
        if not self.plan.regions:
            self.notify("no regions drawn", severity="warning")
            return
        removed = self.plan.regions.pop()
        self.notify(f"removed region {removed}")
        self._refresh_plan()

    def action_finish(self) -> None:
        self.exit(self.plan)

    # --- viewer -------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        entry = self.entries[event.cursor_row]
        try:
            drawn = self._show_with_terminal_released(entry)
        except self._expected_errors as exc:
            self.notify(str(exc), title=f"cannot show {entry.name}", severity="error")
            return
        self.plan.regions.extend(drawn)
        self._refresh_plan()

    def _terminal_released(self) -> AbstractContextManager[None]:
        """Hand the terminal back, so the viewer's messages can be read.

        A headless driver (the tests) has no terminal to hand back.
        """
        return nullcontext() if self.is_headless else self.suspend()

    def _show_with_terminal_released(self, entry: LayoutEntry) -> list[str]:
        """Block in the viewer until its windows are closed.

        ``App.suspend`` does not take the terminal back when its body raises,
        so a failure is carried out of the block and raised after it.
        """
        failure: Exception | None = None
        drawn: list[str] = []
        with self._terminal_released():
            try:
                drawn = self._show(entry, list(self.plan.regions))
            except Exception as exc:  # noqa: BLE001 - raised again just below
                failure = exc
        if failure is not None:
            raise failure
        return drawn


def run_browser(
    layout: Layout, show: ShowEntry, expected_errors: tuple[type[Exception], ...]
) -> browse.Plan | None:
    """Run the browser; ``None`` if it was left without the quit key."""
    return SlideBrowser(layout, show, expected_errors).run()
