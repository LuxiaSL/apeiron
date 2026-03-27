"""Textual TUI application for Prompt Weaver — Matrix Edition."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Static

from .engine import CombinatorialEngine
from .models import GeneratedPrompt
from .palettes import Palette, palette_for_template
from .store import PromptStore
from .hyperobject.viewport import HyperobjectViewport
from .widgets import (
    EntropyMeter,
    GlitchPrompt,
    HackerLog,
    HistoryLog,
    MatrixBanner,
    MatrixRain,
)

# ── category highlight colors (fixed, independent of palette) ────────────
CATEGORY_STYLES: dict[str, str] = {
    "subject_form": "bold bright_white",
    "material_substance": "bold bright_yellow",
    "texture_density": "bold bright_green",
    "light_behavior": "bold bright_cyan",
    "color_logic": "bold bright_magenta",
    "atmosphere_field": "bold #6688ff",
    "phenomenon_pattern": "bold #ff6644",
    "spatial_logic": "bold #aaffaa",
    "scale_perspective": "bold #ffaa44",
    "temporal_state": "bold #ff88ff",
    "setting_location": "bold #44ffcc",
    "medium_render": "bold #ff8866",
}

LABEL_STYLES: dict[str, str] = {
    "subject_form": "dim",
    "material_substance": "dim yellow",
    "texture_density": "dim green",
    "light_behavior": "dim cyan",
    "color_logic": "dim magenta",
    "atmosphere_field": "dim blue",
    "phenomenon_pattern": "dim red",
    "spatial_logic": "dim",
    "scale_perspective": "dim yellow",
    "temporal_state": "dim magenta",
    "setting_location": "dim cyan",
    "medium_render": "dim red",
}

# ── glitch artifact ──────────────────────────────────────────────────────
ARTIFACT_CHANCE: float = 0.002  # 1 in 500
CORRUPTION_CHARS = "░▒▓█╗╔╚╝║═▀▄▐▌◄►"

# ── milestones ───────────────────────────────────────────────────────────
MILESTONES: frozenset[int] = frozenset(
    {10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000}
)

# ── auto-generate ────────────────────────────────────────────────────────
AUTO_INTERVAL: float = 2.0  # seconds between auto-generated prompts
logger = logging.getLogger(__name__)


def _highlight_prompt(prompt_text: str, components: dict[str, list[str]]) -> Text:
    """Highlight component words in a prompt string with category colors."""
    text = Text(prompt_text)
    for category, words in components.items():
        style = CATEGORY_STYLES.get(category, "bold bright_green")
        for word in words:
            start = 0
            while True:
                idx = prompt_text.find(word, start)
                if idx == -1:
                    break
                text.stylize(style, idx, idx + len(word))
                start = idx + len(word)
    return text


def _corrupt_text(text: str) -> str:
    """Apply visual corruption to prompt text for artifact easter egg."""
    result = list(text)
    for i in range(len(result)):
        roll = random.random()
        if roll < 0.12:
            result[i] = random.choice(CORRUPTION_CHARS)
        elif roll < 0.16:
            result[i] = result[i] * random.randint(2, 4)
        elif roll < 0.20:
            result[i] = ""
    return "".join(result)


class PromptWeaverApp(App[None]):
    """Combinatorial prompt generator TUI — Matrix Edition."""

    TITLE = "promptweaver"
    SUB_TITLE = "// combinatorial prompt generator"

    CSS = """
    Screen {
        background: #000000;
        color: #00ff41;
    }

    MatrixBanner {
        dock: top;
        height: auto;
        background: #000000;
        padding: 0 1;
    }

    #body {
        height: 1fr;
    }

    #main-col {
        width: 3fr;
        margin: 0 1;
    }

    /* ── visual area (rain/hyper/trace) — ABOVE info panels ── */

    #matrix-rain {
        height: 1fr;
        min-height: 6;
    }

    #hyperobject-viewport {
        display: none;
        height: 1fr;
        min-height: 6;
    }

    #hacker-log {
        height: 1fr;
        min-height: 6;
    }

    /* ── info panels — BELOW visual area ── */

    #info-scroll {
        height: auto;
        max-height: 50%;
    }

    #prompt-display {
        margin-bottom: 0;
    }

    #negative-display {
        margin: 0;
    }

    #components-display {
        margin: 0;
    }

    #entropy-display {
        margin-top: 0;
    }

    /* ── sidebar ── */

    #sidebar {
        width: 1fr;
        min-width: 28;
        max-width: 38;
        margin: 0 1 0 0;
    }

    Footer {
        background: #001100;
        color: #00ff41;
    }

    /* ── HYPER-ACTIVE: collapse info, maximize viewport ── */

    Screen.hyper-active MatrixBanner {
        display: none;
    }

    Screen.hyper-active #info-scroll {
        max-height: 8;
    }

    Screen.hyper-active #negative-display {
        display: none;
    }

    Screen.hyper-active #components-display {
        display: none;
    }

    Screen.hyper-active #hyperobject-viewport {
        min-height: 12;
    }
    """

    BINDINGS = [
        Binding("space", "next_prompt", "GENERATE", priority=True),
        Binding("enter", "next_prompt", "GENERATE", show=False, priority=True),
        Binding("t", "cycle_template", "TEMPLATE"),
        Binding("f", "toggle_favorite", "FAV"),
        Binding("a", "toggle_auto", "AUTO"),
        Binding("v", "toggle_hyperobject", "HYPER"),
        Binding("h", "toggle_hacker_log", "TRACE"),
        Binding("c", "copy_prompt", "COPY"),
        Binding("n", "copy_negative", "NEG"),
        Binding("q", "quit_app", "EXIT"),
    ]

    def __init__(
        self,
        db_path: Optional[Path] = None,
        hyper: bool = False,
    ) -> None:
        super().__init__()
        self.engine = CombinatorialEngine()
        self.store = PromptStore(db_path=db_path)
        self.current: Optional[GeneratedPrompt] = None
        self._template_filter: Optional[str] = None
        self._template_idx: int = 0
        self._is_artifact: bool = False
        self._current_palette: Optional[Palette] = None
        self._hacker_visible: bool = False
        self._favorites: set[str] = set()
        self._auto_timer: Optional[Timer] = None
        self._hyper_visible: bool = hyper
        self._start_hyper: bool = hyper
        self._clipboard_command: list[str] | None = None
        self._clipboard_checked: bool = False

    def compose(self) -> ComposeResult:
        yield MatrixBanner()
        with Horizontal(id="body"):
            with Vertical(id="main-col"):
                # Visual area first (hero position — gets the space)
                yield MatrixRain(id="matrix-rain")
                yield HyperobjectViewport(id="hyperobject-viewport")
                yield HackerLog(id="hacker-log")
                # Info panels below (compact, scrollable)
                with VerticalScroll(id="info-scroll"):
                    yield GlitchPrompt(id="prompt-display")
                    yield Static(id="negative-display")
                    yield Static(id="components-display")
                    yield EntropyMeter(id="entropy-display")
            with Vertical(id="sidebar"):
                yield HistoryLog(id="history-log")
        yield Footer()

    def on_mount(self) -> None:
        self._favorites = self.store.get_favorited_hashes()
        self.query_one("#history-log", HistoryLog).set_favorites(self._favorites)

        # Start in hyperobject mode if --hyper flag was passed
        if self._start_hyper:
            rain = self.query_one("#matrix-rain", MatrixRain)
            hyper = self.query_one("#hyperobject-viewport", HyperobjectViewport)
            rain.display = False
            hyper.display = True
            self.screen.add_class("hyper-active")

        self._generate()

    # ── palette ───────────────────────────────────────────────────────

    def _apply_palette(self, palette: Palette) -> None:
        """Push palette to all widgets + update screen chrome."""
        if palette is self._current_palette:
            return
        self._current_palette = palette

        self.screen.styles.color = palette.primary

        self.query_one(MatrixBanner).set_palette(palette)
        self.query_one("#prompt-display", GlitchPrompt).set_palette(palette)
        self.query_one("#history-log", HistoryLog).set_palette(palette)
        self.query_one("#matrix-rain", MatrixRain).set_palette(palette)
        self.query_one("#hacker-log", HackerLog).set_palette(palette)
        self.query_one("#entropy-display", EntropyMeter).set_palette(palette)
        self.query_one("#hyperobject-viewport", HyperobjectViewport).set_palette(palette)

    # ── generation ────────────────────────────────────────────────────

    def _generate(self) -> None:
        self.current = self.engine.generate_unique(
            self.store.seen_hashes,
            template_id=self._template_filter,
        )
        self.store.save(self.current)
        self._is_artifact = random.random() < ARTIFACT_CHANCE
        logger.debug(
            "Generated prompt hash=%s template=%s hyper=%s",
            self.current.hash,
            self.current.template_id,
            self._hyper_visible,
        )
        self._render()

    # ── clipboard helper ──────────────────────────────────────────────

    def _resolve_clipboard_command(self) -> list[str] | None:
        """Detect and cache the first available clipboard command."""
        if self._clipboard_checked:
            return self._clipboard_command

        self._clipboard_checked = True
        for cmd in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
            ["pbcopy"],
        ):
            if shutil.which(cmd[0]):
                self._clipboard_command = cmd
                logger.debug("Using clipboard command: %s", cmd)
                break

        return self._clipboard_command

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard. Returns True on success."""
        cmd = self._resolve_clipboard_command()
        if cmd is None:
            return False

        encoded = text.encode()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=encoded, timeout=1)
            return proc.returncode == 0
        except FileNotFoundError:
            self._clipboard_command = None
            self._clipboard_checked = False
            logger.exception("Clipboard command disappeared: %s", cmd)
            return False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.warning("Clipboard command timed out: %s", cmd)
            return False
        except Exception:
            logger.exception("Clipboard copy failed via %s", cmd)
            return False

    def _copy_to_clipboard_async(self, text: str, *, negative: bool) -> None:
        def worker() -> None:
            success = self._copy_to_clipboard(text)
            self.call_from_thread(self._finish_clipboard_copy, success, negative)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_clipboard_copy(self, success: bool, negative: bool) -> None:
        if success:
            self.notify("negative copied!" if negative else "copied!", timeout=1)
        else:
            self.notify("no clipboard tool found", severity="warning", timeout=2)

    # ── rendering ─────────────────────────────────────────────────────

    def _render(self, *, animate: bool = True) -> None:
        if not self.current:
            return

        p = self.current

        # Apply template-driven palette
        palette = palette_for_template(p.template_id)
        self._apply_palette(palette)

        # ── positive prompt ──────────────────────────────────────────
        star = " \u2605" if p.hash in self._favorites else ""
        glitch_widget = self.query_one("#prompt-display", GlitchPrompt)

        if self._is_artifact:
            corrupted = _corrupt_text(p.positive)
            glitch_widget.set_static(
                Text(corrupted, style="bold #ff0044"),
                title=f"[bold #ff0044]{p.template_id}{star}[/]",
                subtitle="[bold #ff0044]// ARTIFACT DETECTED[/]",
                border_style="#ff0044",
            )
        else:
            highlighted = _highlight_prompt(p.positive, p.components)
            title = f"[bold {palette.primary}]{p.template_id}{star}[/]"
            subtitle = f"[{palette.dim}]0x{p.hash}[/]"
            if animate:
                glitch_widget.decode(
                    plain_text=p.positive,
                    final_renderable=highlighted,
                    title=title,
                    subtitle=subtitle,
                )
            else:
                glitch_widget.set_static(
                    highlighted,
                    title=title,
                    subtitle=subtitle,
                )

        # ── negative prompt ──────────────────────────────────────────
        self.query_one("#negative-display", Static).update(
            Panel(
                Text(p.negative, style=f"dim italic {palette.negative}"),
                title=f"[{palette.negative}]// negative[/]",
                border_style=palette.negative_border,
                padding=(0, 1),
            )
        )

        # ── component breakdown ──────────────────────────────────────
        tbl = Table(
            show_header=False, box=None, padding=(0, 2, 0, 0), expand=True
        )
        tbl.add_column("Category", min_width=22, no_wrap=True)
        tbl.add_column("Selection")
        for cat, words in p.components.items():
            tbl.add_row(
                Text(cat, style=LABEL_STYLES.get(cat, "dim green")),
                Text(", ".join(words), style=CATEGORY_STYLES.get(cat, "green")),
            )
        self.query_one("#components-display", Static).update(
            Panel(
                tbl,
                title=f"[{palette.dim}]// components[/]",
                border_style=palette.border_dim,
                padding=(0, 1),
            )
        )

        # ── entropy meter ────────────────────────────────────────────
        self.query_one("#entropy-display", EntropyMeter).set_progress(
            count=self.store.count,
            total=self.engine.total_combinations,
            template_filter=self._template_filter,
            auto_active=self._auto_timer is not None,
        )

        # ── history log ──────────────────────────────────────────────
        self.query_one("#history-log", HistoryLog).add_entry(
            hash_str=p.hash,
            template_id=p.template_id,
        )

        # ── hyperobject viewport ───────────────────────────────────────
        self.query_one(
            "#hyperobject-viewport", HyperobjectViewport
        ).set_prompt(p)

        # ── hacker trace log ─────────────────────────────────────────
        hacker = self.query_one("#hacker-log", HackerLog)
        hacker.add_trace(
            count=self.store.count,
            template_id=p.template_id,
            hash_str=p.hash,
            n_components=len(p.components),
            is_artifact=self._is_artifact,
        )

        # ── milestone check ──────────────────────────────────────────
        count = self.store.count
        if count in MILESTONES:
            self.notify(
                f"// MILESTONE: #{count:,} prompts generated",
                timeout=4,
            )
            hacker.add_milestone(count)

    # ── actions ───────────────────────────────────────────────────────

    def action_next_prompt(self) -> None:
        self._generate()

    def action_cycle_template(self) -> None:
        options: list[Optional[str]] = [None, *self.engine.template_ids]
        self._template_idx = (self._template_idx + 1) % len(options)
        self._template_filter = options[self._template_idx]
        label = self._template_filter or "all"
        self.notify(f"template: {label}", timeout=2)
        self._generate()

    def action_toggle_favorite(self) -> None:
        if not self.current:
            return
        new_state = self.store.toggle_favorite(self.current.hash)
        if new_state:
            self._favorites.add(self.current.hash)
            self.notify("\u2605 favorited", timeout=1)
        else:
            self._favorites.discard(self.current.hash)
            self.notify("unfavorited", timeout=1)
        self.query_one("#history-log", HistoryLog).set_favorites(self._favorites)
        self._render(animate=False)

    def action_toggle_auto(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None
            self.sub_title = "// combinatorial prompt generator"
            self.notify("auto-generate: OFF", timeout=1)
            # Re-render to clear the [AUTO] indicator
            self._render(animate=False)
        else:
            self._auto_timer = self.set_interval(AUTO_INTERVAL, self._generate)
            self.sub_title = "// combinatorial prompt generator [AUTO]"
            self.notify(f"auto-generate: ON ({AUTO_INTERVAL}s)", timeout=1)
            self._generate()

    def action_toggle_hyperobject(self) -> None:
        rain = self.query_one("#matrix-rain", MatrixRain)
        hyper = self.query_one("#hyperobject-viewport", HyperobjectViewport)
        hacker = self.query_one("#hacker-log", HackerLog)
        self._hyper_visible = not self._hyper_visible
        if self._hyper_visible:
            rain.display = False
            hacker.display = False
            hyper.display = True
            self._hacker_visible = False
            self.screen.add_class("hyper-active")
        else:
            hyper.display = False
            rain.display = not self._hacker_visible
            hacker.display = self._hacker_visible
            self.screen.remove_class("hyper-active")

    def action_toggle_hacker_log(self) -> None:
        rain = self.query_one("#matrix-rain", MatrixRain)
        hacker = self.query_one("#hacker-log", HackerLog)
        hyper = self.query_one("#hyperobject-viewport", HyperobjectViewport)
        self._hacker_visible = not self._hacker_visible
        if self._hacker_visible:
            rain.display = False
            hyper.display = False
            hacker.display = True
            self._hyper_visible = False
            self.screen.remove_class("hyper-active")
        else:
            hacker.display = False
            if self._hyper_visible:
                hyper.display = True
                self.screen.add_class("hyper-active")
            else:
                rain.display = True

    def action_copy_prompt(self) -> None:
        if not self.current:
            return
        self._copy_to_clipboard_async(self.current.positive, negative=False)

    def action_copy_negative(self) -> None:
        if not self.current:
            return
        self._copy_to_clipboard_async(self.current.negative, negative=True)

    def action_quit_app(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None
        self.store.close()
        self.exit()
