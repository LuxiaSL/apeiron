"""Custom matrix-themed widgets for PromptWeaver TUI."""

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

from .palettes import DEFAULT_PALETTE_NAME, PALETTES, Palette

try:
    import pyfiglet
except ImportError:
    pyfiglet = None  # type: ignore[assignment]

# Glitch character palette
GLITCH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*<>{}[]|/\\~=+-"

_DEFAULT = PALETTES[DEFAULT_PALETTE_NAME]


class MatrixBanner(Static):
    """ASCII art title banner rendered with pyfiglet."""

    DEFAULT_CSS = """
    MatrixBanner {
        height: auto;
        padding: 0 2;
    }
    """

    def __init__(
        self,
        text: str = "PROMPTWEAVER",
        font: str = "doom",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._banner_text = text
        self._font = font
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        if palette is not self._palette:
            self._palette = palette
            self.refresh()

    def render(self) -> Text:
        if pyfiglet is not None:
            try:
                banner = pyfiglet.figlet_format(
                    self._banner_text, font=self._font, width=120
                )
            except Exception:
                banner = self._banner_text
        else:
            banner = f"  [ {self._banner_text} ]  "
        return Text(banner.rstrip("\n"), style=f"bold {self._palette.primary}")


class GlitchPrompt(Static):
    """Prompt display with glitch-decode reveal animation.

    Characters scramble randomly then "lock in" left-to-right, mimicking
    a password cracker / matrix decode sequence.
    """

    DEFAULT_CSS = """
    GlitchPrompt {
        height: auto;
        margin-bottom: 1;
    }
    """

    DECODE_FPS: float = 1 / 24
    TOTAL_FRAMES: int = 14

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._target_plain: str = ""
        self._target_renderable: RenderableType = Text("")
        self._panel_title: str = ""
        self._panel_subtitle: str = ""
        self._frame: int = 0
        self._timer: Optional[Timer] = None
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def decode(
        self,
        plain_text: str,
        final_renderable: RenderableType,
        title: str = "",
        subtitle: str = "",
    ) -> None:
        """Start glitch-decode animation, ending with final_renderable."""
        self._target_plain = plain_text
        self._target_renderable = final_renderable
        self._panel_title = title
        self._panel_subtitle = subtitle
        self._frame = 0
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.DECODE_FPS, self._tick)

    def _tick(self) -> None:
        self._frame += 1
        p = self._palette

        if self._frame >= self.TOTAL_FRAMES:
            self.update(
                Panel(
                    self._target_renderable,
                    title=self._panel_title,
                    subtitle=self._panel_subtitle,
                    border_style=p.border,
                    padding=(1, 2),
                )
            )
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            return

        # Build partially decoded text with palette-tinted flashes
        progress = self._frame / self.TOTAL_FRAMES
        locked_count = int(len(self._target_plain) * progress)

        text = Text()
        for i, ch in enumerate(self._target_plain):
            if i < locked_count:
                text.append(ch, style=p.bright)
            elif ch in (" ", ",", ".", "\n", ";", ":"):
                text.append(ch)
            else:
                glyph = random.choice(GLITCH_CHARS)
                roll = random.random()
                if roll < 0.10:
                    text.append(glyph, style=p.accent)
                elif roll < 0.14:
                    text.append(glyph, style="bright_white")
                else:
                    text.append(glyph, style=p.rain_mid)

        border_color = random.choice(
            [p.rain_mid, p.border, p.border_dim, p.accent]
        )
        self.update(
            Panel(
                text,
                title=self._panel_title,
                subtitle=f"[{p.dim}]decoding...[/]",
                border_style=border_color,
                padding=(1, 2),
            )
        )

    def set_static(
        self,
        renderable: RenderableType,
        title: str = "",
        subtitle: str = "",
        border_style: Optional[str] = None,
    ) -> None:
        """Set content immediately without animation."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.update(
            Panel(
                renderable,
                title=title,
                subtitle=subtitle,
                border_style=border_style or self._palette.border,
                padding=(1, 2),
            )
        )


class HistoryLog(Static):
    """Scrolling log of recent prompt generations."""

    DEFAULT_CSS = """
    HistoryLog {
        height: 100%;
        width: 100%;
    }
    """

    MAX_ENTRIES: int = 50
    VISIBLE_ENTRIES: int = 18

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._entries: list[tuple[str, str]] = []  # (hash, template_id)
        self._favorites: set[str] = set()
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def set_favorites(self, favorites: set[str]) -> None:
        """Update the set of favorited hashes and refresh display."""
        self._favorites = favorites
        self._refresh_display()

    def add_entry(self, hash_str: str, template_id: str) -> None:
        """Append a generation to the history log."""
        self._entries.append((hash_str, template_id))
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]
        self._refresh_display()

    def _refresh_display(self) -> None:
        p = self._palette
        visible = list(reversed(self._entries))[: self.VISIBLE_ENTRIES]
        text = Text()
        for i, (h, tid) in enumerate(visible):
            star = " \u2605" if h in self._favorites else ""
            if i == 0:
                text.append(" > ", style=f"bold {p.primary}")
                text.append(f"0x{h}{star}", style=f"bold {p.primary}")
                text.append(f"\n   {tid}\n", style=p.bright)
            else:
                style = p.bright if i <= 3 else p.dim
                text.append(f"   0x{h}{star}", style=style)
                text.append(f"\n   {tid}\n", style=p.dim)

        self.update(
            Panel(
                text or Text("  awaiting generation...", style=p.dim),
                title=f"[{p.dim}]// history[/]",
                border_style=p.border_dim,
                padding=(0, 1),
            )
        )


class MatrixRain(Static):
    """Cascading characters — digital rain effect."""

    DEFAULT_CSS = """
    MatrixRain {
        height: 1fr;
        min-height: 3;
        overflow: hidden;
    }
    """

    RAIN_CHARS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789@#$%&*<>{}|/~"
    )

    def __init__(self, density: float = 0.03, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._drops: list[dict[str, int]] = []
        self._density = density
        self._timer: Optional[Timer] = None
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 10, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        width = self.size.width
        height = self.size.height
        if width <= 0 or height <= 0:
            return

        self._drops = [
            d for d in self._drops if d["pos"] - d["length"] < height
        ]

        for col in range(0, width, 2):
            if random.random() < self._density:
                self._drops.append(
                    {
                        "col": col,
                        "pos": random.randint(-6, 0),
                        "speed": random.choice([1, 1, 2]),
                        "length": random.randint(4, max(5, height // 2)),
                    }
                )

        for drop in self._drops:
            drop["pos"] += drop["speed"]

        self._render_frame(width, height)

    def _render_frame(self, width: int, height: int) -> None:
        p = self._palette
        cells: dict[tuple[int, int], str] = {}
        for drop in self._drops:
            col = drop["col"]
            head = drop["pos"]
            length = drop["length"]
            for row in range(max(0, head - length), min(height, head + 1)):
                if col >= width:
                    continue
                dist = head - row
                if dist == 0:
                    cells[(row, col)] = p.rain_head
                elif dist <= 2:
                    cells[(row, col)] = p.rain_bright
                elif dist <= length // 2:
                    cells[(row, col)] = p.rain_mid
                else:
                    cells[(row, col)] = p.rain_dim

        text = Text()
        for row in range(height):
            for col in range(width):
                style = cells.get((row, col))
                if style:
                    text.append(random.choice(self.RAIN_CHARS), style=style)
                else:
                    text.append(" ")
            if row < height - 1:
                text.append("\n")

        self.update(text)


class HackerLog(Static):
    """Terminal-style generation trace log, toggled with 'h' key."""

    DEFAULT_CSS = """
    HackerLog {
        display: none;
        height: 1fr;
        min-height: 3;
        overflow: hidden;
    }
    """

    MAX_LINES: int = 80

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._lines: list[tuple[str, str]] = []  # (timestamp, message)
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def add_trace(
        self,
        count: int,
        template_id: str,
        hash_str: str,
        n_components: int,
        *,
        is_artifact: bool = False,
    ) -> None:
        """Log a generation event."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        tag = " !! ARTIFACT" if is_artifact else ""
        msg = (
            f"[{ts}] GEN #{count:>6,} | {template_id:<22}"
            f"| 0x{hash_str} | {n_components} slots{tag}"
        )
        self._lines.append((ts, msg))
        if len(self._lines) > self.MAX_LINES:
            self._lines = self._lines[-self.MAX_LINES:]
        self._refresh_display()

    def add_milestone(self, count: int) -> None:
        """Log a milestone event."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = f"[{ts}] >>> MILESTONE: #{count:,} prompts generated <<<"
        self._lines.append((ts, msg))
        self._refresh_display()

    def _refresh_display(self) -> None:
        p = self._palette
        text = Text()
        for i, (_, msg) in enumerate(reversed(self._lines)):
            style = p.bright if i == 0 else p.dim
            text.append(f"  {msg}\n", style=style)

        self.update(
            Panel(
                text or Text("  // awaiting trace data...", style=p.dim),
                title=f"[{p.dim}]// system trace[/]",
                border_style=p.border_dim,
                padding=(0, 1),
            )
        )


class EntropyMeter(Static):
    """Visual meter showing exploration of the combinatorial space."""

    DEFAULT_CSS = """
    EntropyMeter {
        height: auto;
        margin-top: 1;
    }
    """

    BAR_WIDTH: int = 32

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._palette: Palette = _DEFAULT

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def set_progress(
        self,
        count: int,
        total: int,
        template_filter: Optional[str] = None,
        auto_active: bool = False,
    ) -> None:
        """Update the entropy meter with current coverage stats."""
        p = self._palette
        pct = (count / total * 100) if total > 0 else 0.0

        if total > 1 and count > 0:
            log_progress = math.log10(count + 1) / math.log10(total)
        else:
            log_progress = 0.0

        filled = int(self.BAR_WIDTH * min(log_progress, 1.0))
        empty = self.BAR_WIDTH - filled

        bar = Text()
        bar.append("  ", style="")
        bar.append("\u2588" * filled, style=p.bright)
        bar.append("\u2591" * empty, style=p.rain_dim)
        bar.append("  ", style="")
        bar.append(f"#{count:,}", style=f"bold {p.primary}")
        bar.append("  of  ", style="dim")
        bar.append(f"~{total:,}", style=p.dim)
        bar.append("  \u00b7  ", style="dim")
        bar.append(f"{pct:.8f}%", style=p.accent)

        if template_filter:
            bar.append(f"  [{template_filter}]", style=p.accent)
        else:
            bar.append("  [all]", style="dim")

        if auto_active:
            bar.append("  [AUTO]", style=f"bold {p.accent}")

        self.update(bar)
