"""Color palettes for PromptWeaver TUI.

Each template maps to a palette that shifts the entire TUI chrome —
borders, banner, rain, stats — while component category highlights
stay fixed for readability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Named color set for the TUI chrome."""

    name: str
    primary: str  # main UI text color (hex)
    bright: str  # brightest highlights (Rich style)
    dim: str  # muted/dim text (Rich style)
    accent: str  # secondary flash color (Rich style)
    border: str  # active panel border
    border_dim: str  # inactive panel border
    negative: str  # negative prompt text
    negative_border: str  # negative panel border
    rain_head: str  # rain drop head (always white)
    rain_bright: str  # rain near head
    rain_mid: str  # rain middle
    rain_dim: str  # rain tail


# ── palette definitions ──────────────────────────────────────────────────

PALETTES: dict[str, Palette] = {
    "matrix": Palette(
        name="matrix",
        primary="#00ff41",
        bright="bright_green",
        dim="dim green",
        accent="bright_cyan",
        border="bright_green",
        border_dim="#006600",
        negative="#aa4444",
        negative_border="#664444",
        rain_head="bold bright_white",
        rain_bright="bold bright_green",
        rain_mid="green",
        rain_dim="#005500",
    ),
    "cyberpunk": Palette(
        name="cyberpunk",
        primary="#ff44ff",
        bright="bright_magenta",
        dim="dim magenta",
        accent="bright_cyan",
        border="bright_magenta",
        border_dim="#660066",
        negative="#aa8844",
        negative_border="#665533",
        rain_head="bold bright_white",
        rain_bright="bold bright_magenta",
        rain_mid="magenta",
        rain_dim="#550055",
    ),
    "amber": Palette(
        name="amber",
        primary="#ffaa00",
        bright="bold #ffcc44",
        dim="dim yellow",
        accent="bold #ff6600",
        border="#ffaa00",
        border_dim="#664400",
        negative="#6666aa",
        negative_border="#444466",
        rain_head="bold bright_white",
        rain_bright="bold #ffcc44",
        rain_mid="#aa7700",
        rain_dim="#553300",
    ),
    "ocean": Palette(
        name="ocean",
        primary="#00aaff",
        bright="bold #44ccff",
        dim="dim blue",
        accent="bright_cyan",
        border="#44aaff",
        border_dim="#004466",
        negative="#aa6644",
        negative_border="#664433",
        rain_head="bold bright_white",
        rain_bright="bold #44ccff",
        rain_mid="#0066aa",
        rain_dim="#003355",
    ),
    "void": Palette(
        name="void",
        primary="#aa66ff",
        bright="bold #cc88ff",
        dim="dim #7744aa",
        accent="bright_white",
        border="#aa66ff",
        border_dim="#442266",
        negative="#66aa44",
        negative_border="#446633",
        rain_head="bold bright_white",
        rain_bright="bold #cc88ff",
        rain_mid="#7744aa",
        rain_dim="#332255",
    ),
    "ember": Palette(
        name="ember",
        primary="#ff4422",
        bright="bold #ff6644",
        dim="dim red",
        accent="#ffaa00",
        border="#ff4422",
        border_dim="#661100",
        negative="#4488aa",
        negative_border="#334466",
        rain_head="bold bright_white",
        rain_bright="bold #ff6644",
        rain_mid="#aa2200",
        rain_dim="#551100",
    ),
    "frost": Palette(
        name="frost",
        primary="#88ccff",
        bright="bold #bbddff",
        dim="dim #6699bb",
        accent="bright_white",
        border="#88ccff",
        border_dim="#335566",
        negative="#cc8866",
        negative_border="#665544",
        rain_head="bold bright_white",
        rain_bright="bold #bbddff",
        rain_mid="#5588aa",
        rain_dim="#224455",
    ),
    "biomech": Palette(
        name="biomech",
        primary="#aaff00",
        bright="bold #ccff44",
        dim="dim #88aa00",
        accent="bold #00ffaa",
        border="#aaff00",
        border_dim="#446600",
        negative="#aa4488",
        negative_border="#663355",
        rain_head="bold bright_white",
        rain_bright="bold #ccff44",
        rain_mid="#66aa00",
        rain_dim="#334400",
    ),
}

# ── template → palette mapping ───────────────────────────────────────────

TEMPLATE_PALETTES: dict[str, str] = {
    "material_study": "amber",
    "textural_macro": "amber",
    "environmental": "ocean",
    "atmospheric_depth": "ocean",
    "process_state": "ember",
    "material_collision": "ember",
    "specimen": "frost",
    "minimal_object": "frost",
    "abstract_field": "void",
    "temporal_diptych": "void",
    "liminal": "cyberpunk",
    "ruin_state": "cyberpunk",
    "essence": "biomech",
    "site_decay": "matrix",
}

DEFAULT_PALETTE_NAME = "matrix"


def palette_for_template(template_id: str) -> Palette:
    """Get the palette associated with a template, falling back to matrix."""
    name = TEMPLATE_PALETTES.get(template_id, DEFAULT_PALETTE_NAME)
    return PALETTES.get(name, PALETTES[DEFAULT_PALETTE_NAME])
