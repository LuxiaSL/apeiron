"""Post-processing effect stack for the hyperobject renderer.

Effects operate on a completed ``CharGrid`` (flat array of ``Cell`` objects)
and modify it in place before display.  Each effect is a function with
signature ``(grid: CharGrid, ...) -> None``.

The ``apply_effects`` dispatcher runs a named list of effects in order,
and ``effect_for_word`` maps ``medium_render`` component words to effect
stacks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .lut import clamp
from .rasterizer import Cell, CharGrid


# ── style helpers ──────────────────────────────────────────────────────


def _add_dim(style: str) -> str:
    """Prepend ``dim`` to a style string if not already present."""
    if "dim" in style:
        return style
    return ("dim " + style).strip()


def _is_empty(cell: Cell) -> bool:
    """Return True if the cell contains only whitespace."""
    return cell.char.strip() == ""


# ── effects ────────────────────────────────────────────────────────────


def apply_scanlines(grid: CharGrid, period: int = 2) -> None:
    """Dim every *period*-th row to simulate CRT scanlines."""
    period = max(1, period)
    for row in range(grid.height):
        if row % period == 0:
            for col in range(grid.width):
                cell = grid.get(col, row)
                cell.style = _add_dim(cell.style)


def apply_vignette(grid: CharGrid, strength: float = 0.5) -> None:
    """Dim cells near viewport edges based on radial distance from center.

    *strength* controls how aggressively edges are dimmed (0 = none,
    1 = maximum).
    """
    strength = clamp(strength, 0.0, 1.0)
    if strength < 1e-6:
        return

    cx = grid.width / 2.0
    cy = grid.height / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    if max_r < 1e-6:
        return

    for row in range(grid.height):
        dy = (row - cy) / max(cy, 1.0)
        for col in range(grid.width):
            dx = (col - cx) / max(cx, 1.0)
            r = math.sqrt(dx * dx + dy * dy)
            # Normalise so that corners sit at r~1.41; apply strength curve.
            factor = clamp(r * strength, 0.0, 1.0)
            if factor > 0.4:
                cell = grid.get(col, row)
                cell.style = _add_dim(cell.style)


_BLOOM_CHARS_DEFAULT: str = "\u2588#%@\u25cf\u25c9"  # █#%@●◉


def apply_bloom(
    grid: CharGrid, threshold_chars: str = _BLOOM_CHARS_DEFAULT
) -> None:
    """Bright characters spread a faint glow to adjacent empty cells.

    Any cell whose ``char`` is in *threshold_chars* causes its 4-connected
    neighbours to receive a dim dot if they are currently empty.
    """
    # Collect bloom sources first to avoid feedback within one pass.
    sources: list[tuple[int, int]] = []
    tc = set(threshold_chars)
    for row in range(grid.height):
        for col in range(grid.width):
            if grid.get(col, row).char in tc:
                sources.append((col, row))

    for col, row in sources:
        for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nc, nr = col + dc, row + dr
            if grid.in_bounds(nc, nr):
                neighbour = grid.get(nc, nr)
                if _is_empty(neighbour):
                    neighbour.char = "\u00b7"  # ·
                    neighbour.style = "dim"


def apply_noise_grain(grid: CharGrid, density: float = 0.05) -> None:
    """Randomly replace some empty cells with faint dots.

    *density* is the probability that any given empty cell receives a
    noise dot.
    """
    density = clamp(density, 0.0, 1.0)
    grain_chars = ("\u00b7", "\u2219", ".")  # ·, ∙, .
    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.get(col, row)
            if _is_empty(cell) and random.random() < density:
                cell.char = random.choice(grain_chars)
                cell.style = "dim"


def apply_edge_glow(grid: CharGrid) -> None:
    """Cells adjacent to non-empty cells receive a faint glow character.

    Only empty cells are affected.  The glow character is a dim dot.
    """
    glow_targets: list[tuple[int, int]] = []
    for row in range(grid.height):
        for col in range(grid.width):
            if not _is_empty(grid.get(col, row)):
                for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nc, nr = col + dc, row + dr
                    if grid.in_bounds(nc, nr) and _is_empty(grid.get(nc, nr)):
                        glow_targets.append((nc, nr))

    for col, row in glow_targets:
        cell = grid.get(col, row)
        if _is_empty(cell):
            cell.char = "\u00b7"  # ·
            cell.style = "dim"


def apply_crt_warp(grid: CharGrid) -> None:
    """Barrel distortion: remap cell positions radially from center.

    Creates the illusion of CRT screen curvature by pulling cells
    toward the edges.  Operates by building a remapped copy of the
    grid and writing it back.
    """
    w, h = grid.width, grid.height
    if w < 4 or h < 4:
        return

    cx = w / 2.0
    cy = h / 2.0
    # Barrel distortion coefficient (subtle).
    k = 0.15
    max_r = math.sqrt(cx * cx + cy * cy)
    if max_r < 1e-6:
        return

    new_cells: list[Cell] = [Cell() for _ in range(w * h)]

    for row in range(h):
        for col in range(w):
            # Normalised coords from center.
            dx = (col - cx) / cx
            dy = (row - cy) / cy
            r = math.sqrt(dx * dx + dy * dy)

            # Barrel distortion: push outward.
            if r > 1e-6:
                distorted_r = r + k * r * r * r
                scale = distorted_r / r
            else:
                scale = 1.0

            src_col = int(cx + dx * scale * cx)
            src_row = int(cy + dy * scale * cy)

            if grid.in_bounds(src_col, src_row):
                src = grid.get(src_col, src_row)
                new_cells[row * w + col] = Cell(char=src.char, style=src.style)
            # else: remains an empty Cell (black border from warp)

    grid.cells = new_cells


# ── word -> effect mapping ─────────────────────────────────────────────

# Maps medium_render words to ordered lists of effect names.
_EFFECT_MAP: dict[str, list[str]] = {
    "oil_impasto": ["bloom", "edge_glow"],
    "charcoal": ["noise_grain", "edge_glow", "vignette"],
    "risograph": ["scanlines", "noise_grain"],
    "daguerreotype": ["vignette", "noise_grain"],
    "3d_render": [],  # clean pass-through
    "3d render": [],
    "glitch_art": ["scanlines", "noise_grain", "crt_warp"],
    "crt": ["scanlines", "crt_warp", "bloom"],
    "blueprint": ["scanlines", "edge_glow"],
}

# Fallback stacks, indexed by hash.
_FALLBACK_STACKS: list[list[str]] = [
    ["vignette"],
    ["scanlines", "bloom"],
    ["noise_grain", "edge_glow"],
    ["vignette", "scanlines"],
    ["bloom", "noise_grain"],
    ["edge_glow", "vignette"],
    ["crt_warp", "scanlines"],
    ["bloom", "edge_glow", "vignette"],
]


def _stable_hash(word: str) -> int:
    """FNV-1a hash for deterministic cross-platform results."""
    h: int = 0x811C9DC5
    for ch in word:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def effect_for_word(word: str) -> list[str]:
    """Map a ``medium_render`` word to an ordered list of effect names."""
    key = word.lower().strip()
    if key in _EFFECT_MAP:
        return list(_EFFECT_MAP[key])
    # Deterministic fallback.
    idx = _stable_hash(key) % len(_FALLBACK_STACKS)
    return list(_FALLBACK_STACKS[idx])


# ── effect dispatcher ──────────────────────────────────────────────────

_EFFECT_FUNCTIONS: dict[str, object] = {
    "scanlines": apply_scanlines,
    "vignette": apply_vignette,
    "bloom": apply_bloom,
    "noise_grain": apply_noise_grain,
    "edge_glow": apply_edge_glow,
    "crt_warp": apply_crt_warp,
}


def apply_effects(grid: CharGrid, effect_names: list[str]) -> None:
    """Apply a stack of named effects to *grid* in the given order.

    Unknown effect names are silently skipped so that new effect names
    can be added to mappings before their implementations land.
    """
    for name in effect_names:
        fn = _EFFECT_FUNCTIONS.get(name)
        if fn is not None:
            try:
                fn(grid)  # type: ignore[operator]
            except Exception:
                # Never let a single post-fx crash the renderer.
                pass
