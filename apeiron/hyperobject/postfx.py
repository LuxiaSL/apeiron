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

_CRT_WARP_MAP_CACHE: dict[tuple[int, int], tuple[int, ...]] = {}


# ── style helpers ──────────────────────────────────────────────────────


def _add_dim(style: str) -> str:
    """Prepend ``dim`` to a style string if not already present."""
    if "dim" in style:
        return style
    return ("dim " + style).strip()


# ── effects ────────────────────────────────────────────────────────────


def apply_scanlines(grid: CharGrid, period: int = 2) -> None:
    """Dim every *period*-th row to simulate CRT scanlines."""
    period = max(1, period)
    width = grid.width
    cells = grid.cells
    add_dim = _add_dim
    for row in range(grid.height):
        if row % period == 0:
            base = row * width
            for col in range(width):
                cell = cells[base + col]
                cell.style = add_dim(cell.style)


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
    if cx < 1e-6 or cy < 1e-6:
        return

    width = grid.width
    cells = grid.cells
    add_dim = _add_dim
    inv_cx = 1.0 / max(cx, 1.0)
    inv_cy = 1.0 / max(cy, 1.0)
    threshold = 0.4 / strength
    threshold_sq = threshold * threshold
    col_norm_sq = [((col - cx) * inv_cx) ** 2 for col in range(width)]

    for row in range(grid.height):
        row_norm_sq = ((row - cy) * inv_cy) ** 2
        base = row * width
        for col in range(width):
            if row_norm_sq + col_norm_sq[col] > threshold_sq:
                cell = cells[base + col]
                cell.style = add_dim(cell.style)


_BLOOM_CHARS_DEFAULT: str = "\u2588#%@\u25cf\u25c9"  # █#%@●◉


def apply_bloom(
    grid: CharGrid, threshold_chars: str = _BLOOM_CHARS_DEFAULT
) -> None:
    """Bright characters spread a faint glow to adjacent empty cells.

    Any cell whose ``char`` is in *threshold_chars* causes its 4-connected
    neighbours to receive a dim dot if they are currently empty.
    """
    # Collect bloom sources first to avoid feedback within one pass.
    width = grid.width
    height = grid.height
    cells = grid.cells
    sources: list[int] = []
    tc = set(threshold_chars)
    for idx, cell in enumerate(cells):
        if cell.char in tc:
            sources.append(idx)

    for idx in sources:
        col = idx % width
        row = idx // width
        if col > 0:
            neighbour = cells[idx - 1]
            if not neighbour.char.strip():
                neighbour.char = "\u00b7"
                neighbour.style = "dim"
        if col + 1 < width:
            neighbour = cells[idx + 1]
            if not neighbour.char.strip():
                neighbour.char = "\u00b7"
                neighbour.style = "dim"
        if row > 0:
            neighbour = cells[idx - width]
            if not neighbour.char.strip():
                neighbour.char = "\u00b7"
                neighbour.style = "dim"
        if row + 1 < height:
            neighbour = cells[idx + width]
            if not neighbour.char.strip():
                neighbour.char = "\u00b7"
                neighbour.style = "dim"


def apply_noise_grain(grid: CharGrid, density: float = 0.05) -> None:
    """Randomly replace some empty cells with faint dots.

    *density* is the probability that any given empty cell receives a
    noise dot.
    """
    density = clamp(density, 0.0, 1.0)
    grain_chars = ("\u00b7", "\u2219", ".")
    cells = grid.cells
    rand = random.random
    randrange = random.randrange
    for cell in cells:
        if not cell.char.strip() and rand() < density:
            cell.char = grain_chars[randrange(3)]
            cell.style = "dim"


def apply_edge_glow(grid: CharGrid) -> None:
    """Cells adjacent to non-empty cells receive a faint glow character.

    Only empty cells are affected.  The glow character is a dim dot.
    """
    width = grid.width
    height = grid.height
    cells = grid.cells
    glow_targets: list[int] = []

    for idx, cell in enumerate(cells):
        if cell.char.strip():
            col = idx % width
            row = idx // width
            if col > 0 and not cells[idx - 1].char.strip():
                glow_targets.append(idx - 1)
            if col + 1 < width and not cells[idx + 1].char.strip():
                glow_targets.append(idx + 1)
            if row > 0 and not cells[idx - width].char.strip():
                glow_targets.append(idx - width)
            if row + 1 < height and not cells[idx + width].char.strip():
                glow_targets.append(idx + width)

    for idx in glow_targets:
        cell = cells[idx]
        if not cell.char.strip():
            cell.char = "\u00b7"
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
    if cx < 1e-6 or cy < 1e-6:
        return

    remap = _CRT_WARP_MAP_CACHE.get((w, h))
    if remap is None:
        col_norms = [(col - cx) / cx for col in range(w)]
        row_norms = [(row - cy) / cy for row in range(h)]
        remap_list: list[int] = [-1] * (w * h)
        for row in range(h):
            dy = row_norms[row]
            base = row * w
            for col in range(w):
                dx = col_norms[col]
                r = math.sqrt(dx * dx + dy * dy)
                if r > 1e-6:
                    distorted_r = r + k * r * r * r
                    scale = distorted_r / r
                else:
                    scale = 1.0

                src_col = int(cx + dx * scale * cx)
                src_row = int(cy + dy * scale * cy)
                if 0 <= src_col < w and 0 <= src_row < h:
                    remap_list[base + col] = src_row * w + src_col
        remap = tuple(remap_list)
        _CRT_WARP_MAP_CACHE[(w, h)] = remap

    src_cells = grid.cells
    scratch = grid.fx_scratch
    if len(scratch) != w * h:
        scratch = [Cell() for _ in range(w * h)]

    for idx, src_idx in enumerate(remap):
        dst = scratch[idx]
        if src_idx >= 0:
            src = src_cells[src_idx]
            dst.char = src.char
            dst.style = src.style
            dst.depth = src.depth
        else:
            dst.char = " "
            dst.style = ""
            dst.depth = 1.0

    grid.cells = scratch
    grid.fx_scratch = src_cells


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
