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


# ── depth-aware effects ───────────────────────────────────────────────


def apply_depth_fog(grid: CharGrid, start: float = 0.35, fade: float = 0.7) -> None:
    """Dim cells based on z-depth, creating atmospheric perspective.

    Auto-normalizes to the actual depth range of the rendered geometry
    so the effect works regardless of camera distance.  Near cells
    (normalized depth < *start*) stay crisp, mid-range cells are dimmed,
    and far cells (> *fade*) degrade to faint dots.
    """
    cells = grid.cells
    # Find occupied depth range for normalization.
    min_d = 1.0
    max_d = 0.0
    for cell in cells:
        if cell.char.strip():
            if cell.depth < min_d:
                min_d = cell.depth
            if cell.depth > max_d:
                max_d = cell.depth
    depth_range = max_d - min_d
    if depth_range < 1e-6:
        return  # flat depth — nothing to fog

    add_dim = _add_dim
    inv_range = 1.0 / depth_range
    for cell in cells:
        if not cell.char.strip():
            continue
        nd = (cell.depth - min_d) * inv_range  # normalized 0–1
        if nd <= start:
            continue
        cell.style = add_dim(cell.style)
        if nd > fade:
            cell.char = "\u00b7"  # ·


def apply_depth_of_field(
    grid: CharGrid, focus: float = 0.35, sharp_range: float = 0.2,
) -> None:
    """Selective focus — degrade characters far from the focus plane.

    Auto-normalizes to the geometry's actual depth range.  Cells near
    *focus* (within *sharp_range*, in normalized depth) stay crisp.
    Beyond that, characters shift toward simpler forms and the style
    dims — simulating optical bokeh.
    """
    cells = grid.cells
    # Find occupied depth range.
    min_d = 1.0
    max_d = 0.0
    for cell in cells:
        if cell.char.strip():
            if cell.depth < min_d:
                min_d = cell.depth
            if cell.depth > max_d:
                max_d = cell.depth
    depth_range = max_d - min_d
    if depth_range < 1e-6:
        return

    add_dim = _add_dim
    inv_range = 1.0 / depth_range
    ramp = _DOF_RAMP
    ramp_len = len(ramp)
    brightness_map = _DOF_BRIGHTNESS
    max_blur_range = max(1.0 - sharp_range, 0.3)

    for cell in cells:
        if not cell.char.strip():
            continue
        nd = (cell.depth - min_d) * inv_range
        defocus = abs(nd - focus)
        if defocus <= sharp_range:
            continue
        blur = min((defocus - sharp_range) / max_blur_range, 1.0)
        # Degrade character along luminance ramp.
        brightness = brightness_map.get(cell.char, 0.5)
        new_brightness = brightness * (1.0 - blur * 0.7)
        new_idx = max(1, min(int(new_brightness * (ramp_len - 1) + 0.5), ramp_len - 1))
        cell.char = ramp[new_idx]
        if blur > 0.3:
            cell.style = add_dim(cell.style)


# Luminance ramp used by depth_of_field for character degradation.
_DOF_RAMP: str = " .,-~:;=!*#$@"
_DOF_BRIGHTNESS: dict[str, float] = {
    ch: i / max(len(_DOF_RAMP) - 1, 1)
    for i, ch in enumerate(_DOF_RAMP)
}


def apply_depth_contour(grid: CharGrid, threshold: float = 0.12) -> None:
    """Draw contour lines at depth discontinuities.

    Auto-normalizes to the geometry's depth range, then detects sharp
    transitions between adjacent cells and places directional edge
    characters at boundaries — topographic contour lines through 3D.
    """
    width = grid.width
    height = grid.height
    cells = grid.cells

    # Compute actual depth range for threshold scaling.
    min_d = 1.0
    max_d = 0.0
    for cell in cells:
        if cell.char.strip():
            if cell.depth < min_d:
                min_d = cell.depth
            if cell.depth > max_d:
                max_d = cell.depth
    depth_range = max_d - min_d
    if depth_range < 1e-6:
        return
    # Scale threshold to actual depth range so contours appear at the
    # same visual density regardless of camera distance.
    abs_threshold = threshold * depth_range

    contours: list[tuple[int, str]] = []

    for row in range(height):
        base = row * width
        for col in range(width):
            idx = base + col
            d = cells[idx].depth
            if d >= 1.0:  # empty cell
                continue

            h_edge = (
                row + 1 < height
                and abs(d - cells[idx + width].depth) > abs_threshold
            )
            v_edge = (
                col + 1 < width
                and abs(d - cells[idx + 1].depth) > abs_threshold
            )

            if h_edge and v_edge:
                contours.append((idx, "\u253c"))  # ┼
            elif h_edge:
                contours.append((idx, "\u2500"))  # ─
            elif v_edge:
                contours.append((idx, "\u2502"))  # │

    for idx, char in contours:
        cells[idx].char = char


# ── temporal effects ──────────────────────────────────────────────────


def apply_rolling_bars(
    grid: CharGrid, speed: float = 3.0, frequency: float = 0.4,
) -> None:
    """Horizontal dim bands that scroll downward — CRT interference.

    Uses ``grid.time`` for animation.  *speed* controls scroll rate,
    *frequency* controls how many bands are visible at once.
    """
    width = grid.width
    cells = grid.cells
    add_dim = _add_dim
    time = grid.time

    for row in range(grid.height):
        phase = (row + time * speed) * frequency
        modulation = math.sin(phase)
        if modulation < -0.3:
            base = row * width
            for col in range(width):
                cell = cells[base + col]
                if cell.char.strip():
                    cell.style = add_dim(cell.style)


def apply_flicker(grid: CharGrid, intensity: float = 0.08) -> None:
    """Per-cell random dimming — dying fluorescent tube.

    Uses ``grid.time`` to create frame-varying but deterministic-within-
    frame noise.  *intensity* (0--1) controls the fraction of cells
    affected each frame.
    """
    cells = grid.cells
    add_dim = _add_dim
    # Golden-ratio hash of frame index for per-frame variation.
    frame_seed = int(grid.time * 8) * 0x9E3779B9 & 0xFFFFFFFF
    threshold = int(clamp(intensity, 0.0, 1.0) * 255)

    for idx, cell in enumerate(cells):
        if not cell.char.strip():
            continue
        # Fast integer hash: multiply by large prime, XOR with seed.
        h = ((idx * 2654435761) ^ frame_seed) & 0xFFFFFFFF
        if (h & 0xFF) < threshold:
            cell.style = add_dim(cell.style)


def apply_pulse(
    grid: CharGrid, speed: float = 1.5, wavelength: float = 0.4,
) -> None:
    """Radial brightness wave — sonar ring sweeping outward from center.

    Uses ``grid.time`` for animation.  *speed* controls expansion rate,
    *wavelength* controls the spacing between concentric rings.
    """
    width = grid.width
    height = grid.height
    cells = grid.cells
    add_dim = _add_dim
    cx = width / 2.0
    cy = height / 2.0
    time = grid.time
    inv_wl = 1.0 / max(wavelength, 0.01)
    two_pi = math.pi * 2.0
    inv_cx = 1.0 / max(cx, 1.0)
    inv_cy = 1.0 / max(cy, 1.0)

    for row in range(height):
        dy = (row - cy) * inv_cy
        dy_sq = dy * dy
        base = row * width
        for col in range(width):
            cell = cells[base + col]
            if not cell.char.strip():
                continue
            dx = (col - cx) * inv_cx
            dist = math.sqrt(dx * dx + dy_sq)
            wave = math.sin((dist - time * speed) * inv_wl * two_pi)
            if wave < -0.3:
                cell.style = add_dim(cell.style)


# ── character-space effects ───────────────────────────────────────────


def apply_chromatic_split(grid: CharGrid, offset: int = 1) -> None:
    """Horizontal color fringing — shifted dim copy at geometry edges.

    Non-empty cells are duplicated *offset* columns to the right with a
    dimmed style, writing only into empty destinations.  Simulates
    chromatic aberration from misregistered printing plates.
    """
    width = grid.width
    height = grid.height
    src_cells = grid.cells
    n = width * height
    scratch = grid.fx_scratch
    if len(scratch) != n:
        scratch = [Cell() for _ in range(n)]

    # Copy source state to scratch.
    for i in range(n):
        src = src_cells[i]
        dst = scratch[i]
        dst.char = src.char
        dst.style = src.style
        dst.depth = src.depth

    add_dim = _add_dim
    # Overlay shifted dim copies into empty cells.
    for row in range(height):
        base = row * width
        for col in range(width):
            src = src_cells[base + col]
            if not src.char.strip():
                continue
            dst_col = col + offset
            if 0 <= dst_col < width:
                dst = scratch[base + dst_col]
                if not dst.char.strip():
                    dst.char = src.char
                    dst.style = add_dim(src.style)
                    dst.depth = src.depth

    grid.cells = scratch
    grid.fx_scratch = src_cells


def apply_ghosting(grid: CharGrid, max_offset: int = 2) -> None:
    """Phosphor persistence — faint copies shifted downward.

    Every non-empty cell leaves a dim echo 1--*max_offset* rows below.
    Closer ghosts use dimmed source style; deeper ghosts fade to plain
    ``dim``.  Only writes into currently empty cells.
    """
    width = grid.width
    height = grid.height
    cells = grid.cells
    add_dim = _add_dim

    # Collect all ghosts before writing to avoid feedback.
    ghosts: list[tuple[int, str, str]] = []  # (target_idx, char, style)
    for row in range(height):
        base = row * width
        for col in range(width):
            src = cells[base + col]
            if not src.char.strip():
                continue
            for dy in range(1, max_offset + 1):
                target_row = row + dy
                if target_row >= height:
                    break
                target_idx = target_row * width + col
                # Deeper ghosts get progressively fainter.
                style = add_dim(src.style) if dy == 1 else "dim"
                ghosts.append((target_idx, src.char, style))

    for target_idx, char, style in ghosts:
        cell = cells[target_idx]
        if not cell.char.strip():
            cell.char = char
            cell.style = style


def apply_contour_extract(grid: CharGrid) -> None:
    """Extract silhouette outline — blank the interior, keep edges.

    Non-empty cells that have *no* empty 4-connected neighbour (i.e.
    fully interior) are blanked.  Only cells on the geometry boundary
    survive.  The inverse of ``edge_glow``.
    """
    width = grid.width
    height = grid.height
    cells = grid.cells

    # First pass: classify each non-empty cell as edge or interior.
    is_edge = bytearray(width * height)  # 0 = interior, 1 = edge

    for row in range(height):
        base = row * width
        for col in range(width):
            idx = base + col
            if not cells[idx].char.strip():
                continue
            # Edge if on grid boundary or any 4-neighbour is empty.
            if (
                col == 0
                or col + 1 >= width
                or row == 0
                or row + 1 >= height
                or not cells[idx - 1].char.strip()
                or not cells[idx + 1].char.strip()
                or not cells[idx - width].char.strip()
                or not cells[idx + width].char.strip()
            ):
                is_edge[idx] = 1

    # Second pass: blank interior cells.
    for idx, cell in enumerate(cells):
        if cell.char.strip() and not is_edge[idx]:
            cell.char = " "
            cell.style = ""


# 4x4 Bayer ordered-dither matrix (normalized to [0, 1]).
_BAYER_4X4: tuple[tuple[float, ...], ...] = (
    (0.0 / 16, 8.0 / 16, 2.0 / 16, 10.0 / 16),
    (12.0 / 16, 4.0 / 16, 14.0 / 16, 6.0 / 16),
    (3.0 / 16, 11.0 / 16, 1.0 / 16, 9.0 / 16),
    (15.0 / 16, 7.0 / 16, 13.0 / 16, 5.0 / 16),
)

# Approximate visual density of common ASCII/Unicode characters.
# Characters not in this map default to 0.5.
_HALFTONE_DENSITY: dict[str, float] = {}
_DENSITY_RAMP = " .\u00b7,:-~;=+*#$@\u2588"
for _i, _ch in enumerate(_DENSITY_RAMP):
    _HALFTONE_DENSITY[_ch] = _i / max(len(_DENSITY_RAMP) - 1, 1)


def apply_halftone(grid: CharGrid) -> None:
    """Ordered dither — Bayer-matrix halftone dot pattern.

    Each cell's character is mapped to an approximate brightness.  If
    that brightness falls below the Bayer threshold at the cell's
    position, the cell is blanked.  Creates structured dot patterns
    reminiscent of newspaper halftone printing.
    """
    width = grid.width
    cells = grid.cells
    density_map = _HALFTONE_DENSITY
    bayer = _BAYER_4X4

    for idx, cell in enumerate(cells):
        if not cell.char.strip():
            continue
        col = idx % width
        row = idx // width
        threshold = bayer[row & 3][col & 3]
        brightness = density_map.get(cell.char, 0.5)
        if brightness < threshold:
            cell.char = " "
            cell.style = ""


# ── word -> effect mapping ─────────────────────────────────────────────

# Maps medium_render words to ordered lists of effect names.
_EFFECT_MAP: dict[str, list[str]] = {
    # ── original mappings ──
    "oil_impasto": ["bloom", "edge_glow"],
    "charcoal": ["noise_grain", "edge_glow", "vignette"],
    "risograph": ["scanlines", "noise_grain"],
    "daguerreotype": ["vignette", "noise_grain", "depth_fog"],
    "3d_render": [],  # clean pass-through
    "3d render": [],
    "glitch_art": ["scanlines", "noise_grain", "crt_warp"],
    "crt": ["scanlines", "crt_warp", "bloom"],
    "blueprint": ["scanlines", "edge_glow"],
    # ── new mappings (depth / temporal / character-space) ──
    "glass plate negative": ["depth_fog", "vignette", "noise_grain"],
    "carbon print process": ["halftone", "vignette", "depth_fog"],
    "mri scan": ["contour_extract", "scanlines", "depth_fog"],
    "holographic interferometry": ["chromatic_split", "rolling_bars", "bloom"],
    "cross-processed film": ["ghosting", "chromatic_split", "vignette"],
    "fresco buon": ["halftone", "noise_grain", "edge_glow"],
    "oil pastel drawing": ["ghosting", "bloom", "edge_glow"],
    "electron micrograph": ["contour_extract", "depth_of_field", "scanlines"],
    "infrared photography": ["depth_fog", "bloom", "flicker"],
    "cyanotype": ["contour_extract", "vignette", "depth_fog"],
    "tintype": ["depth_fog", "noise_grain", "vignette"],
    "photogram": ["contour_extract", "bloom", "depth_fog"],
    "screen print": ["halftone", "chromatic_split"],
    "thermal imaging": ["depth_contour", "bloom", "pulse"],
    "x-ray": ["contour_extract", "depth_fog", "flicker"],
    "lenticular print": ["chromatic_split", "rolling_bars"],
    "kirlian photograph": ["edge_glow", "bloom", "ghosting", "flicker"],
    "solarized print": ["chromatic_split", "vignette", "depth_of_field"],
    "linocut": ["contour_extract", "halftone"],
    "stereoscopic": ["chromatic_split", "depth_fog"],
}

# Fallback stacks, indexed by hash.
_FALLBACK_STACKS: list[list[str]] = [
    # ── original stacks ──
    ["vignette"],
    ["scanlines", "bloom"],
    ["noise_grain", "edge_glow"],
    ["vignette", "scanlines"],
    ["bloom", "noise_grain"],
    ["edge_glow", "vignette"],
    ["crt_warp", "scanlines"],
    ["bloom", "edge_glow", "vignette"],
    # ── new stacks (mixing old + new effects) ──
    ["depth_fog", "vignette"],
    ["ghosting", "scanlines"],
    ["chromatic_split", "bloom"],
    ["rolling_bars", "depth_fog"],
    ["contour_extract", "edge_glow"],
    ["halftone", "vignette", "depth_fog"],
    ["depth_of_field", "noise_grain"],
    ["ghosting", "chromatic_split"],
    ["flicker", "scanlines", "depth_fog"],
    ["pulse", "bloom", "edge_glow"],
    ["depth_contour", "depth_fog"],
    ["chromatic_split", "ghosting", "vignette"],
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
    # original
    "scanlines": apply_scanlines,
    "vignette": apply_vignette,
    "bloom": apply_bloom,
    "noise_grain": apply_noise_grain,
    "edge_glow": apply_edge_glow,
    "crt_warp": apply_crt_warp,
    # depth-aware
    "depth_fog": apply_depth_fog,
    "depth_of_field": apply_depth_of_field,
    "depth_contour": apply_depth_contour,
    # temporal
    "rolling_bars": apply_rolling_bars,
    "flicker": apply_flicker,
    "pulse": apply_pulse,
    # character-space
    "chromatic_split": apply_chromatic_split,
    "ghosting": apply_ghosting,
    "contour_extract": apply_contour_extract,
    "halftone": apply_halftone,
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
