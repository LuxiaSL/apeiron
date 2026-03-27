"""ASCII surface shaders for the hyperobject renderer.

Maps brightness values (0.0-1.0) to ASCII/Unicode characters using
configurable character ramps. Component words from ``material_substance``
are deterministically mapped to shader presets via hashing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lut import clamp


# ── shader ramp ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ShaderRamp:
    """A brightness-to-character mapping.

    ``chars`` should be at least 2 characters long, ordered dark (index 0)
    to bright (last index).  ``shade()`` maps a float in [0, 1] onto one
    of these characters.  10 characters is the conventional length but
    longer ramps (e.g. donut.c's 13-char ramp) give finer gradation.
    """

    chars: str

    def __post_init__(self) -> None:
        if len(self.chars) < 2:
            raise ValueError(
                f"ShaderRamp requires at least 2 characters, got {len(self.chars)}"
            )


# ── presets ────────────────────────────────────────────────────────────
# Each ramp is exactly 10 characters: dark (spaces) -> bright (dense).
# Padding with repeats at either end to reach 10 where necessary.

SHADER_PRESETS: dict[str, ShaderRamp] = {
    "donut": ShaderRamp(" .,-~:;=!*#$@"),
    # donut.c's original 13-char luminance ramp — the gold standard
    "block": ShaderRamp(" \u2591\u2591\u2592\u2592\u2593\u2593\u2588\u2588\u2588"),
    # " ░░▒▒▓▓███"
    "ascii": ShaderRamp(" .\u00b7:-=+*#@"),
    # " .·:-=+*#@"
    "circuit": ShaderRamp(" \u00b7\u2500\u2502\u250c\u2510\u2514\u2518\u253c\u2551"),
    # " ·─│┌┐└┘┼║"
    "organic": ShaderRamp(" .\u00b7\u00b0oO@8&#"),
    # " .·°oO@8&#"
    "minimal": ShaderRamp("    \u00b7\u00b7\u2219\u2219\u2022\u25cf"),
    # "    ··∙∙•●"
    "glass": ShaderRamp("   \u00b7.:\u2591\u2592\u2593\u2588"),
    # "   ·.:░▒▓█"
    "bone": ShaderRamp(" .\u00b7:;+=\u2261#\u2588"),
    # " .·:;+=≡#█"
    "ferrofluid": ShaderRamp(" ~\u223c\u2248\u224b\u223d\u223f\u2307\u2307\u2588"),
    # " ~∼≈≋∽∿⌇⌇█"
    "silk": ShaderRamp("     \u00b7\u00b7..\u2591"),
    # "     ··..░"  (heavy padding at the dark end for a soft look)
    "ceramic": ShaderRamp(" \u00b7.:\u25cb\u25cc\u25cd\u25c9\u25cf\u2588"),
    # " ·.:○◌◍◉●█"
}

# Ordered list for deterministic index-based lookup.
_PRESET_NAMES: list[str] = sorted(SHADER_PRESETS.keys())
_PRESET_COUNT: int = len(_PRESET_NAMES)

DEFAULT_SHADER: ShaderRamp = SHADER_PRESETS["donut"]


# ── shading function ──────────────────────────────────────────────────


def shade(brightness: float, ramp: ShaderRamp) -> str:
    """Map a brightness in [0, 1] to a character from *ramp*.

    Values outside [0, 1] are clamped.  Works with ramps of any length.
    """
    clamped = clamp(brightness, 0.0, 1.0)
    n = len(ramp.chars)
    idx = int(clamped * (n - 1) + 0.5)
    if idx >= n:
        idx = n - 1
    return ramp.chars[idx]


# ── word -> shader mapping ────────────────────────────────────────────


def _stable_hash(word: str) -> int:
    """Deterministic, platform-independent hash of a string.

    Uses FNV-1a so the result is consistent across Python processes
    (unlike the built-in ``hash()`` which is randomized by default).
    """
    h: int = 0x811C9DC5  # FNV offset basis (32-bit)
    for ch in word:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, mask to 32-bit
    return h


def shader_for_word(word: str) -> ShaderRamp:
    """Deterministically select a shader preset for a material_substance word.

    The same word always produces the same shader, regardless of process
    or platform.
    """
    if not word:
        return DEFAULT_SHADER
    idx = _stable_hash(word.lower().strip()) % _PRESET_COUNT
    return SHADER_PRESETS[_PRESET_NAMES[idx]]
