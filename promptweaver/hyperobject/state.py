"""Visual state accumulator — component persistence across prompts.

Each template uses only a subset of the 12 component categories.  The
VisualState holds all 12 slots and updates only the ones present in each
new prompt; the rest carry forward from prior prompts.  Over many
generations the viewport accumulates a unique visual identity shaped by
the full history, not just the current prompt.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Iterator

from ..models import GeneratedPrompt

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────

# Sensible initial values for first launch (before any prompt is generated).
INITIAL_STATE: dict[str, list[str]] = {
    "subject_form":       ["sphere"],
    "material_substance": ["glass"],
    "texture_density":    ["smooth"],
    "light_behavior":     ["soft ambient light"],
    "color_logic":        ["monochromatic"],
    "atmosphere_field":   ["dust motes"],
    "phenomenon_pattern": ["crystallization"],
    "spatial_logic":      ["symmetrical"],
    "scale_perspective":  ["eye level"],
    "temporal_state":     ["suspended"],
    "setting_location":   ["void"],
    "medium_render":      ["3d render"],
}

ALL_CATEGORIES: list[str] = list(INITIAL_STATE.keys())

# ── visual state ──────────────────────────────────────────────────────────


@dataclass
class VisualState:
    """Accumulated visual parameters -- one slot per component category.

    Each slot holds the most recent component word(s) for that category.
    Updated incrementally: only categories present in the new prompt change.
    """

    slots: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in INITIAL_STATE.items()}
    )

    # ── mutation ──────────────────────────────────────────────────────

    def apply_prompt(self, prompt: GeneratedPrompt) -> set[str]:
        """Update slots from *prompt* components.

        Only categories present in the prompt are touched; all others carry
        forward from their last update.

        Returns:
            The set of category names whose values actually changed.

        Raises:
            TypeError: If *prompt* is not a ``GeneratedPrompt``.
        """
        if not isinstance(prompt, GeneratedPrompt):
            raise TypeError(
                f"Expected GeneratedPrompt, got {type(prompt).__name__}"
            )

        changed: set[str] = set()
        for category, words in prompt.components.items():
            if not isinstance(words, list):
                logger.warning(
                    "Skipping category %r: expected list, got %s",
                    category,
                    type(words).__name__,
                )
                continue
            if self.slots.get(category) != words:
                self.slots[category] = list(words)
                changed.add(category)

        if changed:
            logger.debug(
                "VisualState updated %d/%d categories: %s",
                len(changed),
                len(prompt.components),
                ", ".join(sorted(changed)),
            )
        return changed

    def reset(self) -> None:
        """Reset all slots back to ``INITIAL_STATE``."""
        self.slots = {k: list(v) for k, v in INITIAL_STATE.items()}

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, category: str) -> list[str]:
        """Get current word list for *category*.

        Falls back to ``INITIAL_STATE`` if the category is known but
        absent from slots, or returns an empty list for unknown categories.
        """
        return self.slots.get(category, INITIAL_STATE.get(category, []))

    def get_single(self, category: str) -> str:
        """Get the first word for *category* (convenience for single-value slots).

        Returns an empty string when the category is empty or unknown.
        """
        words = self.get(category)
        return words[0] if words else ""

    def all_words(self) -> list[str]:
        """Flat list of all current component words across every slot."""
        result: list[str] = []
        for words in self.slots.values():
            result.extend(words)
        return result

    # ── introspection ─────────────────────────────────────────────────

    @property
    def filled_categories(self) -> list[str]:
        """Categories that have non-empty word lists."""
        return [cat for cat, words in self.slots.items() if words]

    def __len__(self) -> int:
        """Number of categories currently tracked (always 12 by default)."""
        return len(self.slots)

    def __contains__(self, category: str) -> bool:
        return category in self.slots

    def __iter__(self) -> Iterator[str]:
        return iter(self.slots)

    def snapshot(self) -> dict[str, list[str]]:
        """Return a deep copy of the current slots for external use."""
        return copy.deepcopy(self.slots)
