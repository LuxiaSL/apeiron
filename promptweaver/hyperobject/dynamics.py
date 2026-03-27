"""Embedding-driven animation modifiers.

Uses ``EmbeddingCache`` and ``VisualState`` to compute per-prompt dynamics
that drive animation speed, vertex displacement, and visual effects in the
Hyperobject renderer.

All computation is pure numpy linear algebra on L2-normalized vectors.
When embeddings are unavailable, neutral defaults are returned so the
renderer can still function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .embedding_cache import EmbeddingCache, _HAS_NUMPY
from .state import VisualState

if _HAS_NUMPY:
    import numpy as np

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────

_DEFAULT_ENERGY: float = 0.45
_DEFAULT_TENSION: float = 0.0
_DEFAULT_VOID_PROXIMITY: float = 0.5


# ── data ──────────────────────────────────────────────────────────────────


@dataclass
class PromptDynamics:
    """Computed per-prompt metrics that drive animation behavior.

    Attributes
    ----------
    energy:
        0.0--1.0.  Mean pairwise cosine distance among all component
        vectors.  Drives animation speed and intensity.
    tension:
        0.0--1.0.  Maximum cross-category pairwise cosine similarity.
        High values (>0.7) indicate semantic interference between
        categories.
    void_proximity:
        0.0--1.0.  ``1 - mean(similarity to category centroids)``.
        High values mean the prompt occupies an under-explored
        semantic region; low values mean it sits near well-populated
        cluster centres.
    gravity_poles:
        3-D unit directions derived from component CLIP vectors.
        Used for per-vertex displacement each frame.
    """

    energy: float = _DEFAULT_ENERGY
    tension: float = _DEFAULT_TENSION
    void_proximity: float = _DEFAULT_VOID_PROXIMITY
    gravity_poles: list[tuple[float, float, float]] = field(default_factory=list)


# ── computation ───────────────────────────────────────────────────────────


def compute_dynamics(
    state: VisualState,
    cache: EmbeddingCache | None,
) -> PromptDynamics:
    """Derive animation dynamics from the current visual state.

    Parameters
    ----------
    state:
        The accumulated visual state (all 12 category slots, including
        carry-forward values from prior prompts).
    cache:
        Pre-loaded embedding cache.  Pass ``None`` (or an unavailable
        cache) to get neutral defaults.

    Returns
    -------
    PromptDynamics
        Metrics that the renderer uses to modulate animation.
    """
    if cache is None or not cache.available or not _HAS_NUMPY:
        logger.debug("Embedding cache unavailable -- returning default dynamics")
        return PromptDynamics()

    try:
        return _compute(state, cache)
    except Exception:
        logger.exception("Dynamics computation failed -- returning defaults")
        return PromptDynamics()


def _compute(state: VisualState, cache: EmbeddingCache) -> PromptDynamics:
    """Internal implementation -- assumes numpy + cache are available."""

    vecs = cache.prompt_vectors(state.slots)
    n = vecs.shape[0]

    # ── energy: mean pairwise cosine distance ─────────────────────────
    energy = _DEFAULT_ENERGY
    if n >= 2:
        sim_matrix = cache.pairwise_similarity(vecs)
        mask = ~np.eye(n, dtype=bool)
        mean_sim = float(sim_matrix[mask].mean())
        energy = float(np.clip(1.0 - mean_sim, 0.0, 1.0))

    # ── tension: max cross-category pairwise similarity ───────────────
    tension = _DEFAULT_TENSION
    if n >= 2:
        # Build index→category mapping so we can identify cross-category pairs
        idx_to_cat: list[str] = []
        for category, words in state.slots.items():
            for w in words:
                if cache.lookup(w) is not None:
                    idx_to_cat.append(category)

        if len(idx_to_cat) == n:
            max_cross_sim = _DEFAULT_TENSION
            for i in range(n):
                for j in range(i + 1, n):
                    if idx_to_cat[i] != idx_to_cat[j]:
                        s = float(sim_matrix[i, j])
                        if s > max_cross_sim:
                            max_cross_sim = s
            tension = float(np.clip(max_cross_sim, 0.0, 1.0))

    # ── void proximity: 1 - mean similarity to category centroids ─────
    void_proximity = _DEFAULT_VOID_PROXIMITY
    if n >= 1:
        mean_vec = vecs.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec = mean_vec / norm
            centroid_sims = cache.centroid_similarities(mean_vec)
            if centroid_sims.size > 0:
                void_proximity = float(
                    np.clip(1.0 - centroid_sims.mean(), 0.0, 1.0)
                )

    # ── gravity poles: first 3 components of each CLIP vector ─────────
    gravity_poles: list[tuple[float, float, float]] = []
    for words in state.slots.values():
        for w in words:
            vec = cache.lookup(w)
            if vec is not None and vec.shape[0] >= 3:
                direction = vec[:3].astype(float)
                dir_norm = float(np.linalg.norm(direction))
                if dir_norm > 0.0:
                    direction = direction / dir_norm
                gravity_poles.append(
                    (float(direction[0]), float(direction[1]), float(direction[2]))
                )

    return PromptDynamics(
        energy=energy,
        tension=tension,
        void_proximity=void_proximity,
        gravity_poles=gravity_poles,
    )
