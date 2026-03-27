"""Numpy-only embedding loader for pre-computed CLIP/T5 vectors.

Loads ``curated_embeddings.npz`` (or a compatible archive) at init and
provides fast word-to-vector lookup, category centroids, and pairwise
similarity operations.  No torch, no transformers -- pure numpy.

Graceful degradation:
- If numpy is not installed the module is still importable; methods
  raise ``RuntimeError`` on use.
- If the ``.npz`` file is missing the constructor logs a warning and
  the instance reports ``available == False``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── numpy soft-import ─────────────────────────────────────────────────────

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _require_numpy() -> None:
    """Raise if numpy is unavailable."""
    if not _HAS_NUMPY:
        raise RuntimeError(
            "numpy is required for EmbeddingCache but is not installed. "
            "Install it with: pip install numpy"
        )


# ── default path ──────────────────────────────────────────────────────────

DEFAULT_NPZ_PATH: Path = (
    Path(__file__).parent.parent / "data" / "curated_embeddings.npz"
)


# ── cache ─────────────────────────────────────────────────────────────────


class EmbeddingCache:
    """Numpy-only embedding lookup.  No torch required.

    Parameters
    ----------
    npz_path:
        Path to a ``.npz`` archive with keys like
        ``{category}/{word}/clip`` and ``{category}/{word}/t5``, plus a
        ``_metadata_json`` key containing JSON metadata.

    Attributes
    ----------
    available:
        ``True`` when the cache loaded successfully and is ready to use.
    clip:
        ``dict[str, np.ndarray]`` mapping word -> 512-d CLIP vector.
    centroids:
        ``dict[str, np.ndarray]`` mapping category -> L2-normalized
        mean CLIP vector for all words in that category.
    metadata:
        Raw metadata dict from the ``.npz`` file.
    """

    def __init__(self, npz_path: Path | None = None) -> None:
        self.clip: dict[str, Any] = {}
        self.centroids: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.available: bool = False

        resolved_path = npz_path if npz_path is not None else DEFAULT_NPZ_PATH

        if not _HAS_NUMPY:
            logger.warning(
                "numpy not installed -- EmbeddingCache will be unavailable"
            )
            return

        if not resolved_path.exists():
            logger.warning(
                "Embedding file not found: %s -- EmbeddingCache will be unavailable",
                resolved_path,
            )
            return

        try:
            self._load(resolved_path)
            self.available = True
            logger.info(
                "EmbeddingCache loaded: %d words, %d category centroids",
                len(self.clip),
                len(self.centroids),
            )
        except Exception:
            logger.exception("Failed to load embeddings from %s", resolved_path)

    # ── loading ───────────────────────────────────────────────────────

    def _load(self, npz_path: Path) -> None:
        """Parse the npz archive and build lookup structures."""
        _require_numpy()

        data = np.load(str(npz_path), allow_pickle=True)

        # Metadata
        if "_metadata_json" in data.files:
            self.metadata = json.loads(str(data["_metadata_json"]))

        # Build word -> CLIP vector lookup and group words by category
        category_words: dict[str, list[str]] = {}

        for key in data.files:
            if key.startswith("_"):
                continue
            if not key.endswith("/clip"):
                continue

            # Key format: {category}/{word}/clip
            parts = key.rsplit("/", 1)  # ["{cat}/{word}", "clip"]
            if len(parts) != 2:
                continue
            cat_word = parts[0]

            slash_idx = cat_word.find("/")
            if slash_idx < 0:
                continue
            category = cat_word[:slash_idx]
            word = cat_word[slash_idx + 1 :]

            vec = data[key]
            if vec.ndim == 0:
                continue  # scalar / corrupt entry

            self.clip[word] = vec.astype(np.float32)
            category_words.setdefault(category, []).append(word)

        # Precompute per-category centroids (mean of member vectors, re-normalized)
        for cat, words in category_words.items():
            vecs = [self.clip[w] for w in words if w in self.clip]
            if not vecs:
                continue
            stacked = np.stack(vecs)
            centroid = stacked.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0.0:
                centroid = centroid / norm
            self.centroids[cat] = centroid

    # ── public API ────────────────────────────────────────────────────

    def lookup(self, word: str) -> Any:
        """Get the CLIP vector for *word*, or ``None`` if not found.

        Returns
        -------
        np.ndarray | None
            512-d float32 vector, or ``None``.
        """
        if not self.available:
            return None
        return self.clip.get(word)

    def prompt_vectors(self, components: dict[str, list[str]]) -> Any:
        """Collect CLIP vectors for all component words.

        Parameters
        ----------
        components:
            ``{category: [word, ...]}`` mapping (e.g. from
            ``VisualState.slots`` or ``GeneratedPrompt.components``).

        Returns
        -------
        np.ndarray
            ``(N, clip_dim)`` array of matched vectors.  If no words
            are found, returns a ``(1, 512)`` zero array so downstream
            math does not break.
        """
        _require_numpy()

        if not self.available:
            return np.zeros((1, 512), dtype=np.float32)

        vecs: list[Any] = []
        for words in components.values():
            for w in words:
                vec = self.clip.get(w)
                if vec is not None:
                    vecs.append(vec)

        if not vecs:
            clip_dim = self.metadata.get("clip_dim", 512)
            return np.zeros((1, clip_dim), dtype=np.float32)

        return np.stack(vecs)

    def pairwise_similarity(self, vecs: Any) -> Any:
        """Compute NxN cosine similarity matrix.

        Because all vectors are L2-normalized, this is simply ``vecs @ vecs.T``.

        Parameters
        ----------
        vecs:
            ``(N, D)`` array of L2-normalized vectors.

        Returns
        -------
        np.ndarray
            ``(N, N)`` similarity matrix with values in [-1, 1].
        """
        _require_numpy()
        return np.asarray(vecs, dtype=np.float32) @ np.asarray(vecs, dtype=np.float32).T

    def centroid_similarities(self, mean_vec: Any) -> Any:
        """Similarity of *mean_vec* to each category centroid.

        Parameters
        ----------
        mean_vec:
            ``(D,)`` L2-normalized vector.

        Returns
        -------
        np.ndarray
            1-d array of cosine similarities, one per category (in the
            same order as ``self.centroids``).  Empty array if no
            centroids are available.
        """
        _require_numpy()

        if not self.centroids:
            return np.array([], dtype=np.float32)

        mean_vec = np.asarray(mean_vec, dtype=np.float32)
        return np.array(
            [float(np.dot(mean_vec, c)) for c in self.centroids.values()],
            dtype=np.float32,
        )
