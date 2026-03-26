#!/usr/bin/env python3
"""
Dual-Space Embedding Engine (CLIP + T5)
═══════════════════════════════════════════════════════════════════════════════

Computes visual-semantic (CLIP) and linguistic-semantic (T5) embeddings for
prompt components, enabling dual-space diversity analysis.

CLIP captures how words relate in image-generation space.
T5 captures how words relate in the text-encoder space used by diffusion models.
Together they give a more complete picture of "prompt diversity."

Usage:
    # Compute dual embeddings for current components
    uv run python -m promptweaver.tools.embeddings

    # Compute for a generated candidates file
    uv run python -m promptweaver.tools.embeddings --input candidates.yaml

    # CLIP-only (faster, skip T5)
    uv run python -m promptweaver.tools.embeddings --clip-only

    # Custom models
    uv run python -m promptweaver.tools.embeddings \
        --clip-model openai/clip-vit-large-patch14 \
        --t5-model google/t5-v1_1-xxl

Output:
    - embeddings.npz: Compressed archive with CLIP + T5 embeddings
      Keys: "{category}/{word}/clip", "{category}/{word}/t5", "_metadata"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# Default models — configurable via CLI
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
# flan-t5-large is a good balance of quality/speed. For Flux-equivalent
# fidelity, use google/t5-v1_1-xxl (11B params, needs GPU).
DEFAULT_T5_MODEL = "google/flan-t5-large"

# Batch size for encoding (avoids OOM on large candidate pools)
ENCODE_BATCH_SIZE = 128


class CLIPTextEmbedder:
    """CLIP text encoder — visual-semantic embeddings."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL) -> None:
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP: {model_name} on {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.embedding_dim: int = self.model.config.projection_dim
        self.model.eval()
        logger.info(f"CLIP ready — {self.embedding_dim}d embeddings")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode texts → (n, dim) L2-normalized embeddings."""
        import torch

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch = texts[start : start + ENCODE_BATCH_SIZE]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.text_model(**inputs)
                features = self.model.text_projection(text_outputs.pooler_output)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


class T5TextEmbedder:
    """T5 encoder — linguistic-semantic embeddings via mean-pooled hidden states."""

    def __init__(self, model_name: str = DEFAULT_T5_MODEL) -> None:
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading T5 encoder: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.embedding_dim: int = self.model.config.d_model
        self.model.eval()
        logger.info(f"T5 ready — {self.embedding_dim}d hidden states")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode texts → (n, dim) L2-normalized mean-pooled embeddings."""
        import torch

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch = texts[start : start + ENCODE_BATCH_SIZE]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,  # Match CLIP's token limit
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                hidden = outputs.last_hidden_state  # (batch, seq, dim)

                # Mean-pool over non-padding tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                # L2 normalize
                pooled = pooled / pooled.norm(dim=-1, keepdim=True)
                all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-SPACE CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DualEmbeddings:
    """CLIP + T5 embeddings for a set of words, with joint-space operations."""

    words: list[str]
    clip: np.ndarray  # (n, clip_dim), L2-normalized
    t5: np.ndarray | None  # (n, t5_dim), L2-normalized — None if clip-only

    def joint_similarity_matrix(self, alpha: float = 0.5) -> np.ndarray:
        """
        Weighted combination of CLIP and T5 cosine similarity matrices.

        sim(a,b) = α * clip_cos(a,b) + (1-α) * t5_cos(a,b)

        If T5 is None, returns pure CLIP similarity.
        """
        clip_sim = self.clip @ self.clip.T

        if self.t5 is None:
            return clip_sim

        t5_sim = self.t5 @ self.t5.T
        return alpha * clip_sim + (1.0 - alpha) * t5_sim

    def clip_similarity_matrix(self) -> np.ndarray:
        return self.clip @ self.clip.T

    def t5_similarity_matrix(self) -> np.ndarray | None:
        if self.t5 is None:
            return None
        return self.t5 @ self.t5.T


class DualSpaceEmbedder:
    """Orchestrates CLIP + T5 embedding computation."""

    def __init__(
        self,
        clip_model: str = DEFAULT_CLIP_MODEL,
        t5_model: str | None = DEFAULT_T5_MODEL,
    ) -> None:
        self.clip_embedder = CLIPTextEmbedder(clip_model)
        self.t5_embedder: T5TextEmbedder | None = None
        if t5_model:
            self.t5_embedder = T5TextEmbedder(t5_model)

    def embed(self, words: list[str]) -> DualEmbeddings:
        """Compute dual embeddings for a word list."""
        logger.info(f"Embedding {len(words)} words...")
        clip_emb = self.clip_embedder.encode_batch(words)
        t5_emb = None
        if self.t5_embedder:
            t5_emb = self.t5_embedder.encode_batch(words)
        return DualEmbeddings(words=words, clip=clip_emb, t5=t5_emb)

    def embed_categories(
        self, categories: dict[str, list[str]]
    ) -> dict[str, DualEmbeddings]:
        """Compute dual embeddings per category."""
        results: dict[str, DualEmbeddings] = {}
        for cat, words in categories.items():
            if not words:
                logger.warning(f"Skipping empty category: {cat}")
                continue
            logger.info(f"Category {cat}: {len(words)} words")
            results[cat] = self.embed(words)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def farthest_point_sampling(
    sim_matrix: np.ndarray,
    k: int,
    seed_indices: list[int] | None = None,
) -> list[int]:
    """
    Greedy farthest-point sampling in similarity space.

    Iteratively selects the point with the LOWEST maximum similarity
    to the already-selected set, maximizing coverage.

    Args:
        sim_matrix: (n, n) pairwise similarity matrix
        k: Number of points to select
        seed_indices: Optional starting points

    Returns:
        List of selected indices, ordered by selection
    """
    n = sim_matrix.shape[0]
    if k >= n:
        return list(range(n))

    selected = list(seed_indices) if seed_indices else [0]
    selected_set = set(selected)

    # Track minimum similarity to selected set per candidate
    min_sim = np.full(n, np.inf)
    for idx in selected:
        min_sim = np.minimum(min_sim, sim_matrix[:, idx])

    while len(selected) < k:
        # Mask already-selected
        masked = min_sim.copy()
        for idx in selected_set:
            masked[idx] = np.inf

        best = int(np.argmin(masked))
        if best in selected_set:
            break

        selected.append(best)
        selected_set.add(best)
        min_sim = np.minimum(min_sim, sim_matrix[:, best])

    return selected


def greedy_opposite_pairs(
    sim_matrix: np.ndarray,
    words: list[str],
) -> dict[str, str]:
    """
    Greedy maximum-distance pairing for semantic opposites.

    Repeatedly finds the most distant unpaired pair, assigns them as
    mutual opposites, and removes both from the pool.

    Args:
        sim_matrix: (n, n) pairwise similarity matrix
        words: Corresponding word list

    Returns:
        Dict mapping each word to its opposite
    """
    n = len(words)
    if n < 2:
        return {}

    available = set(range(n))
    opposites: dict[str, str] = {}

    while len(available) >= 2:
        avail = list(available)
        min_sim = float("inf")
        best_pair: tuple[int, int] | None = None

        for idx_a, i in enumerate(avail):
            for j in avail[idx_a + 1 :]:
                if sim_matrix[i, j] < min_sim:
                    min_sim = sim_matrix[i, j]
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        opposites[words[i]] = words[j]
        opposites[words[j]] = words[i]
        available.discard(i)
        available.discard(j)

    # Odd one out: pair with most distant overall
    if len(available) == 1:
        last = available.pop()
        sims = sim_matrix[last].copy()
        sims[last] = np.inf  # Exclude self
        best_opp = int(np.argmin(sims))
        opposites[words[last]] = words[best_opp]

    return opposites


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DiversityStats:
    """Diversity metrics for a set of embeddings."""

    count: int
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    std_similarity: float
    most_similar_pair: tuple[str, str, float]
    most_distant_pair: tuple[str, str, float]
    redundant_pairs: int  # Above threshold
    redundancy_threshold: float


def analyze_diversity(
    sim_matrix: np.ndarray,
    words: list[str],
    redundancy_threshold: float = 0.85,
) -> DiversityStats:
    """Compute diversity stats for a similarity matrix."""
    n = len(words)
    mask = ~np.eye(n, dtype=bool)
    pairwise = sim_matrix[mask]

    # Most similar
    sim_copy = sim_matrix.copy()
    np.fill_diagonal(sim_copy, -np.inf)
    max_idx = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)

    # Most distant
    np.fill_diagonal(sim_copy, np.inf)
    min_idx = np.unravel_index(np.argmin(sim_copy), sim_copy.shape)

    redundant = int(np.sum(sim_matrix[mask] > redundancy_threshold) // 2)

    return DiversityStats(
        count=n,
        mean_similarity=float(pairwise.mean()),
        min_similarity=float(pairwise.min()),
        max_similarity=float(pairwise.max()),
        std_similarity=float(pairwise.std()),
        most_similar_pair=(
            words[max_idx[0]],
            words[max_idx[1]],
            float(sim_matrix[max_idx]),
        ),
        most_distant_pair=(
            words[min_idx[0]],
            words[min_idx[1]],
            float(sim_matrix[min_idx]),
        ),
        redundant_pairs=redundant,
        redundancy_threshold=redundancy_threshold,
    )


def cross_category_contamination(
    category_embeddings: dict[str, DualEmbeddings],
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Check if any words are closer to another category's centroid than their own.

    Returns list of contaminated items with details.
    """
    # Compute centroids per category (in joint space)
    centroids: dict[str, np.ndarray] = {}
    for cat, emb in category_embeddings.items():
        clip_centroid = emb.clip.mean(axis=0)
        clip_centroid /= np.linalg.norm(clip_centroid)

        if emb.t5 is not None:
            t5_centroid = emb.t5.mean(axis=0)
            t5_centroid /= np.linalg.norm(t5_centroid)
            # Store both — we'll compute joint sim below
            centroids[cat] = np.concatenate([clip_centroid, t5_centroid])
        else:
            centroids[cat] = clip_centroid

    contaminated: list[dict[str, Any]] = []
    cat_names = list(centroids.keys())

    for cat, emb in category_embeddings.items():
        for i, word in enumerate(emb.words):
            # Compute similarity to each centroid
            sims: dict[str, float] = {}
            for other_cat in cat_names:
                other_centroid = centroids[other_cat]

                if emb.t5 is not None:
                    clip_part = other_centroid[: emb.clip.shape[1]]
                    t5_part = other_centroid[emb.clip.shape[1] :]
                    clip_sim = float(np.dot(emb.clip[i], clip_part))
                    t5_sim = float(np.dot(emb.t5[i], t5_part))
                    sims[other_cat] = alpha * clip_sim + (1 - alpha) * t5_sim
                else:
                    sims[other_cat] = float(np.dot(emb.clip[i], other_centroid))

            assigned_sim = sims[cat]
            closest_cat = max(sims, key=lambda c: sims[c])

            if closest_cat != cat:
                contaminated.append(
                    {
                        "word": word,
                        "assigned": cat,
                        "closest": closest_cat,
                        "sim_assigned": assigned_sim,
                        "sim_closest": sims[closest_cat],
                        "delta": sims[closest_cat] - assigned_sim,
                    }
                )

    return sorted(contaminated, key=lambda x: x["delta"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════


def save_embeddings(
    category_embeddings: dict[str, DualEmbeddings],
    output_path: Path,
    clip_model: str,
    t5_model: str | None,
) -> None:
    """Save all embeddings to compressed npz archive."""
    save_dict: dict[str, np.ndarray] = {}

    for cat, emb in category_embeddings.items():
        for i, word in enumerate(emb.words):
            save_dict[f"{cat}/{word}/clip"] = emb.clip[i]
            if emb.t5 is not None:
                save_dict[f"{cat}/{word}/t5"] = emb.t5[i]

    metadata = {
        "clip_model": clip_model,
        "t5_model": t5_model,
        "clip_dim": int(next(iter(category_embeddings.values())).clip.shape[1]),
        "t5_dim": (
            int(next(iter(category_embeddings.values())).t5.shape[1])
            if next(iter(category_embeddings.values())).t5 is not None
            else None
        ),
        "categories": {
            cat: len(emb.words) for cat, emb in category_embeddings.items()
        },
        "total_words": sum(len(e.words) for e in category_embeddings.values()),
        "timestamp": datetime.now().isoformat(),
    }
    save_dict["_metadata_json"] = np.array(json.dumps(metadata))

    np.savez_compressed(output_path, **save_dict)
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved embeddings to {output_path} ({size_kb:.1f} KB)")


def load_embeddings(path: Path) -> dict[str, DualEmbeddings]:
    """Load embeddings from npz archive, grouped by category."""
    data = np.load(path, allow_pickle=True)

    metadata = json.loads(str(data["_metadata_json"]))
    has_t5 = metadata.get("t5_model") is not None

    # Group by category
    cat_words: dict[str, list[str]] = {}
    cat_clip: dict[str, list[np.ndarray]] = {}
    cat_t5: dict[str, list[np.ndarray]] = {}

    for key in data.files:
        if key.startswith("_"):
            continue

        parts = key.rsplit("/", 1)
        if len(parts) != 2:
            continue

        cat_word, space = parts
        cat, word = cat_word.split("/", 1)

        if cat not in cat_words:
            cat_words[cat] = []
            cat_clip[cat] = []
            cat_t5[cat] = []

        if space == "clip":
            cat_words[cat].append(word)
            cat_clip[cat].append(data[key])
        elif space == "t5":
            cat_t5[cat].append(data[key])

    results: dict[str, DualEmbeddings] = {}
    for cat in cat_words:
        results[cat] = DualEmbeddings(
            words=cat_words[cat],
            clip=np.array(cat_clip[cat]),
            t5=np.array(cat_t5[cat]) if has_t5 and cat_t5.get(cat) else None,
        )

    return results


def load_components_yaml(path: Path) -> dict[str, list[str]]:
    """Load component words from YAML (supports both dict and list formats)."""
    with open(path) as f:
        data = yaml.safe_load(f)

    categories: dict[str, list[str]] = {}

    components = data.get("components", data)
    for cat, items in components.items():
        if not items:
            continue
        if isinstance(items[0], dict):
            categories[cat] = [item["word"] for item in items]
        else:
            categories[cat] = list(items)

    return categories


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_analysis(
    category_embeddings: dict[str, DualEmbeddings],
    alpha: float,
) -> None:
    """Print diversity analysis for all categories."""
    print(f"\n{'=' * 75}")
    print("EMBEDDING ANALYSIS")
    print(f"{'=' * 75}")

    for cat, emb in sorted(category_embeddings.items()):
        sim = emb.joint_similarity_matrix(alpha)
        stats = analyze_diversity(sim, emb.words)

        print(f"\n{cat.upper()} ({stats.count} items)")
        print(
            f"  Similarity: min={stats.min_similarity:.3f}  "
            f"mean={stats.mean_similarity:.3f}  max={stats.max_similarity:.3f}  "
            f"std={stats.std_similarity:.3f}"
        )
        print(
            f"  Most similar:  '{stats.most_similar_pair[0]}' <-> "
            f"'{stats.most_similar_pair[1]}' ({stats.most_similar_pair[2]:.3f})"
        )
        print(
            f"  Most distant:  '{stats.most_distant_pair[0]}' <-> "
            f"'{stats.most_distant_pair[1]}' ({stats.most_distant_pair[2]:.3f})"
        )
        if stats.redundant_pairs > 0:
            print(f"  WARNING: {stats.redundant_pairs} redundant pairs (>{stats.redundancy_threshold})")

    # Cross-category contamination
    contaminated = cross_category_contamination(category_embeddings, alpha)
    if contaminated:
        print(f"\n{'─' * 75}")
        print(f"CROSS-CATEGORY CONTAMINATION ({len(contaminated)} items)")
        print(f"{'─' * 75}")
        for item in contaminated[:15]:
            print(
                f"  '{item['word']}' — assigned:{item['assigned']}, "
                f"closer to:{item['closest']} "
                f"(delta={item['delta']:+.3f})"
            )
        if len(contaminated) > 15:
            print(f"  ... and {len(contaminated) - 15} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dual CLIP+T5 embeddings for prompt components"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input YAML (default: data/components.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for embeddings .npz",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help=f"CLIP model (default: {DEFAULT_CLIP_MODEL})",
    )
    parser.add_argument(
        "--t5-model",
        default=DEFAULT_T5_MODEL,
        help=f"T5 model (default: {DEFAULT_T5_MODEL})",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Skip T5, use CLIP embeddings only",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for CLIP vs T5 in joint similarity (default: 0.5)",
    )
    parser.add_argument(
        "--update-opposites",
        action="store_true",
        help="Recompute and write opposites back to components YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without writing files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths
    data_dir = Path(__file__).parent.parent / "data"
    input_path = args.input or data_dir / "components.yaml"
    output_path = args.output or data_dir / "embeddings.npz"

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    t5_model = None if args.clip_only else args.t5_model

    print("=" * 75)
    print("Dual-Space Embedding Computation")
    print("=" * 75)
    print(f"  CLIP model:  {args.clip_model}")
    print(f"  T5 model:    {t5_model or '(disabled)'}")
    print(f"  Alpha:       {args.alpha}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")

    # Load words
    categories = load_components_yaml(input_path)
    total = sum(len(w) for w in categories.values())
    print(f"  Categories:  {len(categories)} ({total} total words)")

    # Compute embeddings
    embedder = DualSpaceEmbedder(
        clip_model=args.clip_model,
        t5_model=t5_model,
    )
    category_embeddings = embedder.embed_categories(categories)

    # Analysis
    print_analysis(category_embeddings, args.alpha)

    # Optionally update opposites
    if args.update_opposites and not args.dry_run:
        negatable = {
            "color_logic",
            "light_behavior",
            "atmosphere_field",
            "temporal_state",
            "texture_density",
            "medium_render",
        }

        with open(input_path) as f:
            yaml_data = yaml.safe_load(f)

        for cat, emb in category_embeddings.items():
            if cat not in negatable:
                continue
            sim = emb.joint_similarity_matrix(args.alpha)
            opposites = greedy_opposite_pairs(sim, emb.words)

            items = yaml_data.get("components", {}).get(cat, [])
            for item in items:
                word = item["word"] if isinstance(item, dict) else item
                if word in opposites:
                    if isinstance(item, dict):
                        item["opposite"] = opposites[word]

        with open(input_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nUpdated opposites in {input_path}")

    # Save embeddings
    if not args.dry_run:
        save_embeddings(category_embeddings, output_path, args.clip_model, t5_model)
    else:
        print(f"\n[DRY RUN] Would save to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
