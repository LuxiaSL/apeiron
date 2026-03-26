#!/usr/bin/env python3
"""
Diversity Selection Pipeline
═══════════════════════════════════════════════════════════════════════════════

Takes generated candidates, computes dual CLIP+T5 embeddings, and selects the
most diverse subset using farthest-point sampling in joint embedding space.

This is the core quality gate — it turns a noisy pool of LLM-generated words
into a tight, maximally-spread set where every item earns its place.

Pipeline:
    1. Load generated candidates YAML
    2. Compute dual-space embeddings (CLIP + T5)
    3. Cross-category contamination filter (drop misclassified items)
    4. Redundancy pre-filter (drop items > threshold similarity to existing)
    5. Farthest-point sampling per category in joint space
    6. Compute diversity elbow (where adding items starts diluting)
    7. Recompute semantic opposites for negatable categories
    8. Output final components YAML

Usage:
    # Full pipeline from candidates → selected components
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates_20260325.yaml

    # Custom selection size and alpha
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates.yaml \
        --k 80 --alpha 0.6

    # CLIP-only (faster, skip T5)
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates.yaml --clip-only

    # Dry run (analyze without writing)
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualEmbeddings,
    DualSpaceEmbedder,
    DiversityStats,
    analyze_diversity,
    cross_category_contamination,
    farthest_point_sampling,
    greedy_opposite_pairs,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

# Categories that get semantic opposites for negative prompt generation
NEGATABLE_CATEGORIES = frozenset({
    "color_logic", "light_behavior", "atmosphere_field",
    "temporal_state", "texture_density", "medium_render",
})


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    """Result of the selection pipeline for one category."""

    category: str
    candidates_count: int
    filtered_count: int  # After contamination + redundancy filter
    selected_count: int
    selected_words: list[str]
    opposites: dict[str, str]  # word → opposite (negatable categories only)
    before_stats: DiversityStats
    after_stats: DiversityStats
    elbow_k: int | None  # Where diversity starts dropping sharply
    contaminated_dropped: list[str]
    redundant_dropped: list[str]


def filter_contaminated(
    category: str,
    words: list[str],
    embeddings: DualEmbeddings,
    all_category_embeddings: dict[str, DualEmbeddings],
    alpha: float,
) -> tuple[list[str], list[str]]:
    """
    Drop words that are closer to another category's centroid than their own.

    Returns (kept_words, dropped_words).
    """
    contaminated = cross_category_contamination(
        {**all_category_embeddings, category: embeddings},
        alpha=alpha,
    )

    # Only drop items from the category we're currently filtering
    drop_words = {
        item["word"] for item in contaminated
        if item["assigned"] == category and item["delta"] > 0.01  # Small margin
    }

    kept = [w for w in words if w not in drop_words]
    dropped = [w for w in words if w in drop_words]

    if dropped:
        logger.info(f"  Contamination filter: dropped {len(dropped)} items")
        for w in dropped[:5]:
            match = next(
                (c for c in contaminated if c["word"] == w), None
            )
            if match:
                logger.info(
                    f"    '{w}' → closer to {match['closest']} "
                    f"(delta={match['delta']:+.3f})"
                )
        if len(dropped) > 5:
            logger.info(f"    ... and {len(dropped) - 5} more")

    return kept, dropped


def filter_redundant(
    words: list[str],
    sim_matrix: np.ndarray,
    threshold: float = 0.88,
) -> tuple[list[str], list[str]]:
    """
    Drop words that are too similar to another word already in the pool.
    When a redundant pair is found, drop the one that appears later in the list
    (preserving seed ordering).

    Returns (kept_words, dropped_words).
    """
    n = len(words)
    drop_indices: set[int] = set()

    for i in range(n):
        if i in drop_indices:
            continue
        for j in range(i + 1, n):
            if j in drop_indices:
                continue
            if sim_matrix[i, j] > threshold:
                # Drop the later one (seeds come first in the list)
                drop_indices.add(j)

    kept = [w for i, w in enumerate(words) if i not in drop_indices]
    dropped = [w for i, w in enumerate(words) if i in drop_indices]

    if dropped:
        logger.info(f"  Redundancy filter (>{threshold}): dropped {len(dropped)} items")

    return kept, dropped


def compute_elbow(
    sim_matrix: np.ndarray,
    max_k: int | None = None,
) -> tuple[int, list[float]]:
    """
    Run farthest-point sampling incrementally and track the minimum distance
    of each newly added point. The "elbow" is where this distance starts
    dropping sharply — additional items are no longer adding much diversity.

    Returns (elbow_k, distances_per_step).
    """
    n = sim_matrix.shape[0]
    max_k = max_k or n

    # Start with two most distant points
    masked = sim_matrix.copy()
    np.fill_diagonal(masked, np.inf)
    min_idx = np.unravel_index(np.argmin(masked), masked.shape)
    selected = [min_idx[0], min_idx[1]]
    selected_set = set(selected)

    min_sim = np.minimum(sim_matrix[:, selected[0]], sim_matrix[:, selected[1]])
    distances: list[float] = []  # Min distance of each added point

    while len(selected) < min(max_k, n):
        sim_masked = min_sim.copy()
        for idx in selected_set:
            sim_masked[idx] = np.inf

        best = int(np.argmin(sim_masked))
        if best in selected_set:
            break

        # Record the "diversity contribution" of this point
        distances.append(float(1.0 - sim_masked[best]))  # Convert similarity to distance

        selected.append(best)
        selected_set.add(best)
        min_sim = np.minimum(min_sim, sim_matrix[:, best])

    if len(distances) < 3:
        return len(distances) + 2, distances

    # Find elbow: largest drop in the distance curve
    diffs = [distances[i] - distances[i + 1] for i in range(len(distances) - 1)]
    elbow_idx = int(np.argmax(diffs))
    elbow_k = elbow_idx + 3  # +2 for initial pair, +1 for 0-indexing

    return elbow_k, distances


def select_category(
    category: str,
    candidates: list[str],
    embedder: DualSpaceEmbedder,
    all_category_embeddings: dict[str, DualEmbeddings],
    k: int,
    alpha: float,
    redundancy_threshold: float,
    auto_k: bool = False,
) -> SelectionResult:
    """Run the full selection pipeline for one category."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Category: {category} ({len(candidates)} candidates)")
    logger.info(f"{'─'*60}")

    # Embed candidates
    emb = embedder.embed(candidates)

    # Stats before any filtering
    sim_before = emb.joint_similarity_matrix(alpha)
    before_stats = analyze_diversity(sim_before, candidates)

    # Stage 1: Cross-category contamination filter
    kept, contaminated_dropped = filter_contaminated(
        category, candidates, emb, all_category_embeddings, alpha,
    )

    # Re-embed if we dropped anything (indices changed)
    if contaminated_dropped:
        emb = embedder.embed(kept)

    # Stage 2: Redundancy pre-filter
    sim_filtered = emb.joint_similarity_matrix(alpha)
    kept, redundant_dropped = filter_redundant(
        kept, sim_filtered, threshold=redundancy_threshold,
    )

    if redundant_dropped:
        emb = embedder.embed(kept)

    filtered_count = len(kept)
    logger.info(f"  After filters: {filtered_count} candidates remain")

    # Stage 3: Compute elbow to find natural diversity cutoff
    sim_final = emb.joint_similarity_matrix(alpha)
    elbow_k, distances = compute_elbow(sim_final, max_k=min(k + 20, filtered_count))

    if auto_k:
        effective_k = min(elbow_k, filtered_count)
        logger.info(f"  Elbow detected at k={elbow_k}, using auto-k={effective_k}")
    else:
        effective_k = min(k, filtered_count)
        logger.info(f"  Elbow at k={elbow_k} (using requested k={effective_k})")

    # Stage 4: Farthest-point sampling
    selected_indices = farthest_point_sampling(
        sim_matrix=sim_final,
        k=effective_k,
    )
    selected_words = [kept[i] for i in selected_indices]

    # Stats after selection
    selected_emb = DualEmbeddings(
        words=selected_words,
        clip=emb.clip[selected_indices],
        t5=emb.t5[selected_indices] if emb.t5 is not None else None,
    )
    sim_after = selected_emb.joint_similarity_matrix(alpha)
    after_stats = analyze_diversity(sim_after, selected_words)

    logger.info(
        f"  Diversity: mean sim {before_stats.mean_similarity:.3f} → "
        f"{after_stats.mean_similarity:.3f} "
        f"(improved {before_stats.mean_similarity - after_stats.mean_similarity:+.3f})"
    )
    logger.info(
        f"  Most similar pair: '{after_stats.most_similar_pair[0]}' <-> "
        f"'{after_stats.most_similar_pair[1]}' ({after_stats.most_similar_pair[2]:.3f})"
    )

    # Stage 5: Compute opposites for negatable categories
    opposites: dict[str, str] = {}
    if category in NEGATABLE_CATEGORIES:
        opposites = greedy_opposite_pairs(sim_after, selected_words)
        logger.info(f"  Computed {len(opposites)} opposite pairs")

    return SelectionResult(
        category=category,
        candidates_count=len(candidates),
        filtered_count=filtered_count,
        selected_count=len(selected_words),
        selected_words=selected_words,
        opposites=opposites,
        before_stats=before_stats,
        after_stats=after_stats,
        elbow_k=elbow_k,
        contaminated_dropped=contaminated_dropped,
        redundant_dropped=redundant_dropped,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def build_components_yaml(
    results: dict[str, SelectionResult],
) -> dict[str, Any]:
    """Build components YAML structure from selection results."""
    components: dict[str, list[dict[str, Any]]] = {}

    for cat, result in sorted(results.items()):
        items: list[dict[str, Any]] = []
        for word in result.selected_words:
            entry: dict[str, Any] = {"word": word}
            if word in result.opposites:
                entry["opposite"] = result.opposites[word]
            items.append(entry)
        components[cat] = items

    return {"components": components}


def build_analysis_yaml(
    results: dict[str, SelectionResult],
    alpha: float,
) -> dict[str, Any]:
    """Build analysis/metadata YAML for review."""
    analysis: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "alpha": alpha,
        "categories": {},
    }

    for cat, result in sorted(results.items()):
        analysis["categories"][cat] = {
            "candidates": result.candidates_count,
            "after_filters": result.filtered_count,
            "selected": result.selected_count,
            "elbow_k": result.elbow_k,
            "contaminated_dropped": len(result.contaminated_dropped),
            "redundant_dropped": len(result.redundant_dropped),
            "before_mean_sim": round(result.before_stats.mean_similarity, 4),
            "after_mean_sim": round(result.after_stats.mean_similarity, 4),
            "improvement": round(
                result.before_stats.mean_similarity - result.after_stats.mean_similarity, 4
            ),
            "most_similar_pair": {
                "words": list(result.after_stats.most_similar_pair[:2]),
                "similarity": round(result.after_stats.most_similar_pair[2], 4),
            },
        }

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select most diverse components from generated candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Generated candidates YAML (output of generate.py)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output components YAML (default: data/selected_components_<timestamp>.yaml)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=80,
        help="Target components per category (default: 80)",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Use elbow detection to automatically determine k per category",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="CLIP vs T5 weight in joint similarity (default: 0.5)",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.88,
        help="Drop candidates above this similarity to any kept item (default: 0.88)",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Skip T5, use CLIP embeddings only",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model",
    )
    parser.add_argument(
        "--t5-model",
        default="google/flan-t5-large",
        help="T5 model",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Process only these categories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without writing output files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Load candidates
    candidates_by_cat = load_components_yaml(args.input)

    if args.categories:
        candidates_by_cat = {
            k: v for k, v in candidates_by_cat.items() if k in args.categories
        }

    if not candidates_by_cat:
        logger.error("No categories found in input")
        sys.exit(1)

    t5_model = None if args.clip_only else args.t5_model

    print(f"{'='*60}")
    print("DIVERSITY SELECTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input:       {args.input}")
    print(f"  Categories:  {len(candidates_by_cat)}")
    print(f"  Target k:    {args.k} {'(auto-k enabled)' if args.auto_k else ''}")
    print(f"  Alpha:       {args.alpha}")
    print(f"  CLIP model:  {args.clip_model}")
    print(f"  T5 model:    {t5_model or '(disabled)'}")
    total_candidates = sum(len(v) for v in candidates_by_cat.values())
    print(f"  Total candidates: {total_candidates}")

    # Initialize embedder
    embedder = DualSpaceEmbedder(
        clip_model=args.clip_model,
        t5_model=t5_model,
    )

    # Pre-compute embeddings for all categories (needed for cross-category contamination)
    logger.info("\nPre-computing category embeddings for contamination check...")
    all_cat_embeddings: dict[str, DualEmbeddings] = {}
    for cat, words in candidates_by_cat.items():
        all_cat_embeddings[cat] = embedder.embed(words)

    # Run selection pipeline per category
    results: dict[str, SelectionResult] = {}
    for cat, candidates in sorted(candidates_by_cat.items()):
        results[cat] = select_category(
            category=cat,
            candidates=candidates,
            embedder=embedder,
            all_category_embeddings={
                k: v for k, v in all_cat_embeddings.items() if k != cat
            },
            k=args.k,
            alpha=args.alpha,
            redundancy_threshold=args.redundancy_threshold,
            auto_k=args.auto_k,
        )

    # Summary
    print(f"\n{'='*60}")
    print("SELECTION SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Category':<25} {'Cand':>5} {'Filt':>5} {'Sel':>4} "
        f"{'Elbow':>5} {'Before':>7} {'After':>7} {'Improv':>7}"
    )
    print("-" * 72)

    for cat, r in sorted(results.items()):
        improvement = r.before_stats.mean_similarity - r.after_stats.mean_similarity
        print(
            f"{cat:<25} {r.candidates_count:>5} {r.filtered_count:>5} "
            f"{r.selected_count:>4} {r.elbow_k or '-':>5} "
            f"{r.before_stats.mean_similarity:>7.3f} "
            f"{r.after_stats.mean_similarity:>7.3f} "
            f"{improvement:>+7.3f}"
        )

    total_selected = sum(r.selected_count for r in results.values())
    print(f"\nTotal selected: {total_selected} from {total_candidates} candidates")

    # Output
    if args.dry_run:
        print(f"\n[DRY RUN] Would write components and analysis files")
        return

    data_dir = Path(__file__).parent.parent / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write components YAML
    components_path = args.output or data_dir / f"selected_components_{ts}.yaml"
    components_data = build_components_yaml(results)
    with open(components_path, "w") as f:
        yaml.dump(components_data, f, default_flow_style=False, allow_unicode=True, width=120)
    print(f"\nComponents saved: {components_path}")

    # Write analysis YAML
    analysis_path = components_path.with_name(
        components_path.stem + "_analysis" + components_path.suffix
    )
    analysis_data = build_analysis_yaml(results, args.alpha)
    with open(analysis_path, "w") as f:
        yaml.dump(analysis_data, f, default_flow_style=False, allow_unicode=True, width=120)
    print(f"Analysis saved:   {analysis_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
