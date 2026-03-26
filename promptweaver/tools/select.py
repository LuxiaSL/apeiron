#!/usr/bin/env python3
"""
Diversity Selection Pipeline
═══════════════════════════════════════════════════════════════════════════════

Takes generated candidates, computes dual CLIP+T5 embeddings ONCE into a
global cache, then performs all filtering/reallocation/selection via pure
array lookups — no re-embedding.

Pipeline:
    1. Load generated candidates YAML
    2. Compute dual-space embeddings for every unique word (one-shot)
    3. Iterative cross-category reallocation (contaminated items move to
       the category they're closest to, loop until stable)
    4. Redundancy pre-filter per category
    5. Farthest-point sampling per category in joint space
    6. Recompute semantic opposites for negatable categories
    7. Output final components YAML

Usage:
    # Full pipeline
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates_20260325.yaml --k 60

    # With per-category alpha from analyze.py --alpha-analysis
    uv run python -m promptweaver.tools.select \
        --input data/generated_candidates.yaml --k 60 \
        --alpha-per-category data/alpha_per_category.json
"""

from __future__ import annotations

import argparse
import json
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
    farthest_point_sampling,
    greedy_opposite_pairs,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

NEGATABLE_CATEGORIES = frozenset({
    "color_logic", "light_behavior", "atmosphere_field",
    "temporal_state", "texture_density", "medium_render",
})


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE — compute once, lookup forever
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """
    Pre-computes and stores embeddings for every unique word. All subsequent
    operations (filtering, reallocation, selection) use pure array lookups.
    """

    def __init__(self, embedder: DualSpaceEmbedder, all_words: list[str]) -> None:
        unique = list(dict.fromkeys(all_words))  # Dedupe preserving order
        logger.info(f"Computing embeddings for {len(unique)} unique words (one-shot)...")

        clip_emb = embedder.clip_embedder.encode_batch(unique)
        t5_emb: np.ndarray | None = None
        if embedder.t5_embedder:
            t5_emb = embedder.t5_embedder.encode_batch(unique)

        self._clip: dict[str, np.ndarray] = {
            w: clip_emb[i] for i, w in enumerate(unique)
        }
        self._t5: dict[str, np.ndarray] | None = None
        if t5_emb is not None:
            self._t5 = {w: t5_emb[i] for i, w in enumerate(unique)}

        self.has_t5 = t5_emb is not None
        logger.info(f"Cache ready: {len(unique)} words, CLIP={clip_emb.shape[1]}d"
                     + (f", T5={t5_emb.shape[1]}d" if t5_emb is not None else ""))

    def get_dual(self, words: list[str]) -> DualEmbeddings:
        """Look up pre-computed embeddings for a word list. O(n) array construction, no model calls."""
        clip = np.array([self._clip[w] for w in words])
        t5 = np.array([self._t5[w] for w in words]) if self._t5 else None
        return DualEmbeddings(words=words, clip=clip, t5=t5)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    """Result of the selection pipeline for one category."""

    category: str
    candidates_count: int
    filtered_count: int
    selected_count: int
    selected_words: list[str]
    opposites: dict[str, str]
    before_stats: DiversityStats
    after_stats: DiversityStats
    elbow_k: int | None
    contaminated_dropped: list[str]
    redundant_dropped: list[str]


def compute_centroids(
    categories: dict[str, list[str]],
    cache: EmbeddingCache,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Compute normalized centroids per category from cache. Returns (clip_centroid, t5_centroid)."""
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for cat, words in categories.items():
        if not words:
            continue
        emb = cache.get_dual(words)
        clip_c = emb.clip.mean(axis=0)
        clip_c /= np.linalg.norm(clip_c)
        t5_c = None
        if emb.t5 is not None:
            t5_c = emb.t5.mean(axis=0)
            t5_c /= np.linalg.norm(t5_c)
        centroids[cat] = (clip_c, t5_c)
    return centroids


def find_contaminated(
    category: str,
    words: list[str],
    cache: EmbeddingCache,
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]],
    alpha: float,
    min_delta: float = 0.01,
) -> list[dict[str, Any]]:
    """Find words closer to another category's centroid than their own."""
    emb = cache.get_dual(words)
    contaminated: list[dict[str, Any]] = []

    for i, word in enumerate(words):
        sims: dict[str, float] = {}
        for other_cat, (clip_c, t5_c) in centroids.items():
            clip_sim = float(np.dot(emb.clip[i], clip_c))
            if t5_c is not None and emb.t5 is not None:
                t5_sim = float(np.dot(emb.t5[i], t5_c))
                sims[other_cat] = alpha * clip_sim + (1 - alpha) * t5_sim
            else:
                sims[other_cat] = clip_sim

        assigned_sim = sims.get(category, 0)
        closest = max(sims, key=lambda c: sims[c])

        if closest != category and (sims[closest] - assigned_sim) > min_delta:
            contaminated.append({
                "word": word,
                "assigned": category,
                "closest": closest,
                "delta": sims[closest] - assigned_sim,
            })

    return contaminated


def filter_redundant(
    words: list[str],
    sim_matrix: np.ndarray,
    threshold: float = 0.88,
) -> tuple[list[str], list[str]]:
    """Drop words too similar to an earlier word in the list."""
    n = len(words)
    drop: set[int] = set()

    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if sim_matrix[i, j] > threshold:
                drop.add(j)

    kept = [w for i, w in enumerate(words) if i not in drop]
    dropped = [w for i, w in enumerate(words) if i in drop]
    if dropped:
        logger.info(f"  Redundancy filter (>{threshold}): dropped {len(dropped)} items")
    return kept, dropped


def compute_elbow(
    sim_matrix: np.ndarray,
    max_k: int | None = None,
) -> tuple[int, list[float]]:
    """Track diversity contribution per item added. Returns (elbow_k, distances)."""
    n = sim_matrix.shape[0]
    max_k = max_k or n

    masked = sim_matrix.copy()
    np.fill_diagonal(masked, np.inf)
    min_idx = np.unravel_index(np.argmin(masked), masked.shape)
    selected = [min_idx[0], min_idx[1]]
    selected_set = set(selected)

    min_sim = np.minimum(sim_matrix[:, selected[0]], sim_matrix[:, selected[1]])
    distances: list[float] = []

    while len(selected) < min(max_k, n):
        sim_masked = min_sim.copy()
        for idx in selected_set:
            sim_masked[idx] = np.inf

        best = int(np.argmin(sim_masked))
        if best in selected_set:
            break

        distances.append(float(1.0 - sim_masked[best]))
        selected.append(best)
        selected_set.add(best)
        min_sim = np.minimum(min_sim, sim_matrix[:, best])

    if len(distances) < 3:
        return len(distances) + 2, distances

    diffs = [distances[i] - distances[i + 1] for i in range(len(distances) - 1)]
    elbow_idx = int(np.argmax(diffs))
    return elbow_idx + 3, distances


def select_category(
    category: str,
    candidates: list[str],
    cache: EmbeddingCache,
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]],
    k: int,
    alpha: float,
    redundancy_threshold: float,
    auto_k: bool = False,
) -> SelectionResult:
    """Run the full selection pipeline for one category using cached embeddings."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Category: {category} ({len(candidates)} candidates, alpha={alpha:.3f})")
    logger.info(f"{'─'*60}")

    # Before stats
    emb_before = cache.get_dual(candidates)
    sim_before = emb_before.joint_similarity_matrix(alpha)
    before_stats = analyze_diversity(sim_before, candidates)

    # Stage 1: Contamination filter
    contaminated = find_contaminated(category, candidates, cache, centroids, alpha)
    drop_words = {c["word"] for c in contaminated}
    kept = [w for w in candidates if w not in drop_words]
    contaminated_dropped = [w for w in candidates if w in drop_words]

    if contaminated_dropped:
        logger.info(f"  Contamination filter: dropped {len(contaminated_dropped)} items")
        for item in sorted(contaminated, key=lambda x: x["delta"], reverse=True)[:5]:
            logger.info(f"    '{item['word']}' → closer to {item['closest']} (delta={item['delta']:+.3f})")
        if len(contaminated_dropped) > 5:
            logger.info(f"    ... and {len(contaminated_dropped) - 5} more")

    # Stage 2: Redundancy filter
    emb_kept = cache.get_dual(kept)
    sim_kept = emb_kept.joint_similarity_matrix(alpha)
    kept, redundant_dropped = filter_redundant(kept, sim_kept, redundancy_threshold)

    filtered_count = len(kept)
    logger.info(f"  After filters: {filtered_count} candidates remain")

    if filtered_count == 0:
        logger.warning(f"  No candidates remain after filtering!")
        return SelectionResult(
            category=category, candidates_count=len(candidates),
            filtered_count=0, selected_count=0, selected_words=[],
            opposites={}, before_stats=before_stats,
            after_stats=before_stats, elbow_k=None,
            contaminated_dropped=contaminated_dropped,
            redundant_dropped=redundant_dropped,
        )

    # Stage 3: Elbow + selection
    emb_final = cache.get_dual(kept)
    sim_final = emb_final.joint_similarity_matrix(alpha)
    elbow_k, _ = compute_elbow(sim_final, max_k=min(k + 20, filtered_count))

    if auto_k:
        effective_k = min(elbow_k, filtered_count)
        logger.info(f"  Elbow detected at k={elbow_k}, using auto-k={effective_k}")
    else:
        effective_k = min(k, filtered_count)
        logger.info(f"  Elbow at k={elbow_k} (using requested k={effective_k})")

    selected_indices = farthest_point_sampling(sim_final, effective_k)
    selected_words = [kept[i] for i in selected_indices]

    # After stats
    emb_sel = cache.get_dual(selected_words)
    sim_after = emb_sel.joint_similarity_matrix(alpha)
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

    # Stage 4: Opposites
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


def build_components_yaml(results: dict[str, SelectionResult]) -> dict[str, Any]:
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
    per_cat_alpha: dict[str, float],
    default_alpha: float,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "alphas": per_cat_alpha or {"default": default_alpha},
        "categories": {},
    }
    for cat, r in sorted(results.items()):
        analysis["categories"][cat] = {
            "candidates": r.candidates_count,
            "after_filters": r.filtered_count,
            "selected": r.selected_count,
            "elbow_k": r.elbow_k,
            "alpha_used": per_cat_alpha.get(cat, default_alpha),
            "contaminated_dropped": len(r.contaminated_dropped),
            "redundant_dropped": len(r.redundant_dropped),
            "before_mean_sim": round(r.before_stats.mean_similarity, 4),
            "after_mean_sim": round(r.after_stats.mean_similarity, 4),
            "improvement": round(
                r.before_stats.mean_similarity - r.after_stats.mean_similarity, 4
            ),
            "most_similar_pair": {
                "words": list(r.after_stats.most_similar_pair[:2]),
                "similarity": round(r.after_stats.most_similar_pair[2], 4),
            },
        }
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select most diverse components from generated candidates",
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Generated candidates YAML")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output components YAML")
    parser.add_argument("--k", type=int, default=60,
                        help="Target components per category (default: 60)")
    parser.add_argument("--auto-k", action="store_true",
                        help="Use elbow detection for k")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Default CLIP vs T5 weight (default: 0.5)")
    parser.add_argument("--alpha-per-category", type=Path,
                        help="JSON file: category → alpha")
    parser.add_argument("--redundancy-threshold", type=float, default=0.88)
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5-model", default="google/flan-t5-large")
    parser.add_argument("--categories", nargs="+",
                        help="Process only these categories")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Load ALL categories
    all_candidates = load_components_yaml(args.input)
    if not all_candidates:
        logger.error("No categories found")
        sys.exit(1)

    select_cats = set(all_candidates.keys())
    if args.categories:
        select_cats = {c for c in args.categories if c in all_candidates}
        if not select_cats:
            logger.error(f"No matching categories: {args.categories}")
            sys.exit(1)

    t5_model = None if args.clip_only else args.t5_model

    per_cat_alpha: dict[str, float] = {}
    if args.alpha_per_category and args.alpha_per_category.exists():
        per_cat_alpha = json.loads(args.alpha_per_category.read_text())
        logger.info(f"Loaded per-category alphas from {args.alpha_per_category}")

    total_cands = sum(len(v) for c, v in all_candidates.items() if c in select_cats)

    print(f"{'='*60}")
    print("DIVERSITY SELECTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input:         {args.input}")
    print(f"  All categories: {len(all_candidates)}")
    print(f"  Selecting:     {len(select_cats)} — {sorted(select_cats)}")
    print(f"  Target k:      {args.k} {'(auto-k)' if args.auto_k else ''}")
    print(f"  Alpha:         {'per-category' if per_cat_alpha else args.alpha}")
    print(f"  CLIP:          {args.clip_model}")
    print(f"  T5:            {t5_model or '(disabled)'}")
    print(f"  Candidates:    {total_cands}")

    # ── ONE-SHOT EMBEDDING ──────────────────────────────────────────────
    embedder = DualSpaceEmbedder(clip_model=args.clip_model, t5_model=t5_model)

    # Collect every unique word across all categories
    all_words: list[str] = []
    for words in all_candidates.values():
        all_words.extend(words)

    cache = EmbeddingCache(embedder, all_words)

    # ── SINGLE-PASS REALLOCATION ───────────────────────────────────────
    # Compute centroids from ORIGINAL pools (fixed reference — no drift).
    # Items closer to another centroid get moved there in one pass.
    working = {cat: list(words) for cat, words in all_candidates.items()}

    fixed_centroids = compute_centroids(all_candidates, cache)
    moved_total = 0

    for cat in sorted(working.keys()):
        cat_alpha = per_cat_alpha.get(cat, args.alpha)
        contaminated = find_contaminated(
            cat, working[cat], cache, fixed_centroids, cat_alpha,
        )
        for item in contaminated:
            target = item["closest"]
            word = item["word"]
            if (word in working[cat]
                    and word not in working.get(target, [])):
                working[cat].remove(word)
                working.setdefault(target, []).append(word)
                moved_total += 1

    if moved_total > 0:
        logger.info(f"  Reallocation: moved {moved_total} items (single pass, fixed centroids)")

    # ── PER-CATEGORY SELECTION ──────────────────────────────────────────
    # Use original fixed centroids for contamination during selection.
    # Post-reallocation centroids are too distorted by absorbed foreign words.
    results: dict[str, SelectionResult] = {}
    for cat in sorted(select_cats):
        cat_alpha = per_cat_alpha.get(cat, args.alpha)
        results[cat] = select_category(
            category=cat,
            candidates=working[cat],
            cache=cache,
            centroids={k: v for k, v in fixed_centroids.items() if k != cat},
            k=args.k,
            alpha=cat_alpha,
            redundancy_threshold=args.redundancy_threshold,
            auto_k=args.auto_k,
        )

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELECTION SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Category':<25} {'Cand':>5} {'Filt':>5} {'Sel':>4} "
        f"{'Elbow':>5} {'Alpha':>5} {'Before':>7} {'After':>7} {'Improv':>7}"
    )
    print("-" * 80)

    for cat, r in sorted(results.items()):
        imp = r.before_stats.mean_similarity - r.after_stats.mean_similarity
        a = per_cat_alpha.get(cat, args.alpha)
        print(
            f"{cat:<25} {r.candidates_count:>5} {r.filtered_count:>5} "
            f"{r.selected_count:>4} {r.elbow_k or '-':>5} {a:>5.3f} "
            f"{r.before_stats.mean_similarity:>7.3f} "
            f"{r.after_stats.mean_similarity:>7.3f} "
            f"{imp:>+7.3f}"
        )

    total_sel = sum(r.selected_count for r in results.values())
    print(f"\nTotal selected: {total_sel} from {total_cands} candidates")

    if args.dry_run:
        print("\n[DRY RUN] Would write components and analysis files")
        return

    data_dir = Path(__file__).parent.parent / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_path = args.output or data_dir / f"selected_components_{ts}.yaml"
    with open(comp_path, "w") as f:
        yaml.dump(build_components_yaml(results), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"\nComponents saved: {comp_path}")

    analysis_path = comp_path.with_name(comp_path.stem + "_analysis" + comp_path.suffix)
    with open(analysis_path, "w") as f:
        yaml.dump(build_analysis_yaml(results, per_cat_alpha, args.alpha), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"Analysis saved:   {analysis_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
