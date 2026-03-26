#!/usr/bin/env python3
"""
Diversity Selection Pipeline
═══════════════════════════════════════════════════════════════════════════════

Clean linear pipeline — no iterative loops, no centroid drift.

Steps:
    1. Embed every unique word once (CLIP + T5 cache)
    2. Compute category centroids from original pools (fixed reference)
    3. One-pass contamination: items closer to another centroid get moved there
    4. Per-category selection: redundancy filter → farthest-point sampling
    5. Opposite pairing for negatable categories
    6. Output components YAML

Usage:
    # Basic — all categories, k=60
    uv run python -m promptweaver.tools.select \
        -i data/generated_candidates_20260325_193614.yaml --k 60

    # With per-category alpha
    uv run python -m promptweaver.tools.select \
        -i data/generated_candidates.yaml --k 60 \
        --alpha-per-category data/alpha_per_category.json

    # Write intermediate reallocated pools (for inspection / alpha analysis)
    uv run python -m promptweaver.tools.select \
        -i data/generated_candidates.yaml --k 60 \
        --save-reallocated data/reallocated.yaml
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
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """Compute once, lookup forever. No model calls after init."""

    def __init__(self, embedder: DualSpaceEmbedder, all_words: list[str]) -> None:
        unique = list(dict.fromkeys(all_words))
        logger.info(f"Computing embeddings for {len(unique)} unique words (one-shot)...")

        clip_emb = embedder.clip_embedder.encode_batch(unique)
        t5_emb: np.ndarray | None = None
        if embedder.t5_embedder:
            t5_emb = embedder.t5_embedder.encode_batch(unique)

        self._clip = {w: clip_emb[i] for i, w in enumerate(unique)}
        self._t5 = {w: t5_emb[i] for i, w in enumerate(unique)} if t5_emb is not None else None
        self.has_t5 = t5_emb is not None

        logger.info(f"Cache ready: {len(unique)} words, CLIP={clip_emb.shape[1]}d"
                     + (f", T5={t5_emb.shape[1]}d" if t5_emb is not None else ""))

    def get(self, words: list[str]) -> DualEmbeddings:
        """Look up cached embeddings for a word list."""
        clip = np.array([self._clip[w] for w in words])
        t5 = np.array([self._t5[w] for w in words]) if self._t5 else None
        return DualEmbeddings(words=words, clip=clip, t5=t5)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: CONTAMINATION + REALLOCATION (one pass)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_centroids(
    categories: dict[str, list[str]],
    cache: EmbeddingCache,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Normalized centroids per category: (clip_centroid, t5_centroid | None)."""
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for cat, words in categories.items():
        if not words:
            continue
        emb = cache.get(words)
        clip_c = emb.clip.mean(axis=0)
        clip_c /= np.linalg.norm(clip_c)
        t5_c = None
        if emb.t5 is not None:
            t5_c = emb.t5.mean(axis=0)
            t5_c /= np.linalg.norm(t5_c)
        centroids[cat] = (clip_c, t5_c)
    return centroids


def reallocate_contaminated(
    categories: dict[str, list[str]],
    cache: EmbeddingCache,
    alpha: float | dict[str, float],
    min_delta: float = 0.01,
) -> tuple[dict[str, list[str]], int]:
    """
    Single-pass contamination check + reallocation.

    Computes centroids from the ORIGINAL pools. For each word in each category,
    checks if it's closer to another category's centroid. If so, moves it there.
    Centroids are never recomputed — one pass, no drift.

    Returns (reallocated_categories, num_moved).
    """
    centroids = compute_centroids(categories, cache)
    result = {cat: list(words) for cat, words in categories.items()}
    moves: list[tuple[str, str, str, float]] = []  # (word, from, to, delta)

    for cat, words in categories.items():
        cat_alpha = alpha if isinstance(alpha, float) else alpha.get(cat, 0.5)
        emb = cache.get(words)

        for i, word in enumerate(words):
            sims: dict[str, float] = {}
            for other_cat, (clip_c, t5_c) in centroids.items():
                clip_sim = float(np.dot(emb.clip[i], clip_c))
                if t5_c is not None and emb.t5 is not None:
                    t5_sim = float(np.dot(emb.t5[i], t5_c))
                    sims[other_cat] = cat_alpha * clip_sim + (1 - cat_alpha) * t5_sim
                else:
                    sims[other_cat] = clip_sim

            assigned_sim = sims.get(cat, 0)
            closest = max(sims, key=lambda c: sims[c])
            delta = sims[closest] - assigned_sim

            if closest != cat and delta > min_delta:
                moves.append((word, cat, closest, delta))

    # Apply moves
    moved = 0
    for word, from_cat, to_cat, delta in moves:
        if word in result[from_cat] and word not in result.get(to_cat, []):
            result[from_cat].remove(word)
            result.setdefault(to_cat, []).append(word)
            moved += 1

    if moved > 0:
        logger.info(f"  Reallocation: moved {moved} items across categories")
        # Show top moves
        top = sorted(moves, key=lambda x: x[3], reverse=True)[:5]
        for word, frm, to, delta in top:
            logger.info(f"    '{word}': {frm} → {to} (delta={delta:+.3f})")
        if len(moves) > 5:
            logger.info(f"    ... and {len(moves) - 5} more")

    return result, moved


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PER-CATEGORY SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    category: str
    candidates_count: int
    filtered_count: int
    selected_count: int
    selected_words: list[str]
    opposites: dict[str, str]
    before_stats: DiversityStats
    after_stats: DiversityStats
    elbow_k: int | None
    redundant_dropped: int


def compute_elbow(sim_matrix: np.ndarray, max_k: int | None = None) -> int:
    """Find the elbow in the farthest-point distance curve."""
    n = sim_matrix.shape[0]
    if n < 4:
        return n
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
        return len(distances) + 2

    diffs = [distances[i] - distances[i + 1] for i in range(len(distances) - 1)]
    return int(np.argmax(diffs)) + 3


def select_category(
    category: str,
    words: list[str],
    cache: EmbeddingCache,
    k: int,
    alpha: float,
    redundancy_threshold: float,
) -> SelectionResult:
    """Redundancy filter → farthest-point select → opposites. No contamination here."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"{category} ({len(words)} candidates, alpha={alpha:.3f})")
    logger.info(f"{'─'*60}")

    if not words:
        logger.warning(f"  Empty category!")
        empty_stats = DiversityStats(0, 0, 0, 0, 0, ("", "", 0), ("", "", 0), 0, redundancy_threshold)
        return SelectionResult(category, 0, 0, 0, [], {}, empty_stats, empty_stats, None, 0)

    # Before stats
    emb = cache.get(words)
    sim = emb.joint_similarity_matrix(alpha)
    before_stats = analyze_diversity(sim, words, redundancy_threshold)

    # Redundancy filter
    n = len(words)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if sim[i, j] > redundancy_threshold:
                drop.add(j)

    kept = [w for i, w in enumerate(words) if i not in drop]
    num_dropped = len(drop)
    if num_dropped:
        logger.info(f"  Redundancy filter (>{redundancy_threshold}): dropped {num_dropped}")

    logger.info(f"  After filter: {len(kept)} candidates")

    # Farthest-point selection
    emb_kept = cache.get(kept)
    sim_kept = emb_kept.joint_similarity_matrix(alpha)

    elbow_k = compute_elbow(sim_kept, max_k=min(k + 20, len(kept)))
    effective_k = min(k, len(kept))
    logger.info(f"  Elbow at k={elbow_k} (using k={effective_k})")

    selected_idx = farthest_point_sampling(sim_kept, effective_k)
    selected_words = [kept[i] for i in selected_idx]

    # After stats
    emb_sel = cache.get(selected_words)
    sim_sel = emb_sel.joint_similarity_matrix(alpha)
    after_stats = analyze_diversity(sim_sel, selected_words, redundancy_threshold)

    improvement = before_stats.mean_similarity - after_stats.mean_similarity
    logger.info(
        f"  Diversity: {before_stats.mean_similarity:.3f} → {after_stats.mean_similarity:.3f} "
        f"({improvement:+.3f})"
    )
    logger.info(
        f"  Closest pair: '{after_stats.most_similar_pair[0]}' <-> "
        f"'{after_stats.most_similar_pair[1]}' ({after_stats.most_similar_pair[2]:.3f})"
    )

    # Opposites
    opposites: dict[str, str] = {}
    if category in NEGATABLE_CATEGORIES:
        opposites = greedy_opposite_pairs(sim_sel, selected_words)
        logger.info(f"  Computed {len(opposites)} opposite pairs")

    return SelectionResult(
        category=category,
        candidates_count=len(words),
        filtered_count=len(kept),
        selected_count=len(selected_words),
        selected_words=selected_words,
        opposites=opposites,
        before_stats=before_stats,
        after_stats=after_stats,
        elbow_k=elbow_k,
        redundant_dropped=num_dropped,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def build_components_yaml(results: dict[str, SelectionResult]) -> dict[str, Any]:
    components: dict[str, list[dict[str, Any]]] = {}
    for cat, r in sorted(results.items()):
        items: list[dict[str, Any]] = []
        for word in r.selected_words:
            entry: dict[str, Any] = {"word": word}
            if word in r.opposites:
                entry["opposite"] = r.opposites[word]
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
        "categories": {},
    }
    for cat, r in sorted(results.items()):
        analysis["categories"][cat] = {
            "candidates": r.candidates_count,
            "after_filter": r.filtered_count,
            "selected": r.selected_count,
            "elbow_k": r.elbow_k,
            "alpha": per_cat_alpha.get(cat, default_alpha),
            "redundant_dropped": r.redundant_dropped,
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
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha-per-category", type=Path,
                        help="JSON: category → alpha")
    parser.add_argument("--redundancy-threshold", type=float, default=0.88)
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5-model", default="google/flan-t5-large")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--save-reallocated", type=Path,
                        help="Write reallocated pools to YAML before selection")
    parser.add_argument("--skip-contamination", action="store_true",
                        help="Skip contamination check + reallocation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    all_candidates = load_components_yaml(args.input)
    if not all_candidates:
        logger.error("No categories found")
        sys.exit(1)

    t5_model = None if args.clip_only else args.t5_model

    per_cat_alpha: dict[str, float] = {}
    if args.alpha_per_category and args.alpha_per_category.exists():
        per_cat_alpha = json.loads(args.alpha_per_category.read_text())

    total = sum(len(v) for v in all_candidates.values())
    print(f"{'='*60}")
    print("DIVERSITY SELECTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Categories: {len(all_candidates)} ({total} total words)")
    print(f"  Target k:   {args.k}")
    print(f"  Alpha:      {'per-category' if per_cat_alpha else args.alpha}")
    print(f"  Models:     CLIP={args.clip_model}, T5={t5_model or 'disabled'}")

    # ── EMBED ONCE ──────────────────────────────────────────────────────
    embedder = DualSpaceEmbedder(clip_model=args.clip_model, t5_model=t5_model)
    all_words = [w for words in all_candidates.values() for w in words]
    cache = EmbeddingCache(embedder, all_words)

    # ── CONTAMINATION + REALLOCATION (one pass) ────────────────────────
    if args.skip_contamination:
        working = {cat: list(words) for cat, words in all_candidates.items()}
        logger.info("Skipping contamination check")
    else:
        alpha_for_contam = per_cat_alpha if per_cat_alpha else args.alpha
        working, num_moved = reallocate_contaminated(
            all_candidates, cache, alpha_for_contam,
        )

    # Optionally save reallocated pools
    if args.save_reallocated:
        realloc_data = {"components": {cat: sorted(words) for cat, words in working.items()}}
        with open(args.save_reallocated, "w") as f:
            yaml.dump(realloc_data, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nReallocated pools saved: {args.save_reallocated}")

    # ── PER-CATEGORY SELECTION ──────────────────────────────────────────
    select_cats = set(working.keys())
    if args.categories:
        select_cats = {c for c in args.categories if c in working}

    results: dict[str, SelectionResult] = {}
    for cat in sorted(select_cats):
        cat_alpha = per_cat_alpha.get(cat, args.alpha)
        results[cat] = select_category(
            category=cat,
            words=working[cat],
            cache=cache,
            k=args.k,
            alpha=cat_alpha,
            redundancy_threshold=args.redundancy_threshold,
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
            f"{r.after_stats.mean_similarity:>7.3f} {imp:>+7.3f}"
        )

    total_sel = sum(r.selected_count for r in results.values())
    sel_cands = sum(r.candidates_count for r in results.values())
    print(f"\nTotal selected: {total_sel} from {sel_cands} candidates")

    if args.dry_run:
        print("\n[DRY RUN]")
        return

    # ── WRITE OUTPUT ────────────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_path = args.output or data_dir / f"selected_components_{ts}.yaml"
    with open(comp_path, "w") as f:
        yaml.dump(build_components_yaml(results), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"\nComponents: {comp_path}")

    analysis_path = comp_path.with_name(comp_path.stem + "_analysis" + comp_path.suffix)
    with open(analysis_path, "w") as f:
        yaml.dump(build_analysis_yaml(results, per_cat_alpha, args.alpha), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"Analysis:   {analysis_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
