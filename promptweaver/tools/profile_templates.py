#!/usr/bin/env python3
"""
Template Diversity Profiling
═══════════════════════════════════════════════════════════════════════════════

Profiles each template's diversity potential using dual-space embeddings:

1. Internal diversity — how varied are prompts from a single template?
2. Cross-template uniqueness — how distinct is each template from the others?
3. Category usage analysis — which templates leverage which categories?
4. Pairwise similarity matrix — which templates overlap in output space?

Uses the persistent OpenCLIP + T5-XXL embedding cache when available,
falling back to single-model encoding otherwise.

Prior art: dream_gen's profile_templates.py (StableComponentSelector,
           TemplateProfile, pairwise matrix, clustering).

Usage:
    # Profile with existing embedding cache (no model load if prompts fit)
    uv run python -m promptweaver.tools.profile_templates \
        --components data/curated_gated.yaml

    # Profile with explicit model (slower, builds cache)
    uv run python -m promptweaver.tools.profile_templates \
        --components data/curated_gated.yaml \
        --clip-model openclip --t5-model google/t5-v1_1-xxl

    # Compare two pools
    uv run python -m promptweaver.tools.profile_templates \
        --components data/curated_gated.yaml \
        --baseline-components data/components.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualSpaceEmbedder,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TEMPLATES = DATA_DIR / "templates.yaml"
DEFAULT_COMPONENTS = DATA_DIR / "curated_gated.yaml"
ENCODE_BATCH = 256


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TemplateProfile:
    """Diversity statistics for a single template."""

    name: str
    structure: str
    n_samples: int

    # Internal diversity (within template)
    internal_mean: float = 0.0
    internal_min: float = 0.0
    internal_max: float = 0.0
    internal_range: float = 0.0

    # Cross-template uniqueness (vs all other templates)
    cross_mean: float = 0.0
    cross_min: float = 0.0
    cross_max: float = 0.0

    # Structural info
    n_slots: int = 0
    slot_categories: list[str] = field(default_factory=list)
    multi_slots: int = 0  # slots using same category twice
    theoretical_combinations: int = 0

    def internal_score(self) -> float:
        """Higher = more internally diverse."""
        return (1 - self.internal_mean) * 0.6 + self.internal_range * 0.4

    def uniqueness_score(self) -> float:
        """Higher = more distinct from other templates."""
        return 1 - self.cross_mean

    def combined_score(self) -> float:
        return self.internal_score() * 0.5 + self.uniqueness_score() * 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def parse_slots(structure: str) -> list[tuple[str, int, str]]:
    """Parse {category}, {category:N}, {category:N:sep}."""
    return [
        (m.group(1), int(m.group(2)) if m.group(2) else 1, m.group(3) or " ")
        for m in re.finditer(r"\{(\w+)(?::(\d+))?(?::([^}]*))?\}", structure)
    ]


def generate_prompts(
    template: dict[str, Any],
    components: dict[str, list[str]],
    n: int,
    seed: int = 42,
) -> list[str]:
    """Generate n prompts from a template with random component selection."""
    rng = random.Random(seed)
    structure = template["structure"]
    slots = parse_slots(structure)

    prompts: list[str] = []
    for _ in range(n):
        prompt = structure
        used: dict[str, set[str]] = defaultdict(set)

        for category, count, sep in slots:
            if category not in components:
                continue
            pool = components[category]
            available = [w for w in pool if w not in used[category]]
            if not available:
                available = pool
            sample_size = min(count, len(available))
            selected = rng.sample(available, sample_size)
            used[category].update(selected)

            replacement = sep.join(selected)
            if count > 1:
                pat = f"{{{category}:{count}:{sep}}}" if sep != " " else f"{{{category}:{count}}}"
            else:
                pat = f"{{{category}}}"
            prompt = prompt.replace(pat, replacement, 1)

        prompts.append(prompt)
    return prompts


def compute_theoretical_combinations(
    slots: list[tuple[str, int, str]],
    components: dict[str, list[str]],
) -> int:
    """Total possible unique prompts for a template."""
    # Group slots by category to handle multi-use (e.g., {material_substance} twice)
    cat_needs: dict[str, int] = {}
    for cat, count, _ in slots:
        cat_needs[cat] = cat_needs.get(cat, 0) + count

    total = 1
    for cat, needed in cat_needs.items():
        n = len(components.get(cat, []))
        perm = 1
        for i in range(needed):
            perm *= max(1, n - i)
        total *= perm
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def encode_prompts(
    prompts: list[str],
    embedder: DualSpaceEmbedder,
    space: str = "clip",
) -> np.ndarray:
    """Encode prompts into embeddings. Uses CLIP space for prompt-level diversity."""
    # For full prompts, we use the CLIP embedder (visual space)
    # T5 is more useful for individual words
    if space == "clip":
        return embedder.clip_embedder.encode_batch(prompts)
    elif space == "t5" and embedder.t5_embedder:
        return embedder.t5_embedder.encode_batch(prompts)
    return embedder.clip_embedder.encode_batch(prompts)


def similarity_stats(emb: np.ndarray) -> dict[str, float]:
    """Pairwise cosine similarity statistics."""
    if len(emb) < 2:
        return {"mean": 0, "min": 0, "max": 0, "range": 0}
    sim = emb @ emb.T
    triu = np.triu_indices(len(emb), k=1)
    vals = sim[triu]
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
    }


def cross_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> dict[str, float]:
    """Cross-set similarity statistics."""
    sim = emb_a @ emb_b.T
    return {
        "mean": float(np.mean(sim)),
        "min": float(np.min(sim)),
        "max": float(np.max(sim)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def profile_all(
    templates: dict[str, dict[str, Any]],
    components: dict[str, list[str]],
    embedder: DualSpaceEmbedder,
    n_samples: int = 500,
    seed: int = 42,
    space: str = "clip",
) -> tuple[list[TemplateProfile], dict[str, list[str]], dict[str, np.ndarray]]:
    """Profile all templates.

    Returns (profiles, all_prompts, template_embeddings).
    """
    # Phase 1: Generate all prompts
    print(f"\n[1/3] Generating {n_samples} prompts per template...")
    all_prompts: dict[str, list[str]] = {}
    for name, template in templates.items():
        prompts = generate_prompts(template, components, n_samples, seed=seed)
        all_prompts[name] = prompts

    # Phase 2: Embed all prompts in one batch
    print(f"[2/3] Embedding {sum(len(p) for p in all_prompts.values())} prompts...")
    flat: list[str] = []
    indices: dict[str, tuple[int, int]] = {}
    idx = 0
    for name, prompts in all_prompts.items():
        indices[name] = (idx, idx + len(prompts))
        flat.extend(prompts)
        idx += len(prompts)

    all_emb = encode_prompts(flat, embedder, space=space)
    template_embs: dict[str, np.ndarray] = {
        name: all_emb[s:e] for name, (s, e) in indices.items()
    }

    # Phase 3: Compute statistics
    print("[3/3] Computing diversity statistics...")
    profiles: list[TemplateProfile] = []

    for name, template in templates.items():
        structure = template["structure"]
        slots = parse_slots(structure)
        emb = template_embs[name]

        internal = similarity_stats(emb)

        # Cross-template: this template vs all others combined
        others = [template_embs[n] for n in template_embs if n != name]
        if others:
            combined = np.vstack(others)
            cross = cross_similarity(emb, combined)
        else:
            cross = {"mean": 0, "min": 0, "max": 0}

        # Category usage
        cat_counts: dict[str, int] = {}
        for cat, count, _ in slots:
            cat_counts[cat] = cat_counts.get(cat, 0) + count

        profile = TemplateProfile(
            name=name,
            structure=structure,
            n_samples=n_samples,
            internal_mean=internal["mean"],
            internal_min=internal["min"],
            internal_max=internal["max"],
            internal_range=internal["range"],
            cross_mean=cross["mean"],
            cross_min=cross["min"],
            cross_max=cross["max"],
            n_slots=len(slots),
            slot_categories=[cat for cat, _, _ in slots],
            multi_slots=sum(1 for c in cat_counts.values() if c > 1),
            theoretical_combinations=compute_theoretical_combinations(slots, components),
        )
        profiles.append(profile)
        print(f"  {name}: internal={profile.internal_score():.3f}, unique={profile.uniqueness_score():.3f}")

    return profiles, all_prompts, template_embs


def pairwise_matrix(
    template_embs: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Pairwise similarity matrix between templates."""
    names = list(template_embs.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                matrix[i, j] = similarity_stats(template_embs[a])["mean"]
            else:
                matrix[i, j] = cross_similarity(template_embs[a], template_embs[b])["mean"]

    return names, matrix


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════


def print_profiles(
    profiles: list[TemplateProfile],
    baseline: list[TemplateProfile] | None = None,
) -> None:
    """Print template diversity rankings."""
    print(f"\n{'='*100}")
    print("TEMPLATE DIVERSITY PROFILES")
    print(f"{'='*100}")

    base_lookup = {p.name: p for p in baseline} if baseline else {}
    ranked = sorted(profiles, key=lambda p: p.combined_score(), reverse=True)

    if base_lookup:
        print(f"\n{'Rank':<5} {'Template':<22} {'Combined':>9} {'d':>6} {'Internal':>9} {'d':>6} "
              f"{'Unique':>9} {'d':>6} {'IntMean':>8} {'XMean':>8}")
        print("-" * 100)
    else:
        print(f"\n{'Rank':<5} {'Template':<22} {'Combined':>9} {'Internal':>9} {'Unique':>9} "
              f"{'IntMean':>8} {'IntRange':>9} {'XMean':>8} {'Combos':>12}")
        print("-" * 105)

    for rank, p in enumerate(ranked, 1):
        if base_lookup and p.name in base_lookup:
            b = base_lookup[p.name]
            dc = p.combined_score() - b.combined_score()
            di = p.internal_score() - b.internal_score()
            du = p.uniqueness_score() - b.uniqueness_score()
            print(f"{rank:<5} {p.name:<22} {p.combined_score():>9.3f} {dc:>+6.3f} "
                  f"{p.internal_score():>9.3f} {di:>+6.3f} {p.uniqueness_score():>9.3f} {du:>+6.3f} "
                  f"{p.internal_mean:>8.3f} {p.cross_mean:>8.3f}")
        else:
            combos = f"{p.theoretical_combinations:.2e}" if p.theoretical_combinations > 1e9 else str(p.theoretical_combinations)
            print(f"{rank:<5} {p.name:<22} {p.combined_score():>9.3f} {p.internal_score():>9.3f} "
                  f"{p.uniqueness_score():>9.3f} {p.internal_mean:>8.3f} {p.internal_range:>9.3f} "
                  f"{p.cross_mean:>8.3f} {combos:>12}")


def print_matrix(names: list[str], matrix: np.ndarray) -> None:
    """Print cross-template similarity matrix."""
    print(f"\n{'='*100}")
    print("CROSS-TEMPLATE SIMILARITY MATRIX")
    print(f"{'='*100}")
    print("(diagonal = internal similarity, off-diagonal = cross-template)")

    short = [n[:10] for n in names]
    print(f"\n{'':>14}", end="")
    for s in short:
        print(f" {s:>11}", end="")
    print()
    print("-" * (14 + 11 * len(names)))

    for i in range(len(names)):
        print(f"{short[i]:>14}", end="")
        for j in range(len(names)):
            if i == j:
                print(f"  [{matrix[i,j]:.3f}]", end="")
            else:
                print(f"   {matrix[i,j]:.3f} ", end="")
        print()

    # Most/least similar pairs
    print(f"\n{'-'*100}")
    min_sim, max_sim = float("inf"), float("-inf")
    min_pair, max_pair = None, None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j] < min_sim:
                min_sim = matrix[i, j]
                min_pair = (names[i], names[j])
            if matrix[i, j] > max_sim:
                max_sim = matrix[i, j]
                max_pair = (names[i], names[j])

    if min_pair:
        print(f"Most distinct: {min_pair[0]} <-> {min_pair[1]} (sim={min_sim:.3f})")
    if max_pair:
        print(f"Most similar:  {max_pair[0]} <-> {max_pair[1]} (sim={max_sim:.3f})")


def print_category_usage(
    templates: dict[str, dict[str, Any]],
    components: dict[str, list[str]],
) -> None:
    """Show which categories each template uses."""
    categories = sorted(components.keys())

    print(f"\n{'='*100}")
    print("CATEGORY USAGE BY TEMPLATE")
    print(f"{'='*100}")

    cat_short = [c[:6] for c in categories]
    print(f"{'Template':<22}", end="")
    for cs in cat_short:
        print(f" {cs:>7}", end="")
    print(f" {'Total':>6}")
    print("-" * (22 + 8 * len(categories) + 7))

    for name, template in templates.items():
        slots = parse_slots(template["structure"])
        slot_cats = [s[0] for s in slots]
        print(f"{name:<22}", end="")
        total = 0
        for cat in categories:
            n = slot_cats.count(cat)
            if n > 0:
                print(f" {n:>7}", end="")
                total += n
            else:
                print(f" {'':>7}", end="")
        print(f" {total:>6}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile template diversity potential")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--baseline-components", type=Path,
                        help="Compare against a different component pool")
    parser.add_argument("--samples", "-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model", default="openclip")
    parser.add_argument("--t5-model", default=None,
                        help="T5 model (default: None — use CLIP only for prompt-level)")
    parser.add_argument("--space", choices=["clip", "t5"], default="clip",
                        help="Embedding space for prompt diversity (default: clip)")
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--show-categories", action="store_true")
    parser.add_argument("--show-samples", type=int, metavar="N",
                        help="Show N sample prompts per template")
    parser.add_argument("--save-stats", type=Path,
                        help="Save profile stats to YAML")
    parser.add_argument("-o", "--output", type=Path,
                        help="Save full report to YAML")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    # Load templates
    if not args.templates.exists():
        logger.error(f"Templates not found: {args.templates}")
        sys.exit(1)
    with open(args.templates) as f:
        raw = yaml.safe_load(f)
    templates = {t["id"]: t for t in raw.get("templates", []) if "id" in t}

    # Load components
    if not args.components.exists():
        logger.error(f"Components not found: {args.components}")
        sys.exit(1)
    components = load_components_yaml(args.components)

    total = sum(len(v) for v in components.values())
    print(f"{'='*100}")
    print("TEMPLATE DIVERSITY PROFILER")
    print(f"{'='*100}")
    print(f"  Templates: {len(templates)}")
    print(f"  Components: {total} across {len(components)} categories")
    print(f"  Samples per template: {args.samples}")
    print(f"  Embedding space: {args.space}")

    # Category usage (no models needed)
    if args.show_categories:
        print_category_usage(templates, components)

    # Load embedder for prompt-level encoding
    # For prompts (full sentences), CLIP is more appropriate than T5
    # T5-XXL is too expensive for 6000+ prompt embeddings
    embedder = DualSpaceEmbedder(
        clip_model=args.clip_model,
        t5_model=args.t5_model,
    )

    # Profile
    profiles, all_prompts, template_embs = profile_all(
        templates, components, embedder,
        n_samples=args.samples, seed=args.seed, space=args.space,
    )

    # Baseline comparison
    baseline_profiles: list[TemplateProfile] | None = None
    if args.baseline_components and args.baseline_components.exists():
        baseline_components = load_components_yaml(args.baseline_components)
        print(f"\nProfiling baseline: {args.baseline_components}")
        baseline_profiles, _, _ = profile_all(
            templates, baseline_components, embedder,
            n_samples=args.samples, seed=args.seed, space=args.space,
        )

    # Display
    print_profiles(profiles, baseline_profiles)

    if args.show_matrix:
        names, matrix = pairwise_matrix(template_embs)
        print_matrix(names, matrix)

    if args.show_samples:
        print(f"\n{'='*100}")
        print(f"SAMPLE PROMPTS ({args.show_samples} per template)")
        print(f"{'='*100}")
        for name, prompts in all_prompts.items():
            print(f"\n{name}:")
            for p in prompts[:args.show_samples]:
                trunc = p[:100] + "..." if len(p) > 100 else p
                print(f"  {trunc}")

    # Save stats
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "components": str(args.components),
        "n_samples": args.samples,
        "embedding_space": args.space,
        "profiles": {},
    }
    for p in profiles:
        report["profiles"][p.name] = {
            "structure": p.structure,
            "combined_score": round(p.combined_score(), 4),
            "internal_score": round(p.internal_score(), 4),
            "uniqueness_score": round(p.uniqueness_score(), 4),
            "internal_mean": round(p.internal_mean, 4),
            "internal_range": round(p.internal_range, 4),
            "cross_mean": round(p.cross_mean, 4),
            "n_slots": p.n_slots,
            "slot_categories": p.slot_categories,
            "theoretical_combinations": p.theoretical_combinations,
        }

    if args.save_stats or args.output:
        out = args.save_stats or args.output
        with open(out, "w") as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nReport saved: {out}")

    # Free
    del embedder
    print("\nDone!")


if __name__ == "__main__":
    main()
