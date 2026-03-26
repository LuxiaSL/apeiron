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
    """Diversity statistics for a single template in dual embedding spaces.

    Uses gated scoring analogous to the word selection pipeline:
      Internal diversity → AND gate (must be diverse in BOTH spaces)
      Cross-template uniqueness → OR gate (distinct in EITHER space)
    """

    name: str
    structure: str
    n_samples: int

    # CLIP space (visual diversity)
    clip_internal_mean: float = 0.0
    clip_internal_range: float = 0.0
    clip_cross_mean: float = 0.0

    # T5 space (semantic diversity) — None if single-space
    t5_internal_mean: float | None = None
    t5_internal_range: float | None = None
    t5_cross_mean: float | None = None

    # Structural info
    n_slots: int = 0
    slot_categories: list[str] = field(default_factory=list)
    multi_slots: int = 0
    theoretical_combinations: int = 0

    def _score(self, mean: float, rng: float) -> float:
        return (1 - mean) * 0.6 + rng * 0.4

    def clip_internal_score(self) -> float:
        return self._score(self.clip_internal_mean, self.clip_internal_range)

    def t5_internal_score(self) -> float | None:
        if self.t5_internal_mean is None or self.t5_internal_range is None:
            return None
        return self._score(self.t5_internal_mean, self.t5_internal_range)

    def clip_uniqueness(self) -> float:
        return 1 - self.clip_cross_mean

    def t5_uniqueness(self) -> float | None:
        if self.t5_cross_mean is None:
            return None
        return 1 - self.t5_cross_mean

    def internal_score(self) -> float:
        """AND gate: min of both spaces (must be diverse in BOTH)."""
        clip = self.clip_internal_score()
        t5 = self.t5_internal_score()
        if t5 is None:
            return clip
        return min(clip, t5)

    def uniqueness_score(self) -> float:
        """OR gate: max of both spaces (distinct in EITHER)."""
        clip = self.clip_uniqueness()
        t5 = self.t5_uniqueness()
        if t5 is None:
            return clip
        return max(clip, t5)

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


def encode_prompts_dual(
    prompts: list[str],
    embedder: DualSpaceEmbedder,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode prompts in both spaces. Returns (clip_emb, t5_emb)."""
    clip_emb = embedder.clip_embedder.encode_batch(prompts)
    t5_emb = None
    if embedder.t5_embedder:
        t5_emb = embedder.t5_embedder.encode_batch(prompts)
    return clip_emb, t5_emb


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
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[TemplateProfile], dict[str, list[str]], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Profile all templates in dual embedding spaces.

    Returns (profiles, all_prompts, clip_embeddings, t5_embeddings).
    """
    has_t5 = embedder.t5_embedder is not None

    # Phase 1: Generate all prompts
    total_prompts = n_samples * len(templates)
    print(f"\n[1/3] Generating {n_samples} prompts per template ({total_prompts} total)...")
    all_prompts: dict[str, list[str]] = {}
    for name, template in templates.items():
        prompts = generate_prompts(template, components, n_samples, seed=seed)
        all_prompts[name] = prompts

    # Phase 2: Embed all prompts in one batch (both spaces)
    flat: list[str] = []
    indices: dict[str, tuple[int, int]] = {}
    idx = 0
    for name, prompts in all_prompts.items():
        indices[name] = (idx, idx + len(prompts))
        flat.extend(prompts)
        idx += len(prompts)

    print(f"[2/3] Embedding {len(flat)} prompts in {'CLIP + T5' if has_t5 else 'CLIP'}...")
    all_clip, all_t5 = encode_prompts_dual(flat, embedder)

    clip_embs: dict[str, np.ndarray] = {
        name: all_clip[s:e] for name, (s, e) in indices.items()
    }
    t5_embs: dict[str, np.ndarray] | None = None
    if all_t5 is not None:
        t5_embs = {name: all_t5[s:e] for name, (s, e) in indices.items()}

    # Phase 3: Compute gated statistics
    print("[3/3] Computing dual-space diversity statistics...")
    profiles: list[TemplateProfile] = []

    for name, template in templates.items():
        structure = template["structure"]
        slots = parse_slots(structure)

        # CLIP space
        clip_int = similarity_stats(clip_embs[name])
        clip_others = [clip_embs[n] for n in clip_embs if n != name]
        clip_cross = cross_similarity(clip_embs[name], np.vstack(clip_others)) if clip_others else {"mean": 0}

        # T5 space
        t5_int_mean, t5_int_range, t5_cross_mean_val = None, None, None
        if t5_embs is not None:
            t5_int = similarity_stats(t5_embs[name])
            t5_others = [t5_embs[n] for n in t5_embs if n != name]
            t5_cross = cross_similarity(t5_embs[name], np.vstack(t5_others)) if t5_others else {"mean": 0}
            t5_int_mean = t5_int["mean"]
            t5_int_range = t5_int["range"]
            t5_cross_mean_val = t5_cross["mean"]

        # Category usage
        cat_counts: dict[str, int] = {}
        for cat, count, _ in slots:
            cat_counts[cat] = cat_counts.get(cat, 0) + count

        profile = TemplateProfile(
            name=name,
            structure=structure,
            n_samples=n_samples,
            clip_internal_mean=clip_int["mean"],
            clip_internal_range=clip_int["range"],
            clip_cross_mean=clip_cross["mean"],
            t5_internal_mean=t5_int_mean,
            t5_internal_range=t5_int_range,
            t5_cross_mean=t5_cross_mean_val,
            n_slots=len(slots),
            slot_categories=[cat for cat, _, _ in slots],
            multi_slots=sum(1 for c in cat_counts.values() if c > 1),
            theoretical_combinations=compute_theoretical_combinations(slots, components),
        )
        profiles.append(profile)

        gate_str = ""
        if has_t5:
            gate_str = (f" [AND={profile.internal_score():.3f}, "
                        f"OR={profile.uniqueness_score():.3f}]")
        print(f"  {name}: clip_div={profile.clip_internal_score():.3f}, "
              f"t5_div={profile.t5_internal_score() or 'N/A'}"
              f"{gate_str}")

    return profiles, all_prompts, clip_embs, t5_embs


def pairwise_matrix(
    clip_embs: dict[str, np.ndarray],
    t5_embs: dict[str, np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Pairwise similarity matrices between templates in both spaces."""
    names = list(clip_embs.keys())
    n = len(names)
    clip_matrix = np.zeros((n, n))
    t5_matrix = np.zeros((n, n)) if t5_embs else None

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                clip_matrix[i, j] = similarity_stats(clip_embs[a])["mean"]
                if t5_matrix is not None and t5_embs:
                    t5_matrix[i, j] = similarity_stats(t5_embs[a])["mean"]
            else:
                clip_matrix[i, j] = cross_similarity(clip_embs[a], clip_embs[b])["mean"]
                if t5_matrix is not None and t5_embs:
                    t5_matrix[i, j] = cross_similarity(t5_embs[a], t5_embs[b])["mean"]

    return names, clip_matrix, t5_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════


def print_profiles(
    profiles: list[TemplateProfile],
    baseline: list[TemplateProfile] | None = None,
) -> None:
    """Print template diversity rankings with dual-space gated scores."""
    print(f"\n{'='*115}")
    print("TEMPLATE DIVERSITY PROFILES (AND-gated internal, OR-gated uniqueness)")
    print(f"{'='*115}")

    has_t5 = profiles[0].t5_internal_mean is not None if profiles else False
    base_lookup = {p.name: p for p in baseline} if baseline else {}
    ranked = sorted(profiles, key=lambda p: p.combined_score(), reverse=True)

    if has_t5:
        if base_lookup:
            print(f"\n{'Rk':<4} {'Template':<22} {'Combined':>9} {'d':>7} {'AND(int)':>9} {'d':>7} "
                  f"{'OR(uniq)':>9} {'d':>7} {'CLIPint':>8} {'T5int':>8}")
            print("-" * 115)
        else:
            print(f"\n{'Rk':<4} {'Template':<22} {'Combined':>9} {'AND(int)':>9} {'OR(uniq)':>9} "
                  f"{'CLIPint':>8} {'T5int':>8} {'CLIPx':>8} {'T5x':>8} {'Combos':>12}")
            print("-" * 115)
    else:
        print(f"\n{'Rk':<4} {'Template':<22} {'Combined':>9} {'Internal':>9} {'Unique':>9} "
              f"{'IntMean':>8} {'IntRange':>9} {'XMean':>8} {'Combos':>12}")
        print("-" * 105)

    for rank, p in enumerate(ranked, 1):
        if base_lookup and p.name in base_lookup:
            b = base_lookup[p.name]
            dc = p.combined_score() - b.combined_score()
            di = p.internal_score() - b.internal_score()
            du = p.uniqueness_score() - b.uniqueness_score()
            if has_t5:
                print(f"{rank:<4} {p.name:<22} {p.combined_score():>9.3f} {dc:>+7.3f} "
                      f"{p.internal_score():>9.3f} {di:>+7.3f} {p.uniqueness_score():>9.3f} {du:>+7.3f} "
                      f"{p.clip_internal_mean:>8.3f} {p.t5_internal_mean:>8.3f}")
            else:
                print(f"{rank:<4} {p.name:<22} {p.combined_score():>9.3f} {dc:>+7.3f} "
                      f"{p.internal_score():>9.3f} {di:>+7.3f} {p.uniqueness_score():>9.3f} {du:>+7.3f}")
        else:
            combos = f"{p.theoretical_combinations:.2e}" if p.theoretical_combinations > 1e9 else str(p.theoretical_combinations)
            if has_t5:
                print(f"{rank:<4} {p.name:<22} {p.combined_score():>9.3f} {p.internal_score():>9.3f} "
                      f"{p.uniqueness_score():>9.3f} {p.clip_internal_mean:>8.3f} "
                      f"{p.t5_internal_mean:>8.3f} {p.clip_cross_mean:>8.3f} "
                      f"{p.t5_cross_mean:>8.3f} {combos:>12}")
            else:
                print(f"{rank:<4} {p.name:<22} {p.combined_score():>9.3f} {p.internal_score():>9.3f} "
                      f"{p.uniqueness_score():>9.3f} {p.clip_internal_mean:>8.3f} {p.clip_internal_range:>9.3f} "
                      f"{p.clip_cross_mean:>8.3f} {combos:>12}")


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
    parser.add_argument("--samples", "-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model", default="openclip")
    parser.add_argument("--t5-model", default="google/t5-v1_1-xxl",
                        help="T5 model (default: google/t5-v1_1-xxl, pass 'none' to disable)")
    parser.add_argument("--device", default=None,
                        help="Force device (cuda, cuda:0, cpu). Default: auto-detect")
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

    t5_model = args.t5_model if args.t5_model and args.t5_model.lower() != "none" else None
    total = sum(len(v) for v in components.values())
    total_prompts = args.samples * len(templates)
    print(f"{'='*100}")
    print("TEMPLATE DIVERSITY PROFILER (Dual-Space Gated)")
    print(f"{'='*100}")
    print(f"  Templates: {len(templates)}")
    print(f"  Components: {total} across {len(components)} categories")
    print(f"  Samples per template: {args.samples} ({total_prompts} total)")
    print(f"  CLIP model: {args.clip_model}")
    print(f"  T5 model: {t5_model or '(disabled)'}")
    print(f"  Gating: AND(internal diversity), OR(cross-template uniqueness)")

    # Category usage (no models needed)
    if args.show_categories:
        print_category_usage(templates, components)

    # Load embedder
    embedder = DualSpaceEmbedder(
        clip_model=args.clip_model,
        t5_model=t5_model,
    )

    # Profile
    profiles, all_prompts, clip_embs, t5_embs = profile_all(
        templates, components, embedder,
        n_samples=args.samples, seed=args.seed,
    )

    # Baseline comparison
    baseline_profiles: list[TemplateProfile] | None = None
    if args.baseline_components and args.baseline_components.exists():
        baseline_components = load_components_yaml(args.baseline_components)
        print(f"\nProfiling baseline: {args.baseline_components}")
        baseline_profiles, _, _, _ = profile_all(
            templates, baseline_components, embedder,
            n_samples=args.samples, seed=args.seed,
        )

    # Display
    print_profiles(profiles, baseline_profiles)

    if args.show_matrix:
        names, clip_matrix, t5_matrix = pairwise_matrix(clip_embs, t5_embs)
        print(f"\n  OpenCLIP space:")
        print_matrix(names, clip_matrix)
        if t5_matrix is not None:
            print(f"\n  T5-XXL space:")
            print_matrix(names, t5_matrix)

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
        "clip_model": args.clip_model,
        "t5_model": t5_model,
        "gating": "AND(internal), OR(uniqueness)",
        "profiles": {},
    }
    for p in profiles:
        entry: dict[str, Any] = {
            "structure": p.structure,
            "combined_score": round(p.combined_score(), 4),
            "internal_score_AND": round(p.internal_score(), 4),
            "uniqueness_score_OR": round(p.uniqueness_score(), 4),
            "clip_internal_mean": round(p.clip_internal_mean, 4),
            "clip_internal_range": round(p.clip_internal_range, 4),
            "clip_cross_mean": round(p.clip_cross_mean, 4),
            "n_slots": p.n_slots,
            "slot_categories": p.slot_categories,
            "theoretical_combinations": p.theoretical_combinations,
        }
        if p.t5_internal_mean is not None:
            entry["t5_internal_mean"] = round(p.t5_internal_mean, 4)
            entry["t5_internal_range"] = round(p.t5_internal_range, 4) if p.t5_internal_range else 0
            entry["t5_cross_mean"] = round(p.t5_cross_mean, 4) if p.t5_cross_mean else 0
            entry["clip_internal_score"] = round(p.clip_internal_score(), 4)
            entry["t5_internal_score"] = round(p.t5_internal_score() or 0, 4)
            entry["clip_uniqueness"] = round(p.clip_uniqueness(), 4)
            entry["t5_uniqueness"] = round(p.t5_uniqueness() or 0, 4)
        report["profiles"][p.name] = entry

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
