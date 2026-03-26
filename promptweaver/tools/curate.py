#!/usr/bin/env python3
"""
Post-Selection Curation
═══════════════════════════════════════════════════════════════════════════════

Applies targeted fixes to the selected component pool:
1. Remove garbage entries (YAML artifacts, non-visual terms)
2. Fix miscategorizations (move items to correct category)
3. Cull near-duplicates (pick the better of each pair)
4. Restore high-value original terms that were lost in expansion
5. Integrate void-mined discoveries (filtered for visual relevance via CLIP)

Also provides a --review mode that just lists issues without changing anything.

Usage:
    # Review mode — show all issues
    uv run python -m promptweaver.tools.curate \
        --selected data/selected_final.yaml \
        --original data/components.yaml \
        --review

    # Apply fixes
    uv run python -m promptweaver.tools.curate \
        --selected data/selected_final.yaml \
        --original data/components.yaml \
        -o data/curated_components.yaml

    # Also integrate void discoveries
    uv run python -m promptweaver.tools.curate \
        --selected data/selected_final.yaml \
        --original data/components.yaml \
        --probe data/probe_results.yaml \
        --candidates data/generated_candidates_20260325_193614.yaml \
        -o data/curated_components.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualSpaceEmbedder,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN ISSUES — from the curation review
# ═══════════════════════════════════════════════════════════════════════════════

# Garbage entries to remove entirely (YAML artifacts that survived as "words")
GARBAGE = {
    # From base CLIP selection
    "liminal/transitional:",
    "transformation:",
    "atmospheric/extreme:",
    "consumption:",
    "built/architectural:",
    # From gated OpenCLIP+T5-XXL selection
    "natural/geological:",
    "water/aquatic:",
    "preservation/stasis:",
    "decay/entropy:",
    "growth/emergence:",
    "总 total internal reflection",  # Corrupted: Chinese char prefix on valid entry
}

# Items to move: {word: (from_category, to_category)}
# Only fires if the word is found in from_category, so old entries are harmless
# when running against a pool that doesn't have them.
MISCATEGORIZED: dict[str, tuple[str, str]] = {
    # ── Base CLIP selection issues ────────────────────────────────────
    "floating isolated": ("material_substance", "spatial_logic"),
    "split complementary": ("subject_form", "color_logic"),
    "chitin": ("subject_form", "material_substance"),
    "mercury": ("subject_form", "material_substance"),
    "lapis lazuli": ("subject_form", "material_substance"),
    "larimar": ("subject_form", "material_substance"),
    "calotype negative": ("subject_form", "medium_render"),
    "tyndall beam": ("subject_form", "light_behavior"),
    "contre-jour": ("subject_form", "light_behavior"),
    "graduated recession": ("subject_form", "spatial_logic"),
    "chaotic noise": ("atmosphere_field", "texture_density"),
    "warm dune versus cold surf": ("atmosphere_field", "color_logic"),
    "barn door cutoff": ("scale_perspective", "light_behavior"),
    "mri false color": ("light_behavior", "medium_render"),
    "smooth gradient": ("light_behavior", "texture_density"),
    "glitch echo streak": ("light_behavior", "phenomenon_pattern"),
    "macroblock drift": ("light_behavior", "phenomenon_pattern"),
    "alla prima oil": ("material_substance", "medium_render"),
    "basket weave": ("material_substance", "texture_density"),
    "diamond offset": ("material_substance", "spatial_logic"),
    "tissue scarification": ("medium_render", "temporal_state"),
    "cyc wash": ("phenomenon_pattern", "light_behavior"),
    "gridded snoot spot": ("phenomenon_pattern", "light_behavior"),
    "checkerboard pattern": ("texture_density", "spatial_logic"),
    "differential interference contrast": ("spatial_logic", "medium_render"),
    "mri cross-section": ("spatial_logic", "medium_render"),
    "through a rain-streaked window": ("temporal_state", "scale_perspective"),
    "cryo-electron grid": ("temporal_state", "medium_render"),
    # ── Gated selection: subject_form attractor ───────────────────────
    # Both OpenCLIP and T5-XXL see these as "form-like" because they have
    # strong physical shapes, but they're fundamentally materials.
    "glass": ("subject_form", "material_substance"),
    "bone": ("subject_form", "material_substance"),
    "silk": ("subject_form", "material_substance"),
    "wax": ("subject_form", "material_substance"),
    "macro": ("subject_form", "scale_perspective"),
    "aerial": ("subject_form", "scale_perspective"),
}

# Near-duplicates: keep the first, drop the rest
DUPLICATES_TO_CULL: list[tuple[str, list[str]]] = [
    # (keep, [drop these])
    # Base CLIP selection
    ("mica flake swirl", ["mica flake suspension", "mica dust shimmer"]),
    ("salt spray", ["salt spray fog"]),
    ("volcanic ashfall", ["volcanic ash cloud"]),
    ("papercut silhouette", ["papercutting silhouette"]),
    ("julia set filament curl", ["julia set swirl"]),
    ("gosper curve meander", ["gosper curve growth"]),
    ("dragon curve unfold", ["dragon curve spiral"]),
    ("concrete spalling flakes", ["concrete spall fracture"]),
    ("from a blimp gondola", ["from a gondola", "panoramic gondola"]),
    ("macrame knots", ["macrame knotwork"]),
    ("tufted cushion", ["tufted wool"]),
    ("hospital service corridor", ["hotel service corridor"]),
    ("noctilucent cloud glow", ["lenticular cloud glow"]),
    ("fool's gold", ["pyrite nodules"]),
    # Gated selection
    ("fanned divergence", ["fanning divergence"]),  # spatial_logic spelling variant
]

# High-value originals to restore (only if not already present)
RESTORE_ORIGINALS: dict[str, list[str]] = {
    "atmosphere_field": ["dense fog", "underwater caustics", "smoke haze", "dust motes",
                         "campfire embers", "void darkness"],
    "color_logic": ["iridescent", "complementary orange and blue", "high contrast black and gold"],
    "light_behavior": ["volumetric god rays", "crepuscular rays", "bokeh balls",
                       "subsurface scattering", "dappled light", "harsh directional", "soft diffused"],
    "material_substance": ["glass", "bone", "wax", "aerogel", "silk", "ferrofluid"],
    "medium_render": ["oil impasto", "charcoal sketch", "risograph print", "matte painting"],
    "scale_perspective": ["tilt-shift miniature", "electron microscope",
                          "vast landscape scale", "keyhole view", "dollhouse cutaway"],
    "spatial_logic": ["radial symmetry", "vanishing point focus", "negative space dominant",
                      "central emergence", "infinite regression", "bilateral symmetry",
                      "diagonal tension", "scattered fragments"],
    "temporal_state": ["nascent", "blooming", "fossilized", "freshly formed",
                       "crystallizing", "frozen mid-transformation", "resin-entombed"],
    "texture_density": ["smooth gradient", "rough hewn", "porous foam",
                        "gossamer threads", "granular scatter"],
    "phenomenon_pattern": ["voronoi cells", "chromatic aberration", "fractal branching",
                           "pixel sort", "moiré interference"],
    "setting_location": ["orbital station", "abyssal trench", "salt flat",
                         "neon-lit alley", "fog shrouded moorland"],
    "subject_form": ["void", "membrane", "lattice", "plume", "threshold",
                     "vessel", "monolith", "shard", "veil", "shroud"],
}

# Words to remove entirely (visually meaningless, too vague, or miscategorized beyond repair)
REMOVE_WORDS = {
    "vapor",  # Too vague
    "osmium tetroxide",  # Colorless toxic gas, no visual
    "amine vapor",  # No visual signature
    "printer toner cloud",  # Mundane
    "baking soda dust",  # Mundane
    "neon rain",  # Too vague / setting-like
    "oobleck",  # Material, in wrong category
    "umbra",  # Light term in wrong category
    "domino progression",  # Unclear as light behavior
    "seamless blend",  # Generic compositing term
    "flange",  # Mechanical part, not spatial
    "valance",  # Textile item, not spatial
    # From final review — cross-category dupes (keep in better-fitting category)
    "bismuth crystal",  # Dupe: keep in material_substance, remove from phenomenon_pattern
    "lava flow crust",  # Dupe: keep in phenomenon_pattern, remove from texture_density
    # Bloated clusters — collapse near-identical analogous entries
    "analogous rust to brick",
    "analogous cinnamon to rust",
    "analogous peach to salmon",
    "analogous coral to salmon",
    "analogous sandstone to ochre",
    "analogous sage to olive",
    # Bloated scientific imaging cluster
    "mri cross-section",  # Keep mri scan and mri false color
    "acoustic microscopy",  # Too boring visually
    # Misc from final review
    "marble sculpting studio",  # Setting, not medium
    "rococo winter garden",  # Style/setting, not color logic
    "monochromatic olive drab",  # Too flat
    "rendered lard",  # Near-dupe of rendered tallow
    "linen canvas tooth",  # Near-dupe of linen canvas grain
    "bubble column rise",  # Near-dupe of bubble stream rise
    "angular momentum",  # Physics, not composition
    "palindrome arrangement",  # Linguistic, no visual mapping
    # Gated selection
    "cologht film wash",  # Typo/corruption — not a real term
    "vapor",  # Too vague as subject_form (also listed above)
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP VISUAL RELEVANCE FILTER
# ═══════════════════════════════════════════════════════════════════════════════


def filter_visual_relevance(
    words: list[str],
    clip_embedder: Any,
    threshold: float = 0.15,
) -> list[tuple[str, float]]:
    """
    Score words by CLIP text embedding norm as a proxy for visual relevance.

    CLIP text embeddings for visually meaningful words tend to have higher
    norms (before normalization) because the model has stronger activations
    for concepts it was trained on (image-text pairs). Functional words,
    grammatical tokens, and non-visual concepts produce weaker activations.

    Returns list of (word, score) for words above threshold, sorted by score.
    """
    import torch

    all_scores: list[tuple[str, float]] = []

    for start in range(0, len(words), 64):
        batch = words[start:start + 64]
        with torch.no_grad():
            inputs = clip_embedder.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(clip_embedder.device) for k, v in inputs.items()}
            text_outputs = clip_embedder.model.text_model(**inputs)
            features = clip_embedder.model.text_projection(text_outputs.pooler_output)
            # Use pre-normalization norm as visual relevance proxy
            norms = features.norm(dim=-1).cpu().numpy()

        for i, word in enumerate(batch):
            all_scores.append((word, float(norms[i])))

    # Normalize scores to [0, 1] range
    max_norm = max(s for _, s in all_scores) if all_scores else 1.0
    all_scores = [(w, s / max_norm) for w, s in all_scores]

    return [(w, s) for w, s in all_scores if s > threshold]


# ═══════════════════════════════════════════════════════════════════════════════
# CURATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def apply_curation(
    selected: dict[str, list[str]],
    original: dict[str, list[str]] | None = None,
    review_only: bool = False,
) -> dict[str, list[str]]:
    """Apply all curation fixes. Returns the curated pool."""
    curated = {cat: list(words) for cat, words in selected.items()}
    stats = {"removed_garbage": 0, "moved": 0, "culled_dupes": 0,
             "restored": 0, "removed_bad": 0}

    # 1. Remove garbage
    for cat in curated:
        for word in list(curated[cat]):
            if word in GARBAGE:
                if review_only:
                    print(f"  [GARBAGE] Remove '{word}' from {cat}")
                else:
                    curated[cat].remove(word)
                stats["removed_garbage"] += 1

    # 2. Remove visually meaningless words
    for cat in curated:
        for word in list(curated[cat]):
            if word in REMOVE_WORDS:
                if review_only:
                    print(f"  [REMOVE] '{word}' from {cat}")
                else:
                    curated[cat].remove(word)
                stats["removed_bad"] += 1

    # 3. Fix miscategorizations
    for word, (from_cat, to_cat) in MISCATEGORIZED.items():
        if from_cat in curated and word in curated[from_cat]:
            if review_only:
                print(f"  [MOVE] '{word}': {from_cat} → {to_cat}")
            else:
                curated[from_cat].remove(word)
                curated.setdefault(to_cat, [])
                if word not in curated[to_cat]:
                    curated[to_cat].append(word)
            stats["moved"] += 1

    # 4. Cull near-duplicates
    for keep, drop_list in DUPLICATES_TO_CULL:
        for drop in drop_list:
            for cat in curated:
                if drop in curated[cat]:
                    if review_only:
                        print(f"  [DUPE] Remove '{drop}' (keeping '{keep}') from {cat}")
                    else:
                        curated[cat].remove(drop)
                    stats["culled_dupes"] += 1

    # 5. Restore high-value originals
    if original:
        for cat, words_to_restore in RESTORE_ORIGINALS.items():
            if cat not in curated:
                continue
            for word in words_to_restore:
                # Only restore if it was in the original AND not already present
                if (word in original.get(cat, [])
                        and word not in curated[cat]
                        # Also check it's not in another category already
                        and not any(word in curated[c] for c in curated if c != cat)):
                    if review_only:
                        print(f"  [RESTORE] '{word}' → {cat}")
                    else:
                        curated[cat].append(word)
                    stats["restored"] += 1

    print(f"\nCuration summary:")
    print(f"  Garbage removed:    {stats['removed_garbage']}")
    print(f"  Bad words removed:  {stats['removed_bad']}")
    print(f"  Miscategorizations: {stats['moved']} moved")
    print(f"  Duplicates culled:  {stats['culled_dupes']}")
    print(f"  Originals restored: {stats['restored']}")

    total = sum(len(w) for w in curated.values())
    print(f"  Final pool size:    {total} across {len(curated)} categories")

    return curated


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate the selected component pool — fix issues and restore quality",
    )
    parser.add_argument("--selected", "-s", type=Path, required=True,
                        help="Selected components YAML to curate")
    parser.add_argument("--original", type=Path,
                        help="Original components.yaml for restoration")
    parser.add_argument("--candidates", type=Path,
                        help="Full candidate pool for additional sourcing")
    parser.add_argument("--probe", type=Path,
                        help="Probe results YAML for void discoveries")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output curated YAML")
    parser.add_argument("--review", action="store_true",
                        help="Review mode — list issues without changing anything")
    parser.add_argument("--clip-filter", action="store_true",
                        help="Apply CLIP visual relevance filter to void discoveries")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.selected.exists():
        logger.error(f"Selected not found: {args.selected}")
        sys.exit(1)

    selected = load_components_yaml(args.selected)
    original = load_components_yaml(args.original) if args.original and args.original.exists() else None

    total_before = sum(len(w) for w in selected.values())
    print(f"{'='*60}")
    print(f"CURATION {'REVIEW' if args.review else 'PASS'}")
    print(f"{'='*60}")
    print(f"  Input: {args.selected} ({total_before} items)")
    if original:
        print(f"  Original: {args.original}")

    # Apply known fixes
    curated = apply_curation(selected, original, review_only=args.review)

    # Integrate probe discoveries if provided
    if args.probe and args.probe.exists() and not args.review:
        with open(args.probe) as f:
            probe_data = yaml.safe_load(f)

        # Extract nearest words to void points
        void_words: list[str] = []
        for void in probe_data.get("voids", []):
            nearest = void.get("nearest", "")
            if nearest and nearest not in GARBAGE and nearest not in REMOVE_WORDS:
                void_words.append(nearest)

        if void_words and args.clip_filter:
            print(f"\nFiltering {len(void_words)} void discoveries for visual relevance...")
            embedder = DualSpaceEmbedder(clip_model="openai/clip-vit-base-patch32", t5_model=None)
            scored = filter_visual_relevance(void_words, embedder.clip_embedder)
            print(f"  {len(scored)} passed CLIP visual filter")
            for word, score in scored[:10]:
                print(f"    '{word}': {score:.3f}")

    # Per-category counts
    if not args.review:
        print(f"\n{'Category':<25} {'Count':>5}")
        print("-" * 32)
        for cat, words in sorted(curated.items()):
            print(f"  {cat:<23} {len(words):>5}")

    # Output
    if args.output and not args.review:
        output_data = {"components": {}}
        for cat, words in sorted(curated.items()):
            output_data["components"][cat] = [{"word": w} for w in words]

        with open(args.output, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nCurated components saved: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
