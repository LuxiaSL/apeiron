#!/usr/bin/env python3
"""
Component Quality Analysis
═══════════════════════════════════════════════════════════════════════════════

Standalone analysis tool for examining component quality at any stage:
- Pre-selection: analyze raw candidates for redundancy and contamination
- Post-selection: verify the selected set's diversity
- Comparison: before/after analysis of old vs new components
- Prompt simulation: generate sample prompts and measure end-to-end diversity

Usage:
    # Analyze a components YAML
    uv run python -m promptweaver.tools.analyze \
        --input data/components.yaml

    # Compare old vs new
    uv run python -m promptweaver.tools.analyze \
        --input data/selected_components.yaml \
        --baseline data/components.yaml

    # Simulate prompts and measure diversity
    uv run python -m promptweaver.tools.analyze \
        --input data/selected_components.yaml \
        --simulate 200

    # CLIP-only (faster)
    uv run python -m promptweaver.tools.analyze \
        --input data/components.yaml --clip-only
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualSpaceEmbedder,
    DualEmbeddings,
    analyze_diversity,
    cross_category_contamination,
    load_components_yaml,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def compare_pools(
    baseline_embs: dict[str, DualEmbeddings],
    proposed_embs: dict[str, DualEmbeddings],
    alpha: float,
) -> None:
    """Print side-by-side comparison of baseline vs proposed pools."""
    all_cats = sorted(set(baseline_embs.keys()) | set(proposed_embs.keys()))

    print(
        f"\n{'Category':<25} {'Old#':>4} {'New#':>4} "
        f"{'Old Mean':>8} {'New Mean':>8} {'Delta':>7} {'Verdict':>8}"
    )
    print("-" * 72)

    improved = 0
    degraded = 0

    for cat in all_cats:
        old = baseline_embs.get(cat)
        new = proposed_embs.get(cat)

        if old is None:
            print(f"{cat:<25} {'—':>4} {len(new.words):>4}    (new category)")
            continue
        if new is None:
            print(f"{cat:<25} {len(old.words):>4} {'—':>4}    (removed)")
            continue

        old_sim = old.joint_similarity_matrix(alpha)
        new_sim = new.joint_similarity_matrix(alpha)
        old_stats = analyze_diversity(old_sim, old.words)
        new_stats = analyze_diversity(new_sim, new.words)

        delta = new_stats.mean_similarity - old_stats.mean_similarity
        # Lower mean similarity = better diversity
        verdict = "better" if delta < -0.005 else ("worse" if delta > 0.005 else "same")
        if verdict == "better":
            improved += 1
        elif verdict == "worse":
            degraded += 1

        print(
            f"{cat:<25} {len(old.words):>4} {len(new.words):>4} "
            f"{old_stats.mean_similarity:>8.3f} {new_stats.mean_similarity:>8.3f} "
            f"{delta:>+7.3f} {verdict:>8}"
        )

    print(f"\nImproved: {improved} | Degraded: {degraded} | Unchanged: {len(all_cats) - improved - degraded}")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


def load_templates(path: Path) -> list[dict[str, Any]]:
    """Load templates from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("templates", [])


def parse_slots(structure: str) -> list[tuple[str, int, str]]:
    """Parse {category}, {category:N}, {category:N:sep} → (cat, count, sep)."""
    pattern = r"\{(\w+)(?::(\d+))?(?::([^}]*))?\}"
    return [
        (m.group(1), int(m.group(2)) if m.group(2) else 1, m.group(3) or " ")
        for m in re.finditer(pattern, structure)
    ]


def generate_sample_prompts(
    templates: list[dict[str, Any]],
    components: dict[str, list[str]],
    n_per_template: int = 50,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate sample prompts for diversity measurement."""
    rng = random.Random(seed)
    prompts_by_template: dict[str, list[str]] = {}

    for template in templates:
        tid = template["id"]
        structure = template["structure"]
        slots = parse_slots(structure)

        prompts: list[str] = []
        for _ in range(n_per_template):
            # Compute needs per category
            needs: dict[str, int] = {}
            for cat, count, _ in slots:
                needs[cat] = needs.get(cat, 0) + count

            # Sample without replacement per category
            selections: dict[str, list[str]] = {}
            for cat, total in needs.items():
                pool = components.get(cat, [])
                if not pool:
                    continue
                selections[cat] = rng.sample(pool, min(total, len(pool)))

            # Fill template
            prompt = structure
            consumed: dict[str, int] = {}
            for cat, count, sep in slots:
                idx = consumed.get(cat, 0)
                chosen = selections.get(cat, [])
                batch = chosen[idx: idx + count]
                consumed[cat] = idx + count
                replacement = sep.join(batch) if batch else f"[missing {cat}]"

                if count > 1:
                    pat = f"{{{cat}:{count}:{sep}}}" if sep != " " else f"{{{cat}:{count}}}"
                else:
                    pat = f"{{{cat}}}"
                prompt = prompt.replace(pat, replacement, 1)

            prompts.append(prompt)

        prompts_by_template[tid] = prompts

    return prompts_by_template


def analyze_prompt_diversity(
    prompts_by_template: dict[str, list[str]],
    embedder: DualSpaceEmbedder,
    alpha: float,
) -> None:
    """Embed generated prompts and measure diversity."""
    print(f"\n{'='*60}")
    print("PROMPT DIVERSITY SIMULATION")
    print(f"{'='*60}")

    all_prompts: list[str] = []
    template_ranges: dict[str, tuple[int, int]] = {}

    for tid, prompts in prompts_by_template.items():
        start = len(all_prompts)
        all_prompts.extend(prompts)
        template_ranges[tid] = (start, len(all_prompts))

    if not all_prompts:
        print("  No prompts generated")
        return

    # Embed all prompts
    logger.info(f"Embedding {len(all_prompts)} sample prompts...")
    emb = embedder.embed(all_prompts)
    sim = emb.joint_similarity_matrix(alpha)

    # Per-template diversity
    print(
        f"\n{'Template':<25} {'N':>4} {'Mean Sim':>8} {'Min':>6} {'Max':>6}"
    )
    print("-" * 52)

    for tid, (start, end) in sorted(template_ranges.items()):
        sub_sim = sim[start:end, start:end]
        sub_words = all_prompts[start:end]
        stats = analyze_diversity(sub_sim, sub_words)
        print(
            f"{tid:<25} {stats.count:>4} {stats.mean_similarity:>8.3f} "
            f"{stats.min_similarity:>6.3f} {stats.max_similarity:>6.3f}"
        )

    # Global diversity
    global_stats = analyze_diversity(sim, all_prompts)
    print(f"\n{'Global':<25} {global_stats.count:>4} {global_stats.mean_similarity:>8.3f} "
          f"{global_stats.min_similarity:>6.3f} {global_stats.max_similarity:>6.3f}")

    # Cross-template similarity
    print(f"\nCross-template similarity matrix (mean similarity between templates):")
    tids = sorted(template_ranges.keys())

    # Header
    header = f"{'':>18}"
    for tid in tids:
        header += f" {tid[:7]:>7}"
    print(header)

    for tid_a in tids:
        row = f"{tid_a:>18}"
        sa, ea = template_ranges[tid_a]
        for tid_b in tids:
            sb, eb = template_ranges[tid_b]
            cross = sim[sa:ea, sb:eb]
            if tid_a == tid_b:
                row += f" {'—':>7}"
            else:
                row += f" {cross.mean():>7.3f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze component pool quality and diversity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Components YAML to analyze",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline YAML for before/after comparison",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        metavar="N",
        help="Generate N prompts per template and measure diversity",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="CLIP vs T5 weight (default: 0.5)",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--t5-model",
        default="google/flan-t5-large",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for redundancy warnings (default: 0.85)",
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

    t5_model = None if args.clip_only else args.t5_model
    embedder = DualSpaceEmbedder(
        clip_model=args.clip_model,
        t5_model=t5_model,
    )

    # Load and embed input
    input_cats = load_components_yaml(args.input)
    total = sum(len(v) for v in input_cats.values())
    print(f"{'='*60}")
    print(f"Analyzing: {args.input}")
    print(f"  {len(input_cats)} categories, {total} total words")
    print(f"{'='*60}")

    input_embs = embedder.embed_categories(input_cats)

    # Per-category diversity analysis
    print(
        f"\n{'Category':<25} {'Count':>5} {'Mean':>6} {'Min':>6} "
        f"{'Max':>6} {'Std':>6} {'Redund':>6}"
    )
    print("-" * 62)

    for cat, emb in sorted(input_embs.items()):
        sim = emb.joint_similarity_matrix(args.alpha)
        stats = analyze_diversity(sim, emb.words, args.redundancy_threshold)
        warn = " !" if stats.redundant_pairs > 0 else ""
        print(
            f"{cat:<25} {stats.count:>5} {stats.mean_similarity:>6.3f} "
            f"{stats.min_similarity:>6.3f} {stats.max_similarity:>6.3f} "
            f"{stats.std_similarity:>6.3f} {stats.redundant_pairs:>5}{warn}"
        )

    # Cross-category contamination
    contaminated = cross_category_contamination(input_embs, args.alpha)
    if contaminated:
        print(f"\nCross-category contamination: {len(contaminated)} items")
        for item in contaminated[:10]:
            print(
                f"  '{item['word']}': {item['assigned']} → {item['closest']} "
                f"(delta={item['delta']:+.3f})"
            )
        if len(contaminated) > 10:
            print(f"  ... and {len(contaminated) - 10} more")
    else:
        print("\nNo cross-category contamination detected.")

    # Baseline comparison
    if args.baseline:
        if not args.baseline.exists():
            logger.error(f"Baseline not found: {args.baseline}")
            sys.exit(1)

        baseline_cats = load_components_yaml(args.baseline)
        baseline_embs = embedder.embed_categories(baseline_cats)
        compare_pools(baseline_embs, input_embs, args.alpha)

    # Prompt simulation
    if args.simulate:
        templates_path = Path(__file__).parent.parent / "data" / "templates.yaml"
        if not templates_path.exists():
            logger.error(f"Templates not found: {templates_path}")
            sys.exit(1)

        templates = load_templates(templates_path)
        prompts = generate_sample_prompts(templates, input_cats, n_per_template=args.simulate)
        analyze_prompt_diversity(prompts, embedder, args.alpha)

    print("\nDone!")


if __name__ == "__main__":
    main()
