#!/usr/bin/env python3
"""
Multi-Model Embedding Comparison
═══════════════════════════════════════════════════════════════════════════════

Runs the component pool through multiple embedding models and compares
how each one discriminates between and within categories. Tells us which
models agree on diversity and where they diverge.

Models:
    1. CLIP (openai/clip-vit-base-patch32) — baseline visual-text
    2. OpenCLIP (ViT-bigG/14, LAION-2B) — strongest open CLIP
    3. SigLIP (google/siglip-base-patch16-256) — calibrated similarities
    4. T5 flan-t5-large — general linguistic encoder
    5. T5-v1_1-XXL — diffusion-model T5 (Flux/SD3 text encoder)

Usage:
    uv run python -m promptweaver.tools.multi_embed \
        -i data/curated_v2.yaml \
        -o data/multi_embed_report.yaml
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
import torch
import yaml

from .embeddings import load_components_yaml

logger = logging.getLogger(__name__)

ENCODE_BATCH = 64


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODERS — each returns (n, dim) L2-normalized embeddings
# ═══════════════════════════════════════════════════════════════════════════════


class BaseEncoder:
    name: str = ""
    dim: int = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class CLIPEncoder(BaseEncoder):
    name = "clip-vit-base-patch32"

    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPTokenizerFast
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP base on {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), ENCODE_BATCH):
            batch = texts[i:i + ENCODE_BATCH]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.text_model(**inputs)
            feat = self.model.text_projection(out.pooler_output)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        return np.concatenate(all_emb)


class OpenCLIPEncoder(BaseEncoder):
    name = "openclip-ViT-bigG-14"

    def __init__(self) -> None:
        import open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading OpenCLIP ViT-bigG/14 on {self.device}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
        self.model.eval()
        self.dim = 1280

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), ENCODE_BATCH):
            batch = texts[i:i + ENCODE_BATCH]
            tokens = self.tokenizer(batch).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().float().numpy())
        return np.concatenate(all_emb)


class SigLIPEncoder(BaseEncoder):
    name = "siglip-base-patch16-256"

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SigLIP on {self.device}")
        model_id = "google/siglip-base-patch16-256"
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dim = self.model.config.text_config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), ENCODE_BATCH):
            batch = texts[i:i + ENCODE_BATCH]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.get_text_features(**inputs)
            # Handle both tensor and object returns
            if hasattr(out, 'pooler_output'):
                feat = out.pooler_output
            elif hasattr(out, 'last_hidden_state'):
                feat = out.last_hidden_state[:, 0]
            else:
                feat = out
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().float().numpy())
        return np.concatenate(all_emb)


class T5Encoder(BaseEncoder):
    def __init__(self, model_name: str = "google/flan-t5-large") -> None:
        from transformers import T5EncoderModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = model_name.split("/")[-1]
        logger.info(f"Loading T5 encoder: {model_name} on {self.device}")
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dim = self.model.config.d_model

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), ENCODE_BATCH):
            batch = texts[i:i + ENCODE_BATCH]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            hidden = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            all_emb.append(pooled.cpu().float().numpy())
        return np.concatenate(all_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryStats:
    mean_sim: float
    min_sim: float
    max_sim: float
    std_sim: float
    redundant_pairs: int  # > 0.85


def analyze_category(emb: np.ndarray) -> CategoryStats:
    n = emb.shape[0]
    if n < 2:
        return CategoryStats(0, 0, 0, 0, 0)
    sim = emb @ emb.T
    mask = ~np.eye(n, dtype=bool)
    pairwise = sim[mask]
    redundant = int(np.sum(pairwise > 0.85) // 2)
    return CategoryStats(
        mean_sim=float(pairwise.mean()),
        min_sim=float(pairwise.min()),
        max_sim=float(pairwise.max()),
        std_sim=float(pairwise.std()),
        redundant_pairs=redundant,
    )


def inter_category_sims(cat_embeddings: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Centroid-to-centroid similarities."""
    centroids: dict[str, np.ndarray] = {}
    for cat, emb in cat_embeddings.items():
        c = emb.mean(axis=0)
        centroids[cat] = c / np.linalg.norm(c)

    result: dict[str, dict[str, float]] = {}
    for a in sorted(centroids):
        result[a] = {}
        for b in sorted(centroids):
            result[a][b] = round(float(np.dot(centroids[a], centroids[b])), 4)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model embedding comparison")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--models", nargs="+",
                        choices=["clip", "openclip", "siglip", "t5-large", "t5-xxl"],
                        default=["clip", "openclip", "siglip", "t5-large", "t5-xxl"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    categories = load_components_yaml(args.input)
    total = sum(len(v) for v in categories.values())
    print(f"{'='*70}")
    print(f"MULTI-MODEL EMBEDDING COMPARISON")
    print(f"{'='*70}")
    print(f"  Input: {args.input} ({len(categories)} categories, {total} words)")
    print(f"  Models: {args.models}")

    # Build flat word list preserving category mapping
    all_words: list[str] = []
    word_cats: list[str] = []
    cat_ranges: dict[str, tuple[int, int]] = {}
    for cat, words in sorted(categories.items()):
        start = len(all_words)
        all_words.extend(words)
        word_cats.extend([cat] * len(words))
        cat_ranges[cat] = (start, len(all_words))

    # Run each encoder
    model_results: dict[str, dict[str, Any]] = {}

    for model_name in args.models:
        print(f"\n{'─'*70}")
        print(f"Model: {model_name}")
        print(f"{'─'*70}")

        try:
            if model_name == "clip":
                encoder = CLIPEncoder()
            elif model_name == "openclip":
                encoder = OpenCLIPEncoder()
            elif model_name == "siglip":
                encoder = SigLIPEncoder()
            elif model_name == "t5-large":
                encoder = T5Encoder("google/flan-t5-large")
            elif model_name == "t5-xxl":
                encoder = T5Encoder("google/t5-v1_1-xxl")
            else:
                continue
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        logger.info(f"Encoding {len(all_words)} words ({encoder.dim}d)...")
        emb = encoder.encode(all_words)
        logger.info(f"Done.")

        # Per-category stats
        cat_stats: dict[str, dict[str, Any]] = {}
        print(f"\n  {'Category':<25} {'Mean':>6} {'Min':>6} {'Max':>6} {'Std':>6} {'Redund':>6}")
        print(f"  {'-'*57}")

        for cat, (start, end) in sorted(cat_ranges.items()):
            cat_emb = emb[start:end]
            stats = analyze_category(cat_emb)
            cat_stats[cat] = {
                "mean": round(stats.mean_sim, 4),
                "min": round(stats.min_sim, 4),
                "max": round(stats.max_sim, 4),
                "std": round(stats.std_sim, 4),
                "redundant_pairs": stats.redundant_pairs,
            }
            warn = " !" if stats.redundant_pairs > 0 else ""
            print(
                f"  {cat:<25} {stats.mean_sim:>6.3f} {stats.min_sim:>6.3f} "
                f"{stats.max_sim:>6.3f} {stats.std_sim:>6.3f} {stats.redundant_pairs:>5}{warn}"
            )

        # Inter-category centroid similarities
        cat_emb_dict = {cat: emb[s:e] for cat, (s, e) in cat_ranges.items()}
        inter_sims = inter_category_sims(cat_emb_dict)

        # Summary stats
        all_inter = []
        for a in inter_sims:
            for b in inter_sims[a]:
                if a < b:
                    all_inter.append(inter_sims[a][b])

        mean_inter = np.mean(all_inter)
        min_inter = np.min(all_inter)
        max_inter = np.max(all_inter)

        print(f"\n  Inter-category centroid similarity:")
        print(f"    Mean: {mean_inter:.3f}  Min: {min_inter:.3f}  Max: {max_inter:.3f}")
        print(f"    Range: {max_inter - min_inter:.3f} (higher = better discrimination)")

        model_results[model_name] = {
            "dim": encoder.dim,
            "per_category": cat_stats,
            "inter_category": inter_sims,
            "inter_summary": {
                "mean": round(float(mean_inter), 4),
                "min": round(float(min_inter), 4),
                "max": round(float(max_inter), 4),
                "range": round(float(max_inter - min_inter), 4),
            },
        }

        # Free GPU memory
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── CROSS-MODEL COMPARISON ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    print(f"\n  Mean intra-category similarity (lower = more spread):")
    print(f"  {'Category':<25}", end="")
    for m in args.models:
        if m in model_results:
            print(f" {m:>12}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for m in args.models:
        if m in model_results:
            print(f" {'-'*12}", end="")
    print()

    for cat in sorted(cat_ranges.keys()):
        print(f"  {cat:<25}", end="")
        for m in args.models:
            if m in model_results:
                val = model_results[m]["per_category"][cat]["mean"]
                print(f" {val:>12.3f}", end="")
        print()

    print(f"\n  Inter-category discrimination (centroid similarity range — higher = better):")
    for m in args.models:
        if m in model_results:
            s = model_results[m]["inter_summary"]
            print(f"    {m:<20} range={s['range']:.3f} (mean={s['mean']:.3f}, min={s['min']:.3f}, max={s['max']:.3f})")

    # Save
    if args.output:
        report = {
            "generated_at": datetime.now().isoformat(),
            "input": str(args.input),
            "total_words": total,
            "models": model_results,
        }
        with open(args.output, "w") as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nReport saved: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
