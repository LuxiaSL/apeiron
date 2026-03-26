#!/usr/bin/env python3
"""
Embedding Space Probing — Void Detection, Interpolation, Decoder Probing
═══════════════════════════════════════════════════════════════════════════════

Explores the gaps in the embedding space by:
1. Finding void points (farthest from all existing embeddings)
2. Interpolating between distant word pairs through voids
3. Decoding synthetic embeddings through T5's decoder to discover
   what concepts live in unexplored regions
4. Jittering point clouds around void centers to map semantic neighborhoods

Requires the full T5 model (encoder + decoder), not just the encoder.

Usage:
    # Full pipeline: find voids, decode them, jitter
    uv run python -m promptweaver.tools.probe \
        -i data/selected_final.yaml

    # Just vocabulary mining (fast)
    uv run python -m promptweaver.tools.probe \
        -i data/selected_final.yaml --vocab-only

    # Interpolation probing between specific categories
    uv run python -m promptweaver.tools.probe \
        -i data/selected_final.yaml --interpolate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .embeddings import (
    DualSpaceEmbedder,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

DEFAULT_T5_MODEL = "google/flan-t5-large"


# ═══════════════════════════════════════════════════════════════════════════════
# T5 DECODER — turns embedding-space points back into text
# ═══════════════════════════════════════════════════════════════════════════════


class T5Decoder:
    """
    Uses T5's full encoder-decoder to generate text from synthetic
    encoder hidden states. This lets us "decode" arbitrary points in
    the embedding space into human-readable concepts.
    """

    def __init__(self, model_name: str = DEFAULT_T5_MODEL) -> None:
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading full T5 (encoder+decoder): {model_name} on {self.device}")

        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.model.config.d_model
        self.model.eval()
        logger.info(f"T5 decoder ready — {self.hidden_dim}d hidden states")

    @torch.no_grad()
    def decode_embedding(
        self,
        embedding: np.ndarray,
        max_tokens: int = 15,
        num_beams: int = 5,
        num_return: int = 3,
        temperature: float = 0.8,
    ) -> list[str]:
        """
        Feed a synthetic encoder hidden state through the decoder.

        The embedding is treated as a single-token encoder output.
        The decoder generates text conditioned on this "context."

        Args:
            embedding: (hidden_dim,) array — the point in T5 space to decode
            max_tokens: Maximum output length
            num_beams: Beam search width
            num_return: Number of sequences to return
            temperature: Sampling temperature

        Returns:
            List of decoded text strings
        """
        # Shape: (batch=1, seq_len=1, hidden_dim)
        hidden = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        hidden = hidden.to(self.device)

        # Create a fake attention mask (all ones — attend to our single "token")
        attention_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)

        outputs = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            temperature=temperature,
            do_sample=temperature > 0,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs
        ]

    @torch.no_grad()
    def decode_batch(
        self,
        embeddings: np.ndarray,
        max_tokens: int = 15,
        num_beams: int = 3,
    ) -> list[str]:
        """Decode a batch of embeddings — one output per embedding."""
        results: list[str] = []
        for emb in embeddings:
            texts = self.decode_embedding(emb, max_tokens=max_tokens, num_beams=num_beams, num_return=1)
            results.append(texts[0] if texts else "")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# VOID DETECTION — find the emptiest regions
# ═══════════════════════════════════════════════════════════════════════════════


def find_voids(
    all_embeddings: np.ndarray,
    n_voids: int = 20,
    n_random_probes: int = 5000,
    seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """
    Find points in the embedding space that are maximally far from all
    existing embeddings.

    Strategy: generate random unit vectors, score each by its minimum
    distance to any existing embedding, return the top n_voids.

    Returns list of (void_point, min_distance_to_nearest).
    """
    rng = np.random.RandomState(seed)
    dim = all_embeddings.shape[1]

    # Generate random unit vectors as probe points
    probes = rng.randn(n_random_probes, dim)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)

    # Compute similarity to all existing embeddings
    # (all_embeddings is already normalized)
    sims = probes @ all_embeddings.T  # (n_probes, n_existing)
    max_sims = sims.max(axis=1)  # Nearest existing point per probe

    # Lower max_sim = more isolated = deeper void
    void_indices = np.argsort(max_sims)[:n_voids]

    voids = []
    for idx in void_indices:
        distance = 1.0 - max_sims[idx]  # Convert similarity to distance
        voids.append((probes[idx], float(distance)))

    return voids


def find_voids_iterative(
    all_embeddings: np.ndarray,
    n_voids: int = 20,
    n_probes_per_round: int = 2000,
    n_rounds: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """
    Iterative void finding: each round focuses random probes near the
    best voids from the previous round for higher precision.
    """
    rng = np.random.RandomState(seed)
    dim = all_embeddings.shape[1]

    best_voids: list[tuple[np.ndarray, float]] = []

    for round_num in range(n_rounds):
        if round_num == 0:
            # Random exploration
            probes = rng.randn(n_probes_per_round, dim)
        else:
            # Focus around previous best voids with jitter
            probes = []
            per_void = n_probes_per_round // len(best_voids)
            scale = 0.3 / (round_num + 1)  # Tighter each round
            for void_pt, _ in best_voids:
                jittered = void_pt + rng.randn(per_void, dim) * scale
                probes.append(jittered)
            probes = np.vstack(probes)

        probes /= np.linalg.norm(probes, axis=1, keepdims=True)

        sims = probes @ all_embeddings.T
        max_sims = sims.max(axis=1)

        # Merge with existing best
        for i in range(len(probes)):
            dist = 1.0 - max_sims[i]
            best_voids.append((probes[i], dist))

        # Keep top n_voids
        best_voids.sort(key=lambda x: x[1], reverse=True)
        best_voids = best_voids[:n_voids]

        logger.info(
            f"  Void search round {round_num + 1}: "
            f"best distance={best_voids[0][1]:.4f}, "
            f"worst in top-{n_voids}={best_voids[-1][1]:.4f}"
        )

    return best_voids


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPOLATION — probe between distant words
# ═══════════════════════════════════════════════════════════════════════════════


def find_distant_pairs(
    all_embeddings: np.ndarray,
    words: list[str],
    categories: list[str],
    n_pairs: int = 10,
) -> list[tuple[str, str, str, str, float]]:
    """Find the most distant cross-category word pairs."""
    sims = all_embeddings @ all_embeddings.T

    # Mask same-category pairs
    n = len(words)
    for i in range(n):
        for j in range(n):
            if categories[i] == categories[j]:
                sims[i, j] = 2.0  # Exclude

    pairs: list[tuple[str, str, str, str, float]] = []
    used: set[int] = set()

    for _ in range(n_pairs):
        masked = sims.copy()
        for idx in used:
            masked[idx, :] = 2.0
            masked[:, idx] = 2.0

        min_idx = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = int(min_idx[0]), int(min_idx[1])

        if sims[i, j] >= 2.0:
            break

        pairs.append((words[i], words[j], categories[i], categories[j], float(sims[i, j])))
        used.add(i)
        used.add(j)

    return pairs


def interpolate_and_decode(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    decoder: T5Decoder,
    n_steps: int = 7,
) -> list[tuple[float, list[str]]]:
    """
    Interpolate between two embeddings and decode each step.
    Returns list of (interpolation_t, decoded_texts).
    """
    results: list[tuple[float, list[str]]] = []

    for step in range(n_steps):
        t = step / (n_steps - 1)
        interp = (1 - t) * emb_a + t * emb_b
        interp /= np.linalg.norm(interp)  # Re-normalize to unit sphere

        texts = decoder.decode_embedding(interp, max_tokens=10, num_return=3)
        results.append((t, texts))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# JITTER CLOUD — map semantic neighborhoods
# ═══════════════════════════════════════════════════════════════════════════════


def jitter_and_decode(
    center: np.ndarray,
    decoder: T5Decoder,
    scales: list[float] | None = None,
    n_per_scale: int = 8,
    seed: int = 42,
) -> dict[float, list[str]]:
    """
    Create point clouds around a center, decode each point.
    Different scales reveal the "radius" of the concept.
    """
    if scales is None:
        scales = [0.01, 0.05, 0.1, 0.2, 0.4]

    rng = np.random.RandomState(seed)
    results: dict[float, list[str]] = {}

    for scale in scales:
        jittered = center + rng.randn(n_per_scale, center.shape[0]) * scale
        jittered /= np.linalg.norm(jittered, axis=1, keepdims=True)

        texts = decoder.decode_batch(jittered, max_tokens=10)
        results[scale] = texts

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY MINING — find words in void regions
# ═══════════════════════════════════════════════════════════════════════════════


def mine_vocabulary(
    existing_embeddings: np.ndarray,
    embedder_t5: Any,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Embed the T5 tokenizer's full vocabulary and find tokens that land
    farthest from any existing embedding.

    Returns list of (token_text, min_distance_to_existing).
    """
    tokenizer = embedder_t5.tokenizer
    vocab = tokenizer.get_vocab()

    # Filter to reasonable tokens (skip special, control, fragments)
    candidates: list[str] = []
    for token, idx in vocab.items():
        # Skip special tokens, control tokens, and very short fragments
        if token.startswith("<") or token.startswith("▁") is False and len(token) < 3:
            continue
        # Clean up sentencepiece prefix
        clean = token.replace("▁", " ").strip()
        if len(clean) >= 2 and clean.isascii() and not clean.startswith("<"):
            candidates.append(clean)

    logger.info(f"  Mining {len(candidates)} vocabulary tokens...")

    # Embed in batches
    batch_size = 256
    all_embs: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        embs = embedder_t5.encode_batch(batch)
        all_embs.append(embs)

    vocab_embeddings = np.vstack(all_embs)

    # Find most isolated tokens
    sims = vocab_embeddings @ existing_embeddings.T
    max_sims = sims.max(axis=1)

    distances = 1.0 - max_sims
    top_indices = np.argsort(distances)[-top_k:][::-1]

    results = [(candidates[i], float(distances[i])) for i in top_indices]
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe embedding space voids and decode concepts",
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--t5-model", default=DEFAULT_T5_MODEL)
    parser.add_argument("--n-voids", type=int, default=15,
                        help="Number of void points to find (default: 15)")
    parser.add_argument("--n-interpolations", type=int, default=8,
                        help="Number of distant pairs to interpolate (default: 8)")
    parser.add_argument("--vocab-only", action="store_true",
                        help="Only run vocabulary mining (fast, no decoder)")
    parser.add_argument("--skip-vocab", action="store_true",
                        help="Skip vocabulary mining")
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip decoder probing (just find voids)")
    parser.add_argument("--output", "-o", type=Path,
                        help="Save probe results to YAML")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    categories_data = load_components_yaml(args.input)

    print(f"{'='*60}")
    print("EMBEDDING SPACE PROBING")
    print(f"{'='*60}")

    # Build flat arrays
    all_words: list[str] = []
    all_cats: list[str] = []
    for cat, words in sorted(categories_data.items()):
        all_words.extend(words)
        all_cats.extend([cat] * len(words))

    print(f"  {len(all_words)} words across {len(categories_data)} categories")

    # Load T5 encoder for embeddings
    embedder = DualSpaceEmbedder(clip_model="openai/clip-vit-base-patch32", t5_model=args.t5_model)
    t5_embs = embedder.t5_embedder.encode_batch(all_words)

    probe_results: dict[str, Any] = {}

    # ── VOCABULARY MINING ───────────────────────────────────────────────
    if not args.skip_vocab:
        print(f"\n{'─'*60}")
        print("VOCABULARY MINING")
        print(f"{'─'*60}")
        vocab_results = mine_vocabulary(t5_embs, embedder.t5_embedder, top_k=50)
        print(f"\nTop 50 most isolated vocabulary tokens:")
        print(f"{'Token':<30} {'Distance':>8}")
        print("-" * 40)
        for token, dist in vocab_results[:50]:
            print(f"  {token:<28} {dist:>8.4f}")

        probe_results["vocabulary_mining"] = [
            {"token": t, "distance": round(d, 4)} for t, d in vocab_results
        ]

    if args.vocab_only:
        if args.output:
            with open(args.output, "w") as f:
                yaml.dump(probe_results, f, default_flow_style=False, width=120)
            print(f"\nSaved to {args.output}")
        print("\nDone!")
        return

    # ── VOID DETECTION ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("VOID DETECTION")
    print(f"{'─'*60}")
    voids = find_voids_iterative(t5_embs, n_voids=args.n_voids, n_probes_per_round=3000)

    print(f"\nTop {args.n_voids} void points (most isolated regions):")
    print(f"{'#':<4} {'Distance':>8} {'Nearest Word':<30} {'Category':<20}")
    print("-" * 65)

    for i, (void_pt, dist) in enumerate(voids):
        # Find nearest existing word
        sims = void_pt @ t5_embs.T
        nearest_idx = int(np.argmax(sims))
        print(
            f"  {i+1:<3} {dist:>8.4f} "
            f"{all_words[nearest_idx]:<30} {all_cats[nearest_idx]:<20}"
        )

    probe_results["voids"] = [
        {"distance": round(d, 4), "nearest": all_words[int(np.argmax(v @ t5_embs.T))]}
        for v, d in voids
    ]

    # ── DECODER PROBING ─────────────────────────────────────────────────
    if not args.skip_decode:
        print(f"\n{'─'*60}")
        print("DECODER PROBING — decoding void points")
        print(f"{'─'*60}")

        decoder = T5Decoder(args.t5_model)

        # Decode void points
        print("\nVoid point decodings:")
        void_decodings: list[dict[str, Any]] = []
        for i, (void_pt, dist) in enumerate(voids[:10]):  # Top 10
            texts = decoder.decode_embedding(void_pt, max_tokens=12, num_return=5)
            nearest_idx = int(np.argmax(void_pt @ t5_embs.T))
            print(f"\n  Void #{i+1} (dist={dist:.4f}, nearest='{all_words[nearest_idx]}'):")
            for t in texts:
                print(f"    → {t}")

            void_decodings.append({
                "distance": round(dist, 4),
                "nearest": all_words[nearest_idx],
                "decoded": texts,
            })

        probe_results["void_decodings"] = void_decodings

        # ── JITTER CLOUDS ───────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("JITTER CLOUDS — mapping void neighborhoods")
        print(f"{'─'*60}")

        jitter_results: list[dict[str, Any]] = []
        for i, (void_pt, dist) in enumerate(voids[:5]):  # Top 5
            jitter_map = jitter_and_decode(void_pt, decoder, n_per_scale=6)
            nearest_idx = int(np.argmax(void_pt @ t5_embs.T))

            print(f"\n  Void #{i+1} (nearest='{all_words[nearest_idx]}'):")
            jitter_entry: dict[str, Any] = {"nearest": all_words[nearest_idx], "scales": {}}
            for scale, texts in sorted(jitter_map.items()):
                unique_texts = list(dict.fromkeys(texts))  # Dedupe preserving order
                print(f"    scale={scale}: {unique_texts[:4]}")
                jitter_entry["scales"][scale] = unique_texts

            jitter_results.append(jitter_entry)

        probe_results["jitter_clouds"] = jitter_results

        # ── INTERPOLATION PROBING ───────────────────────────────────────
        print(f"\n{'─'*60}")
        print("INTERPOLATION PROBING — decoding between distant word pairs")
        print(f"{'─'*60}")

        pairs = find_distant_pairs(t5_embs, all_words, all_cats, n_pairs=args.n_interpolations)

        interp_results: list[dict[str, Any]] = []
        for word_a, word_b, cat_a, cat_b, sim in pairs:
            idx_a = all_words.index(word_a)
            idx_b = all_words.index(word_b)

            interp = interpolate_and_decode(t5_embs[idx_a], t5_embs[idx_b], decoder, n_steps=7)

            print(f"\n  '{word_a}' ({cat_a}) ←→ '{word_b}' ({cat_b})  [sim={sim:.3f}]")
            entry: dict[str, Any] = {
                "word_a": word_a, "cat_a": cat_a,
                "word_b": word_b, "cat_b": cat_b,
                "similarity": round(sim, 4),
                "steps": [],
            }
            for t, texts in interp:
                label = f"t={t:.2f}"
                print(f"    {label}: {texts[0]}")
                entry["steps"].append({"t": round(t, 2), "decoded": texts})

            interp_results.append(entry)

        probe_results["interpolations"] = interp_results

    # ── SAVE ────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump(probe_results, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nResults saved to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
