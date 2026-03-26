#!/usr/bin/env python3
"""
Multi-Model Component Generation
═══════════════════════════════════════════════════════════════════════════════

Generates massive candidate pools for prompt component categories using
multiple LLM families for maximum novelty and coverage.

Models:
    - Claude Sonnet (Anthropic) — creative, good taste, prompt caching
    - Kimi K2 (OpenRouter/Moonshot) — different training distribution, cheap
    - DeepSeek (OpenRouter) — interesting generative character, very cheap
    - GPT (OpenAI) — strong at lists, taxonomically complete

Strategy:
    Round-robin across model families so each adds from a different perspective.
    The cumulative "avoid" list grows each iteration, pushing models toward
    increasingly novel outputs.

Usage:
    # Set API keys (use whichever you have)
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENROUTER_API_KEY="sk-or-..."
    export OPENAI_API_KEY="sk-..."

    # Generate for all categories
    uv run python -m promptweaver.tools.generate

    # Specific categories, specific models
    uv run python -m promptweaver.tools.generate \
        --categories color_logic medium_render \
        --models claude kimi deepseek

    # Dry run (show prompts)
    uv run python -m promptweaver.tools.generate --dry-run

    # Resume from checkpoint
    uv run python -m promptweaver.tools.generate --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiohttp
import yaml

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# API endpoints
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Model IDs
ModelName = Literal["claude", "kimi", "deepseek", "gpt"]

MODEL_IDS: dict[ModelName, str] = {
    "claude": "claude-sonnet-4-6",
    "kimi": "moonshotai/kimi-k2-0905",
    "deepseek": "deepseek/deepseek-v3.2",
    "gpt": "gpt-5.4",
}

# Which API each model uses
MODEL_PROVIDERS: dict[ModelName, str] = {
    "claude": "anthropic",
    "kimi": "openrouter",
    "deepseek": "openrouter",
    "gpt": "openai",
}

# Per-million-token pricing for cost tracking (fallback — OpenRouter-reported cost preferred).
# Sources: OpenAI pricing page, OpenRouter model pages (2026-03)
MODEL_PRICING: dict[ModelName, dict[str, float]] = {
    "claude": {"input": 3.00, "cache_write": 3.75, "cache_read": 0.30, "output": 15.00},
    "kimi": {"input": 0.40, "output": 2.00},       # moonshotai/kimi-k2-0905
    "deepseek": {"input": 0.26, "output": 0.38},    # deepseek/deepseek-v3.2
    "gpt": {"input": 2.50, "cached_input": 0.25, "output": 15.00},  # gpt-5.4 short context
}

# Default cost caps per model (USD)
DEFAULT_COST_CAPS: dict[ModelName, float] = {
    "claude": 40.00,
    "kimi": 20.00,
    "deepseek": 20.00,
    "gpt": 15.00,
}

# Generation settings
BATCH_SIZE = 25  # Components per request
TARGET_PER_CATEGORY = 300  # Generate big, select later
MAX_PARALLEL_REQUESTS = 8
REQUEST_DELAY_MS = 80
CHECKPOINT_INTERVAL = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryConfig:
    """Configuration for generating a single category's components."""

    name: str
    description: str
    guidance: str
    anti_examples: list[str]
    seeds: list[str] = field(default_factory=list)
    sub_clusters: list[str] = field(default_factory=list)
    target: int = TARGET_PER_CATEGORY


# These are baked in — derived from the existing components.yaml structure
# and the visual-axis design philosophy.
CATEGORY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "subject_form": {
        "description": "Primary visual entity — archetypal forms and abstract shapes. "
        "Single abstract nouns describing a shape, structure, or organic form.",
        "guidance": "Think sculptural, archetypal, morphological. These are WHAT the image "
        "depicts at its most fundamental: shapes, volumes, structures. Favor uncommon "
        "but visually evocative forms. 1-2 words max. Architectural, biological, "
        "geometric, and mythic forms all welcome.",
        "anti_examples": [
            "tree (too specific/literal)",
            "person (character, not form)",
            "cat (animal, not form)",
            "building (too generic)",
            "beautiful shape (adjective + vague)",
        ],
        "sub_clusters": [
            "geometric/platonic (polyhedra, prisms, tori)",
            "organic/biological (pods, tendrils, coral forms)",
            "architectural (buttresses, arches, domes)",
            "abstract/conceptual (void, threshold, membrane)",
            "geological (formations, strata, crystals)",
        ],
    },
    "material_substance": {
        "description": "Physical materials and substances something could be made of. "
        "Concrete, tangible materials — not textures, colors, or qualities.",
        "guidance": "Real-world materials that have a distinct visual character. Include "
        "natural, synthetic, organic, mineral. The material should be visually "
        "recognizable in an image. 1-3 words max.",
        "anti_examples": [
            "smooth (texture, not material)",
            "red (color)",
            "glowing (light behavior)",
            "expensive (quality judgment)",
            "rough surface (texture description)",
        ],
        "sub_clusters": [
            "organic (chitin, mycelium, bone, amber, keratin)",
            "mineral/geological (obsidian, mica, pumice, slate)",
            "metals/alloys (patinated bronze, anodized titanium)",
            "synthetic/industrial (aerogel, kevlar, resin)",
            "natural fibers (silk, linen, birch bark, papyrus)",
            "unusual/exotic (ferrofluid, oobleck, gallium)",
        ],
    },
    "texture_density": {
        "description": "Surface quality described through tactile metaphor. How something "
        "FEELS to look at — density, pattern, roughness, weave.",
        "guidance": "Think about what you'd feel if you touched the surface. Combine a "
        "texture quality with a visual reference. Often 2-word phrases. Must be "
        "distinct from material_substance (describe the surface quality, not what "
        "it's made of).",
        "anti_examples": [
            "blue (color, not texture)",
            "glass (material, not texture)",
            "beautiful (quality judgment)",
            "old (temporal state)",
        ],
        "sub_clusters": [
            "dense/intricate (filigree, lacework, chainmail)",
            "smooth/minimal (gradient, polished, mirror)",
            "organic/natural (bark ridges, coral pitting, honeycomb)",
            "geometric/regular (waffled, tessellated, herringbone)",
            "rough/chaotic (crackled, pitted, eroded)",
            "textile/woven (braided, quilted, knitted)",
        ],
    },
    "light_behavior": {
        "description": "How light interacts with the scene — specific lighting phenomena, "
        "techniques, and behaviors. NOT brightness/darkness but the character of light.",
        "guidance": "Specific, named lighting phenomena or techniques. Think photography "
        "lighting setups, physics of light, natural light events. Should describe "
        "HOW light behaves, not just whether it exists.",
        "anti_examples": [
            "bright (vague intensity)",
            "dark (absence, not behavior)",
            "sunset (setting/time, not light behavior)",
            "glowing (too generic)",
        ],
        "sub_clusters": [
            "natural phenomena (crepuscular rays, caustics, aurora)",
            "studio/photography (rembrandt, butterfly, barn door)",
            "physics/optics (subsurface scattering, rayleigh, refraction)",
            "atmospheric (volumetric, god rays, light pillars)",
            "artificial (neon glow, LED strip, blacklight)",
            "artistic (chiaroscuro, tenebrism, sfumato)",
        ],
    },
    "color_logic": {
        "description": "Palette relationships and color theory — NOT single colors but "
        "relationships, harmonies, and chromatic strategies.",
        "guidance": "Describe color as a SYSTEM: palettes, relationships, contrasts, "
        "harmonies. Reference specific named colors or combinations. Think color "
        "theory (complementary, analogous, triadic) combined with specific hues. "
        "Cultural and material color references work well (patina green, raw umber).",
        "anti_examples": [
            "blue (single color, not a relationship)",
            "red (single color)",
            "colorful (vague)",
            "bright colors (vague intensity)",
        ],
        "sub_clusters": [
            "monochromatic (tonal variations of a single hue)",
            "complementary/contrast (opposing hues, high tension)",
            "analogous/harmonious (adjacent hues, low tension)",
            "accent-driven (neutral field with single color punctuation)",
            "cultural/material (tobacco-stained, oxidized copper tones)",
            "temperature-based (warm/cool contrast, thermal palette)",
        ],
    },
    "atmosphere_field": {
        "description": "Environmental media between viewer and subject — particles, fog, "
        "fields, suspensions. What fills the AIR in the scene.",
        "guidance": "Think about what's floating, suspended, drifting in the space. "
        "Particles, vapors, fields, biological or physical phenomena that create "
        "atmosphere. NOT the setting itself, but what fills the space within it.",
        "anti_examples": [
            "sky (setting, not atmosphere)",
            "background (composition term)",
            "blue haze (color + vague)",
            "scary atmosphere (mood, not physical)",
        ],
        "sub_clusters": [
            "particles (dust motes, pollen, ash, snow)",
            "vapor/gas (fog, steam, smoke, breath frost)",
            "liquid/aquatic (underwater caustics, bubbles, spray mist)",
            "biological (spores, firefly swarms, pheromone clouds)",
            "thermal/energy (heat shimmer, plasma, static discharge)",
            "crystalline/frozen (frost suspension, ice fog, diamond dust)",
        ],
    },
    "phenomenon_pattern": {
        "description": "Visual processes and patterns — organic, digital, or mathematical. "
        "Things that are HAPPENING or FORMING, not static textures.",
        "guidance": "Dynamic visual processes: growth, decay, interference, fractals, "
        "biological patterns, digital artifacts. These imply MOVEMENT or PROCESS "
        "even in a still image. Distinct from texture_density (patterns are processes, "
        "textures are surfaces).",
        "anti_examples": [
            "smooth (texture, not pattern)",
            "glowing (light, not pattern)",
            "blue lines (color + shape, not a phenomenon)",
            "rough (texture quality)",
        ],
        "sub_clusters": [
            "biological growth (branching, budding, mitosis, mycelial spread)",
            "decay/dissolution (rust bloom, erosion channels, deliquescence)",
            "mathematical/fractal (mandelbrot, voronoi, fibonacci spiral)",
            "digital/glitch (pixel sort, data moshing, scan lines)",
            "optical/wave (moiré, interference, chromatic aberration)",
            "crystalline/mineral (dendrite growth, crystal nucleation)",
        ],
    },
    "spatial_logic": {
        "description": "Compositional arrangement and geometry — how elements are organized "
        "in the frame. The spatial GRAMMAR of the image.",
        "guidance": "Describe arrangements, compositions, geometric relationships. Think "
        "about how a designer or architect would describe spatial organization. "
        "These control the layout and flow of the image, not what's in it.",
        "anti_examples": [
            "big (scale, not arrangement)",
            "left (too simple/directional)",
            "landscape (setting/format, not spatial logic)",
            "messy (quality judgment)",
        ],
        "sub_clusters": [
            "symmetry (bilateral, radial, rotational, approximate)",
            "progression/sequence (cascade, gradient, terrace)",
            "tension/dynamic (diagonal, oblique, cantilevered)",
            "repetition (tessellated, tiled, kaleidoscopic)",
            "flow/organic (spiral, meander, branching)",
            "negative space (isolation, void-dominant, sparse field)",
        ],
    },
    "scale_perspective": {
        "description": "Viewing position and magnification — WHERE you're looking from "
        "and HOW CLOSE. The camera's relationship to the subject.",
        "guidance": "Think camera positions, magnification levels, specific viewpoints. "
        "Each should imply a very different image. Include both literal perspectives "
        "(aerial, macro) and evocative viewing situations (through a keyhole, from "
        "inside a bell jar).",
        "anti_examples": [
            "big (relative size, not perspective)",
            "small (relative size)",
            "far away (vague distance)",
            "normal view (non-specific)",
        ],
        "sub_clusters": [
            "magnification (macro, electron microscope, jeweler's loupe)",
            "aerial/remote (satellite, drone, astronomical)",
            "immersive/interior (inside a vessel, cave mouth, from below)",
            "architectural (cathedral ceiling, corridor vanishing point)",
            "scientific instrument (arthroscope, spectrometer, CT scan)",
            "framing device (porthole, keyhole, through curtains)",
        ],
    },
    "temporal_state": {
        "description": "Moment in a process — decay, growth, transformation, preservation. "
        "WHERE in its lifecycle something currently exists.",
        "guidance": "Frozen moments in processes: something is actively decaying, growing, "
        "crystallizing, being consumed. Should imply a before and after — the viewer "
        "can sense the trajectory. NOT times of day or historical periods.",
        "anti_examples": [
            "old (vague age, not a process state)",
            "new (vague age)",
            "morning (time of day, not temporal state)",
            "medieval (historical period, not process)",
        ],
        "sub_clusters": [
            "growth/emergence (nascent, sprouting, crystallizing, budding)",
            "decay/entropy (corroding, crumbling, deliquescing, fraying)",
            "preservation/stasis (fossilized, resin-entombed, freeze-dried)",
            "transformation (metamorphosing, alloying, fermenting)",
            "consumption (half-eaten, eroded, wind-scoured, tide-worn)",
            "cyclical (molting, shedding, regenerating, re-flowering)",
        ],
    },
    "setting_location": {
        "description": "Environmental context — specific places, spaces, and environments. "
        "WHERE the scene takes place.",
        "guidance": "Specific, visually distinctive places. Not moods or qualities but "
        "actual environments you could photograph. Favor unusual, atmospheric "
        "locations with strong visual character. 2-4 words to be specific enough.",
        "anti_examples": [
            "scary place (mood, not location)",
            "underground (too vague)",
            "beautiful landscape (quality + vague)",
            "room (non-specific)",
        ],
        "sub_clusters": [
            "natural/geological (volcanic crater, salt cave, cenote)",
            "built/architectural (brutalist parking garage, art deco lobby)",
            "overgrown/reclaimed (vine-covered factory, tree-root temple)",
            "water/aquatic (kelp forest, tidal pool, flooded mine)",
            "atmospheric/extreme (cloud inversion, geothermal vent field)",
            "liminal/transitional (elevator shaft, storm drain, loading dock)",
        ],
    },
    "medium_render": {
        "description": "Artistic technique or imaging method — HOW the image appears to "
        "have been MADE. The rendering style or capture technology.",
        "guidance": "Specific named techniques from art, photography, printmaking, "
        "scientific imaging. Each should produce a fundamentally different visual "
        "character. Include historical processes, scientific instruments, "
        "craft techniques.",
        "anti_examples": [
            "pretty (quality judgment)",
            "realistic (too vague)",
            "3d (too broad)",
            "artistic (non-specific style)",
        ],
        "sub_clusters": [
            "photographic (daguerreotype, wet plate, infrared, polaroid)",
            "printmaking (cyanotype, risograph, woodblock, screen print)",
            "painting (impasto, fresco, encaustic, gouache)",
            "drawing (charcoal, silverpoint, conte crayon, graphite)",
            "scientific imaging (electron microscope, MRI, spectrograph)",
            "craft/dimensional (paper cut, ceramic glaze, woven textile, glass blown)",
        ],
    },
}


def build_category_configs(components_path: Path, target: int) -> list[CategoryConfig]:
    """Load seed words from components.yaml and merge with category definitions."""
    with open(components_path) as f:
        data = yaml.safe_load(f)

    configs: list[CategoryConfig] = []
    components = data.get("components", {})

    for cat_name, definition in CATEGORY_DEFINITIONS.items():
        items = components.get(cat_name, [])
        if items and isinstance(items[0], dict):
            seeds = [item["word"] for item in items]
        elif items:
            seeds = list(items)
        else:
            seeds = []

        configs.append(
            CategoryConfig(
                name=cat_name,
                description=definition["description"],
                guidance=definition["guidance"],
                anti_examples=definition["anti_examples"],
                seeds=seeds,
                sub_clusters=definition.get("sub_clusters", []),
                target=target,
            )
        )

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationResult:
    """Result from a single API call."""

    category: str
    model: ModelName
    components: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


@dataclass
class ModelStats:
    """Per-model tracking: requests, tokens, cost."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


@dataclass
class GenerationState:
    """Tracks progress across all categories and models."""

    components: dict[str, set[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    model_stats: dict[str, ModelStats] = field(default_factory=lambda: {
        "claude": ModelStats(), "kimi": ModelStats(),
        "deepseek": ModelStats(), "gpt": ModelStats(),
    })

    # Legacy compat for checkpoint loading
    costs: dict[str, float] = field(default_factory=lambda: {
        "claude": 0.0, "kimi": 0.0, "deepseek": 0.0, "gpt": 0.0,
    })

    def add_components(self, category: str, new: list[str]) -> int:
        if category not in self.components:
            self.components[category] = set()
        before = len(self.components[category])
        self.components[category].update(new)
        return len(self.components[category]) - before

    def record_result(self, result: "GenerationResult") -> None:
        """Record a generation result into per-model stats."""
        ms = self.model_stats.setdefault(result.model, ModelStats())
        ms.requests += 1
        ms.input_tokens += result.input_tokens
        ms.output_tokens += result.output_tokens
        ms.cached_tokens += result.cached_tokens
        ms.cost += result.cost
        # Keep costs dict in sync for checkpoint compat
        self.costs[result.model] = ms.cost

    @property
    def total_requests(self) -> int:
        return sum(ms.requests for ms in self.model_stats.values())

    @property
    def total_tokens(self) -> int:
        return sum(ms.input_tokens + ms.output_tokens for ms in self.model_stats.values())

    @property
    def total_cost(self) -> float:
        return sum(ms.cost for ms in self.model_stats.values())


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are expanding a vocabulary pool for a combinatorial image generation prompt system targeting Midjourney.

The system uses independent visual-axis categories. Each category controls one dimension of the final image. Terms must be:
1. Visually distinct from each other when rendered (different visual outcomes)
2. Specific and concrete — vague/abstract terms get ignored by image models
3. Strictly within the category scope (cross-category contamination ruins the system)
4. 1-4 words per term (shorter is better — every word must earn its place)

The generated images use these terms in structured prompts with ~256 token budget, so each component must carry maximum visual signal per word.

Output ONLY the new terms, one per line. No numbering, no explanations, no quotes."""


def make_category_context(config: CategoryConfig) -> str:
    """Static category context (cacheable)."""
    parts = [
        f"Category: {config.name.upper().replace('_', ' ')}",
        f"\nDescription: {config.description}",
        f"\nGeneration Guidance: {config.guidance}",
        "\nDO NOT generate terms like these:",
    ]
    for ex in config.anti_examples:
        parts.append(f"  - {ex}")

    parts.append("\nSeed examples (generate NEW ones distinct from these):")
    for seed in config.seeds[:20]:  # Cap seed display
        parts.append(f"  - {seed}")
    if len(config.seeds) > 20:
        parts.append(f"  ... ({len(config.seeds)} total)")

    if config.sub_clusters:
        parts.append("\nSub-clusters (generate EQUAL numbers from each):")
        for sc in config.sub_clusters:
            parts.append(f"  - {sc}")
        parts.append("IMPORTANT: Balance across ALL sub-clusters.")

    return "\n".join(parts)


def make_user_message(existing: set[str], batch_size: int = BATCH_SIZE) -> str:
    """Dynamic user message with avoid list (grows each iteration)."""
    if existing:
        sample_size = min(60, len(existing))
        sample = random.sample(list(existing), sample_size)
        avoid = ", ".join(sample)
        if len(existing) > sample_size:
            avoid += f" ... and {len(existing) - sample_size} others"
        return (
            f"AVOID these existing terms: {avoid}\n\n"
            f"Generate {batch_size} new unique terms. One per line, nothing else."
        )
    return f"Generate {batch_size} new unique terms. One per line, nothing else."


def parse_component_lines(content: str) -> list[str]:
    """Parse raw LLM output into clean component list."""
    components: list[str] = []
    for line in content.strip().split("\n"):
        cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", line.strip())
        cleaned = cleaned.strip("\"'`")
        cleaned = cleaned.strip()
        if cleaned and 1 < len(cleaned) < 60 and len(cleaned.split()) <= 5:
            components.append(cleaned.lower())
    return components


# ═══════════════════════════════════════════════════════════════════════════════
# API CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════


async def call_anthropic(
    session: aiohttp.ClientSession,
    category_context: str,
    user_message: str,
    api_key: str,
) -> tuple[list[str], dict[str, int]]:
    """Call Anthropic API with prompt caching for Claude."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    full_system = f"{SYSTEM_PROMPT}\n\n---\n\n{category_context}"

    payload = {
        "model": MODEL_IDS["claude"],
        "max_tokens": 1500,
        "temperature": 0.95,
        "system": [
            {
                "type": "text",
                "text": full_system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message}],
    }

    async with session.post(ANTHROPIC_API_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            error = await resp.text()
            raise RuntimeError(f"Anthropic {resp.status}: {error[:200]}")
        data = await resp.json()

    content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content = block.get("text", "")
            break

    usage = data.get("usage", {})
    tokens = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
    }

    return parse_component_lines(content), tokens


async def call_openrouter(
    session: aiohttp.ClientSession,
    model: ModelName,
    category_context: str,
    user_message: str,
    api_key: str,
) -> tuple[list[str], dict[str, int]]:
    """Call OpenRouter for Kimi K2 or DeepSeek."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/LuxiaSL/promptweaver",
        "X-Title": "PromptWeaver Pool Expansion",
    }

    full_system = f"{SYSTEM_PROMPT}\n\n---\n\n{category_context}"

    payload: dict[str, Any] = {
        "model": MODEL_IDS[model],
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.95,
        "max_tokens": 1500,
    }

    # Prefer DeepInfra for Kimi (lowest latency per OpenRouter, fp4 quant)
    # and DeepInfra for DeepSeek too (lowest latency, fp4)
    if model == "kimi":
        payload["provider"] = {
            "order": ["DeepInfra"],
            "allow_fallbacks": True,
        }
    elif model == "deepseek":
        payload["provider"] = {
            "order": ["DeepInfra"],
            "allow_fallbacks": True,
        }

    async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            error = await resp.text()
            raise RuntimeError(f"OpenRouter/{model} {resp.status}: {error[:200]}")
        data = await resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    tokens = {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }

    # OpenRouter sometimes reports actual cost directly — more accurate than our estimate
    or_cost = data.get("usage", {}).get("cost")
    if or_cost is not None:
        tokens["_reported_cost"] = float(or_cost)

    return parse_component_lines(content), tokens


async def call_openai(
    session: aiohttp.ClientSession,
    category_context: str,
    user_message: str,
    api_key: str,
) -> tuple[list[str], dict[str, int]]:
    """Call OpenAI API directly for GPT models."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    full_system = f"{SYSTEM_PROMPT}\n\n---\n\n{category_context}"

    payload = {
        "model": MODEL_IDS["gpt"],
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.95,
        "max_completion_tokens": 1500,
    }

    async with session.post(OPENAI_API_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            error = await resp.text()
            raise RuntimeError(f"OpenAI {resp.status}: {error[:200]}")
        data = await resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    # OpenAI reports cached tokens in prompt_tokens_details
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    tokens = {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "cached": cached,
    }

    return parse_component_lines(content), tokens


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_cost(model: ModelName, tokens: dict[str, Any]) -> float:
    """
    Estimate cost from token counts.

    For OpenRouter models, prefers the API-reported cost (_reported_cost) when
    available — this accounts for provider-specific pricing and caching discounts
    that our static table can't capture.
    """
    # OpenRouter reports actual cost — use it when present
    reported = tokens.get("_reported_cost")
    if reported is not None:
        return float(reported)

    pricing = MODEL_PRICING[model]
    cost = 0.0

    if model == "claude":
        # Anthropic's breakdown: uncached input, cache write, cache read, output
        cost += tokens.get("input", 0) * pricing["input"] / 1_000_000
        cost += tokens.get("cache_write", 0) * pricing["cache_write"] / 1_000_000
        cost += tokens.get("cache_read", 0) * pricing["cache_read"] / 1_000_000
        cost += tokens.get("output", 0) * pricing["output"] / 1_000_000
    elif model == "gpt" and "cached_input" in pricing:
        # GPT has separate cached input pricing
        cached = tokens.get("cached", 0)
        uncached = tokens.get("input", 0) - cached
        cost += max(uncached, 0) * pricing["input"] / 1_000_000
        cost += cached * pricing["cached_input"] / 1_000_000
        cost += tokens.get("output", 0) * pricing["output"] / 1_000_000
    else:
        cost += tokens.get("input", 0) * pricing["input"] / 1_000_000
        cost += tokens.get("output", 0) * pricing["output"] / 1_000_000

    return cost


MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # Seconds, doubles each retry


async def generate_batch(
    session: aiohttp.ClientSession,
    config: CategoryConfig,
    model: ModelName,
    existing: set[str],
    api_keys: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> GenerationResult | None:
    """Generate one batch for one category with one model, with retry on rate limits."""
    async with semaphore:
        ctx = make_category_context(config)
        msg = make_user_message(existing)
        provider = MODEL_PROVIDERS[model]

        for attempt in range(MAX_RETRIES):
            try:
                if provider == "anthropic":
                    components, tokens = await call_anthropic(
                        session, ctx, msg, api_keys["anthropic"]
                    )
                elif provider == "openai":
                    components, tokens = await call_openai(
                        session, ctx, msg, api_keys["openai"]
                    )
                else:
                    components, tokens = await call_openrouter(
                        session, model, ctx, msg, api_keys["openrouter"]
                    )

                # Filter already-seen
                new_components = [c for c in components if c not in existing]
                cost = estimate_cost(model, tokens)

                return GenerationResult(
                    category=config.name,
                    model=model,
                    components=new_components,
                    input_tokens=tokens.get("input", 0) + tokens.get("cache_read", 0) + tokens.get("cache_write", 0),
                    output_tokens=tokens.get("output", 0),
                    cached_tokens=tokens.get("cache_read", 0) + tokens.get("cached", 0),
                    cost=cost,
                )

            except RuntimeError as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()
                is_overloaded = "529" in error_str or "503" in error_str or "overloaded" in error_str.lower()

                if (is_rate_limit or is_overloaded) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    # Check for Retry-After hint in error message
                    retry_after = _parse_retry_after(error_str)
                    if retry_after and retry_after > delay:
                        delay = retry_after
                    logger.warning(
                        f"  [{model}] {config.name}: rate limited, "
                        f"retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Error [{model}] {config.name}: {e}")
                    return None

            except Exception as e:
                logger.error(f"Error [{model}] {config.name}: {e}")
                return None

        return None


def _parse_retry_after(error_text: str) -> float | None:
    """Try to extract a retry-after value from an error message."""
    import re as _re
    # Look for patterns like "retry after 5s", "Retry-After: 10", "try again in 30 seconds"
    match = _re.search(r'retry.?after[:\s]*(\d+)', error_text, _re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = _re.search(r'try again in (\d+)', error_text, _re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def save_checkpoint(state: GenerationState, path: Path) -> None:
    """Save generation progress for resume."""
    model_stats_data = {}
    for model, ms in state.model_stats.items():
        if ms.requests > 0:
            model_stats_data[model] = {
                "requests": ms.requests,
                "input_tokens": ms.input_tokens,
                "output_tokens": ms.output_tokens,
                "cached_tokens": ms.cached_tokens,
                "cost": round(ms.cost, 4),
            }

    data = {
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_requests": state.total_requests,
            "total_tokens": state.total_tokens,
            "total_cost": round(state.total_cost, 4),
            "errors": len(state.errors),
            "by_model": model_stats_data,
        },
        "components": {
            cat: sorted(words) for cat, words in state.components.items()
        },
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, width=120)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path) -> GenerationState | None:
    """Resume from checkpoint."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        state = GenerationState()
        state.components = {
            cat: set(words) for cat, words in data.get("components", {}).items()
        }

        # Restore per-model stats (new format)
        by_model = data.get("stats", {}).get("by_model", {})
        for model, ms_data in by_model.items():
            state.model_stats[model] = ModelStats(
                requests=ms_data.get("requests", 0),
                input_tokens=ms_data.get("input_tokens", 0),
                output_tokens=ms_data.get("output_tokens", 0),
                cached_tokens=ms_data.get("cached_tokens", 0),
                cost=ms_data.get("cost", 0.0),
            )
            state.costs[model] = ms_data.get("cost", 0.0)

        # Fallback: old checkpoint format with just costs dict
        if not by_model:
            saved_costs = data.get("stats", {}).get("costs", {})
            for model, cost in saved_costs.items():
                state.costs[model] = cost
                state.model_stats[model] = ModelStats(cost=cost)

        total = sum(len(c) for c in state.components.values())
        logger.info(f"Resumed from checkpoint: {total} existing components")
        for model, ms in state.model_stats.items():
            if ms.requests > 0 or ms.cost > 0:
                logger.info(f"  {model}: {ms.requests} reqs, ${ms.cost:.2f}")
        return state
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


async def run_generation(
    categories: list[CategoryConfig],
    models: list[ModelName],
    api_keys: dict[str, str],
    cost_caps: dict[ModelName, float],
    dry_run: bool = False,
    checkpoint_path: Path | None = None,
) -> GenerationState:
    """
    Main generation loop. Round-robins across models, generating for all
    incomplete categories each iteration.
    """
    # Try to resume
    state: GenerationState | None = None
    if checkpoint_path:
        state = load_checkpoint(checkpoint_path)

    if state is None:
        state = GenerationState()
        for config in categories:
            state.components[config.name] = set(config.seeds)

    if dry_run:
        logger.info("=== DRY RUN ===")
        for config in categories:
            ctx = make_category_context(config)
            msg = make_user_message(state.components.get(config.name, set()))
            print(f"\n{'='*60}")
            print(f"Category: {config.name}")
            print(f"{'='*60}")
            print(f"\nSystem ({len(SYSTEM_PROMPT)} chars, cached):\n{SYSTEM_PROMPT[:200]}...")
            print(f"\nCategory Context ({len(ctx)} chars, cached):\n{ctx[:300]}...")
            print(f"\nUser Message:\n{msg}")
        return state

    # Track which models are still active (removed when cost-capped)
    active_models = list(models)
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        model_idx = 0
        max_iterations = 200

        for iteration in range(1, max_iterations + 1):
            if not active_models:
                logger.warning("All models hit cost caps!")
                break

            current_model = active_models[model_idx % len(active_models)]

            # Check cost cap
            if state.costs.get(current_model, 0) >= cost_caps.get(current_model, float("inf")):
                logger.warning(
                    f"Cost cap reached for {current_model}: "
                    f"${state.costs[current_model]:.2f} >= ${cost_caps[current_model]:.2f}"
                )
                active_models.remove(current_model)
                continue

            # Find incomplete categories
            incomplete = [
                c for c in categories
                if len(state.components.get(c.name, set())) < c.target
            ]
            if not incomplete:
                logger.info("All categories complete!")
                break

            logger.info(
                f"\n{'='*60}\n"
                f"ITERATION {iteration} — {current_model.upper()} "
                f"({len(incomplete)} categories remaining)\n"
                f"{'='*60}"
            )

            for config in incomplete:
                count = len(state.components.get(config.name, set()))
                logger.info(f"  {config.name}: {count}/{config.target}")

            # Launch parallel batches for all incomplete categories
            tasks = [
                generate_batch(
                    session=session,
                    config=config,
                    model=current_model,
                    existing=state.components.get(config.name, set()),
                    api_keys=api_keys,
                    semaphore=semaphore,
                )
                for config in incomplete
            ]

            results = await asyncio.gather(*tasks)

            # Process results
            batch_added = 0
            batch_cost = 0.0

            for result in results:
                if result is None:
                    state.errors.append(f"Failed {current_model}")
                    continue

                added = state.add_components(result.category, result.components)
                state.record_result(result)
                batch_added += added
                batch_cost += result.cost

                if added > 0:
                    logger.info(
                        f"  [{result.model}] {result.category}: "
                        f"+{added} new (${result.cost:.4f})"
                    )

            ms = state.model_stats.get(current_model, ModelStats())
            logger.info(
                f"  Batch: +{batch_added} components, ${batch_cost:.4f} "
                f"({current_model} total: ${ms.cost:.2f}/{cost_caps.get(current_model, 0):.0f}, "
                f"all: ${state.total_cost:.2f})"
            )

            # Checkpoint
            if checkpoint_path and iteration % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(state, checkpoint_path)

            model_idx += 1
            await asyncio.sleep(REQUEST_DELAY_MS / 1000)

        # Final checkpoint
        if checkpoint_path:
            save_checkpoint(state, checkpoint_path)

    return state


def save_results(state: GenerationState, output_path: Path) -> None:
    """Save generated components to YAML."""
    output: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_requests": state.total_requests,
            "total_tokens": state.total_tokens,
            "total_cost": f"${state.total_cost:.2f}",
            "costs_by_model": {k: f"${v:.2f}" for k, v in state.costs.items() if v > 0},
            "errors": len(state.errors),
        },
        "components": {
            cat: sorted(words) for cat, words in state.components.items()
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, width=120)
    logger.info(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate component candidates using multiple LLM families",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List categories and exit",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["claude", "kimi", "deepseek", "gpt"],
        default=["claude", "kimi", "deepseek", "gpt"],
        help="Models to use (default: all four)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_PER_CATEGORY,
        help=f"Target candidates per category (default: {TARGET_PER_CATEGORY})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output YAML path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts without API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--claude-cap",
        type=float,
        default=DEFAULT_COST_CAPS["claude"],
        help=f"Claude cost cap USD (default: ${DEFAULT_COST_CAPS['claude']:.0f})",
    )
    parser.add_argument(
        "--kimi-cap",
        type=float,
        default=DEFAULT_COST_CAPS["kimi"],
    )
    parser.add_argument(
        "--deepseek-cap",
        type=float,
        default=DEFAULT_COST_CAPS["deepseek"],
    )
    parser.add_argument(
        "--gpt-cap",
        type=float,
        default=DEFAULT_COST_CAPS["gpt"],
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load categories
    data_dir = Path(__file__).parent.parent / "data"
    components_path = data_dir / "components.yaml"

    if not components_path.exists():
        logger.error(f"Components not found: {components_path}")
        sys.exit(1)

    categories = build_category_configs(components_path, args.target)

    if args.list:
        print("\nCategories:")
        print("-" * 50)
        for cat in categories:
            print(f"  {cat.name} ({len(cat.seeds)} seeds, target: {cat.target})")
        sys.exit(0)

    if args.categories:
        categories = [c for c in categories if c.name in args.categories]
        if not categories:
            logger.error(f"No matching categories: {args.categories}")
            sys.exit(1)

    # Resolve API keys
    api_keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }

    models: list[ModelName] = []
    for m in args.models:
        provider = MODEL_PROVIDERS[m]
        if not args.dry_run and not api_keys.get(provider):
            logger.warning(
                f"Skipping {m}: {provider.upper()}_API_KEY not set"
            )
            continue
        models.append(m)

    if not models and not args.dry_run:
        logger.error("No models available — set at least one API key")
        sys.exit(1)

    cost_caps: dict[ModelName, float] = {
        "claude": args.claude_cap,
        "kimi": args.kimi_cap,
        "deepseek": args.deepseek_cap,
        "gpt": args.gpt_cap,
    }

    checkpoint_path = data_dir / "generation_checkpoint.yaml" if args.resume else None

    logger.info("Starting component generation")
    logger.info(f"  Categories: {[c.name for c in categories]}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Target per category: {args.target}")
    logger.info(f"  Cost caps: { {m: f'${cost_caps[m]:.0f}' for m in models} }")

    # Run
    state = asyncio.run(
        run_generation(
            categories=categories,
            models=models,
            api_keys=api_keys,
            cost_caps=cost_caps,
            dry_run=args.dry_run,
            checkpoint_path=checkpoint_path,
        )
    )

    if args.dry_run:
        return

    # Report
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")

    total = sum(len(c) for c in state.components.values())
    print(f"\nTotal unique components: {total}")
    print(f"Total API requests: {state.total_requests}")
    print(f"Total cost: ${state.total_cost:.2f}")

    # Per-model breakdown
    print(f"\n{'Model':<12} {'Requests':>8} {'Input Tok':>10} {'Output Tok':>10} {'Cached':>10} {'Cost':>10}")
    print("-" * 62)
    for model, ms in sorted(state.model_stats.items()):
        if ms.requests > 0:
            cache_pct = (
                f"({ms.cached_tokens / max(ms.input_tokens, 1) * 100:.0f}%)"
                if ms.cached_tokens > 0 else ""
            )
            print(
                f"{model:<12} {ms.requests:>8} {ms.input_tokens:>10,} "
                f"{ms.output_tokens:>10,} {ms.cached_tokens:>7,} {cache_pct:>3} "
                f"${ms.cost:>8.2f}"
            )

    if state.errors:
        print(f"\nErrors: {len(state.errors)}")

    # Per-category counts
    print(f"\n{'Category':<25} {'Count':>6}")
    print("-" * 33)
    for cat, words in sorted(state.components.items()):
        print(f"  {cat:<23} {len(words):>6}")

    # Save
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f"generated_candidates_{ts}.yaml"

    save_results(state, output_path)


if __name__ == "__main__":
    main()
