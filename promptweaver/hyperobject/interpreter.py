"""Prompt interpreter — maps GeneratedPrompt data to Scene parameters.

This is the bridge between the prompt generation system and the
hyperobject renderer. It translates template IDs to geometry types
and component words to visual parameters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from .lut import Vec3
from .geometry import Mesh, PointCloud, VoxelGrid, HeightMap
from .particles import ParticleSystem, particle_system_for_word
from .postfx import effect_for_word
from .rasterizer import Light
from .scene import GeomKind, Scene
from .shaders import ShaderRamp, shader_for_word, DEFAULT_SHADER
from .transform import Camera


# ── template → geometry kind mapping ────────────────────────────────────

TEMPLATE_GEOM: dict[str, GeomKind] = {
    "material_study":    GeomKind.MESH_FILLED,
    "textural_macro":    GeomKind.HEIGHTMAP,
    "environmental":     GeomKind.HEIGHTMAP,
    "atmospheric_depth": GeomKind.POINT_CLOUD,
    "process_state":     GeomKind.MESH_FILLED,
    "material_collision": GeomKind.DUAL_MESH,
    "specimen":          GeomKind.MESH_WIREFRAME,
    "minimal_object":    GeomKind.SURFACE_DIRECT,  # torus: donut.c-style
    "abstract_field":    GeomKind.POINT_CLOUD,
    "temporal_diptych":  GeomKind.DUAL_MESH,
    "liminal":           GeomKind.MESH_WIREFRAME,
    "ruin_state":        GeomKind.MESH_FILLED,
    "essence":           GeomKind.SURFACE_DIRECT,   # möbius: direct sampling
    "site_decay":        GeomKind.VOXEL_GRID,
}


# ── light presets (from light_behavior) ─────────────────────────────────

LIGHT_PRESETS: dict[str, Light] = {
    "directional_above":   Light(direction=Vec3(0.0, -1.0, 0.3).normalized(), intensity=1.2),
    "dramatic_side":       Light(direction=Vec3(-0.8, -0.4, 0.4).normalized(), intensity=1.3),
    "rim":                 Light(direction=Vec3(-0.5, -0.3, -0.8).normalized(), intensity=1.2),
    "wrap_soft":           Light(direction=Vec3(0.3, -0.7, 0.5).normalized(), intensity=1.0, wrap=0.4),
    "wrap_strong":         Light(direction=Vec3(0.2, -0.5, 0.6).normalized(), intensity=0.9, wrap=0.6),
    "overhead":            Light(direction=Vec3(0.0, -1.0, 0.1).normalized(), intensity=1.2),
    "low_angle":           Light(direction=Vec3(0.3, -0.3, 0.7).normalized(), intensity=1.3),
    "backlight":           Light(direction=Vec3(0.1, -0.3, -1.0).normalized(), intensity=0.9, wrap=0.3),
}

_LIGHT_PRESETS_LIST = list(LIGHT_PRESETS.values())


# ── camera presets (from spatial_logic) ─────────────────────────────────

CAMERA_PRESETS: dict[str, Camera] = {
    "front":     Camera(position=Vec3(0.0, 0.0, 2.5), target=Vec3(0.0, 0.0, 0.0)),
    "elevated":  Camera(position=Vec3(0.0, 1.0, 2.2), target=Vec3(0.0, 0.0, 0.0)),
    "side":      Camera(position=Vec3(2.2, 0.3, 0.8), target=Vec3(0.0, 0.0, 0.0)),
    "isometric": Camera(position=Vec3(1.8, 1.4, 1.8), target=Vec3(0.0, 0.0, 0.0)),
    "close":     Camera(position=Vec3(0.0, 0.2, 1.8), target=Vec3(0.0, 0.0, 0.0)),
    "far":       Camera(position=Vec3(0.0, 0.3, 3.5), target=Vec3(0.0, 0.0, 0.0)),
    "low":       Camera(position=Vec3(0.0, -0.3, 2.5), target=Vec3(0.0, 0.2, 0.0)),
    "orbit":     Camera(position=Vec3(1.8, 0.3, 1.8), target=Vec3(0.0, 0.0, 0.0)),
}

_CAMERA_PRESETS_LIST = list(CAMERA_PRESETS.values())


# ── speed presets (from temporal_state) ─────────────────────────────────

SPEED_PRESETS: list[float] = [
    0.3,   # very slow (suspended, frozen)
    0.5,   # slow (crystallizing, settling)
    0.7,   # moderate-slow (eroding)
    1.0,   # normal
    1.3,   # moderate-fast (blooming, growing)
    1.6,   # fast (shattering, igniting)
    2.0,   # very fast (exploding)
    0.8,   # moderate (default)
]


# ── zoom presets (from scale_perspective) ───────────────────────────────

ZOOM_OFFSETS: list[float] = [
    -0.8,   # much closer (macro, electron microscope)
    -0.5,   # closer
    -0.3,   # slightly closer
    0.0,    # default
    0.3,    # slightly farther
    0.5,    # farther (aerial)
    0.8,    # much farther (geological)
    1.2,    # extreme far (satellite)
]


# ── deterministic word → index hashing ──────────────────────────────────


def _word_hash(word: str, n: int) -> int:
    """Deterministic hash of a word to an index in [0, n)."""
    h = hashlib.md5(word.encode(), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) % n


# ── interpreter ─────────────────────────────────────────────────────────


def interpret_light(words: list[str]) -> Light:
    """Map light_behavior word(s) to a Light preset."""
    if not words:
        return Light(direction=Vec3(0.3, -0.8, 0.5).normalized(), intensity=1.2)
    idx = _word_hash(words[0], len(_LIGHT_PRESETS_LIST))
    return _LIGHT_PRESETS_LIST[idx]


def interpret_camera(words: list[str]) -> Camera:
    """Map spatial_logic word(s) to a Camera preset."""
    if not words:
        return Camera(position=Vec3(0.0, 0.3, 2.5), target=Vec3(0.0, 0.0, 0.0))
    idx = _word_hash(words[0], len(_CAMERA_PRESETS_LIST))
    return _CAMERA_PRESETS_LIST[idx]


def interpret_speed(words: list[str]) -> float:
    """Map temporal_state word(s) to animation speed multiplier."""
    if not words:
        return 1.0
    idx = _word_hash(words[0], len(SPEED_PRESETS))
    return SPEED_PRESETS[idx]


def interpret_zoom(words: list[str], camera: Camera) -> Camera:
    """Adjust camera distance based on scale_perspective word(s)."""
    if not words:
        return camera
    idx = _word_hash(words[0], len(ZOOM_OFFSETS))
    offset = ZOOM_OFFSETS[idx]
    # Move camera along its forward axis
    direction = (camera.position - camera.target).normalized()
    new_pos = camera.position + direction * offset
    return Camera(
        position=new_pos,
        target=camera.target,
        fov=camera.fov,
    )


def interpret_mesh_detail(words: list[str]) -> int:
    """Map subject_form word(s) to mesh subdivision level (0–2)."""
    if not words:
        return 1
    idx = _word_hash(words[0], 3)
    return idx


def interpret_shader(words: list[str]) -> ShaderRamp:
    """Map material_substance word(s) to a ShaderRamp."""
    if not words:
        return DEFAULT_SHADER
    return shader_for_word(words[0])


def interpret_postfx(words: list[str]) -> list[str]:
    """Map medium_render word(s) to a list of post-processing effect names."""
    if not words:
        return []
    return effect_for_word(words[0])


def interpret_particles(words: list[str]) -> ParticleSystem | None:
    """Map atmosphere_field word(s) to a ParticleSystem."""
    if not words:
        return None
    return particle_system_for_word(words[0])


def configure_scene(
    scene: Scene,
    visual_state: dict[str, list[str]],
    template_id: str,
) -> None:
    """Apply all 12 visual state parameters to a scene.

    Reads from every component slot in the visual state and configures
    the scene's geometry kind, light, camera, shader, post-fx, particles,
    and animation speed.
    """
    # Geometry kind from template
    scene.geom_kind = TEMPLATE_GEOM.get(template_id, GeomKind.MESH_FILLED)

    # Light (from light_behavior)
    scene.light = interpret_light(visual_state.get("light_behavior", []))

    # Camera (base from spatial_logic, adjusted by scale_perspective)
    camera = interpret_camera(visual_state.get("spatial_logic", []))
    camera = interpret_zoom(visual_state.get("scale_perspective", []), camera)
    scene.camera = camera

    # Animation speed (from temporal_state)
    scene.anim.speed_scale = interpret_speed(visual_state.get("temporal_state", []))

    # Surface shader (from material_substance)
    shader = interpret_shader(visual_state.get("material_substance", []))
    scene.shader_chars = shader.chars

    # Post-processing effects (from medium_render)
    scene.postfx_names = interpret_postfx(visual_state.get("medium_render", []))

    # Particles (from atmosphere_field)
    scene.particle_system = interpret_particles(visual_state.get("atmosphere_field", []))
