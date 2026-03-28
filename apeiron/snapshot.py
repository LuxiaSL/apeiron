"""Headless hyperobject snapshot renderer.

Runs the 3D ASCII rendering pipeline without the Textual TUI,
producing a single frame suitable for terminal display.
"""

from __future__ import annotations

import logging
from typing import Optional

from rich.text import Text

from .models import GeneratedPrompt
from .palettes import Palette, palette_for_template
from .hyperobject.scene import Scene, GeomKind
from .hyperobject.rasterizer import AsciiRasterizer, TorusSampler, MobiusSampler
from .hyperobject.interpreter import TEMPLATE_GEOM, configure_scene, interpret_mesh_detail
from .hyperobject.state import VisualState
from .hyperobject import primitives

logger = logging.getLogger(__name__)

# Default viewport size (columns x rows)
DEFAULT_WIDTH = 80
DEFAULT_HEIGHT = 24


def _build_geometry(scene: Scene, template_id: str, visual_state: VisualState) -> None:
    """Construct the geometry for a given template (mirrors viewport logic)."""
    scene.clear_geometry()
    geom_kind = TEMPLATE_GEOM.get(template_id, GeomKind.MESH_FILLED)
    scene.geom_kind = geom_kind

    subject_words = visual_state.get("subject_form")

    try:
        if template_id == "material_study":
            detail = interpret_mesh_detail(subject_words)
            scene.mesh = primitives.make_icosahedron(subdivisions=min(detail, 2))

        elif template_id == "textural_macro":
            scene.heightmap = primitives.make_noise_surface()
            scene.heightmap_mesh = None

        elif template_id == "environmental":
            scene.heightmap = primitives.make_terrain()
            scene.heightmap_mesh = None

        elif template_id == "atmospheric_depth":
            scene.cloud = primitives.make_particle_nebula()

        elif template_id == "process_state":
            scene.mesh = primitives.make_metaballs()

        elif template_id == "material_collision":
            mesh_a, mesh_b = primitives.make_intersecting_solids()
            scene.mesh = mesh_a
            scene.mesh_b = mesh_b
            scene.dual_mesh_mode = "overlay"

        elif template_id == "specimen":
            scene.mesh = primitives.make_wireframe_organism()

        elif template_id == "minimal_object":
            scene.surface_sampler = TorusSampler(R=1.0, r=0.5)

        elif template_id == "abstract_field":
            scene.cloud = primitives.make_lorenz_attractor()

        elif template_id == "temporal_diptych":
            mesh_a, mesh_b = primitives.make_split_morph_pair()
            scene.mesh = mesh_a
            scene.mesh_b = mesh_b
            scene.dual_mesh_mode = "morph"

        elif template_id == "liminal":
            scene.mesh = primitives.make_corridor()

        elif template_id == "ruin_state":
            mesh, groups = primitives.make_fragmenting_solid()
            scene.mesh = mesh
            scene.fragment_groups = groups

        elif template_id == "essence":
            scene.surface_sampler = MobiusSampler()

        elif template_id == "site_decay":
            scene.voxels = primitives.make_voxel_grid()

    except Exception:
        logger.exception("Snapshot: failed to build geometry for %s", template_id)
        scene.geom_kind = GeomKind.TESSERACT


def render_snapshot(
    prompt: GeneratedPrompt,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    frames: int = 36,
    tesseract: bool = False,
) -> Text:
    """Render a single-frame ASCII snapshot for a prompt.

    Args:
        prompt: The generated prompt to visualize.
        width: Viewport width in columns.
        height: Viewport height in rows.
        frames: Number of frames to simulate before capturing.
            More frames = more rotation/animation progression.
        tesseract: If True, render the 4D tesseract wireframe instead
            of the template-specific geometry.

    Returns:
        A Rich Text object with the rendered frame.
    """
    scene = Scene()
    rast = AsciiRasterizer(width, height)
    visual_state = VisualState()

    # Apply prompt to visual state
    visual_state.apply_prompt(prompt)

    # Set up palette
    palette = palette_for_template(prompt.template_id)
    scene.styles = (palette.bright, palette.primary, palette.rain_mid, palette.rain_dim)

    if tesseract:
        # Render the tesseract wireframe
        try:
            verts, edges = primitives.make_tesseract()
            scene.tesseract_verts = verts
            scene.tesseract_edges = edges
            scene.geom_kind = GeomKind.TESSERACT
        except Exception:
            logger.exception("Snapshot: failed to build tesseract")
            return Text("  // tesseract render failed", style="dim red")

        # Tesseract needs brighter styles than the palette dim tones
        # to be visible in a static terminal context (no dark background)
        scene.styles = (
            "bold bright_white",
            palette.bright,
            palette.primary,
            palette.primary,
        )
    else:
        # Build template-specific geometry
        _build_geometry(scene, prompt.template_id, visual_state)

        # Load tesseract anyway (needed for transitions/fallback)
        try:
            verts, edges = primitives.make_tesseract()
            scene.tesseract_verts = verts
            scene.tesseract_edges = edges
        except Exception:
            pass

    # Configure scene from prompt components (skip for tesseract —
    # it would override our boosted styles)
    if not tesseract:
        configure_scene(scene, visual_state.slots, prompt.template_id)

    # Simulate frames to reach an interesting state
    dt = 1.0 / 18.0
    for _ in range(frames):
        scene.tick(dt)

    # Render the final frame
    scene.render(rast)

    return rast.grid.to_rich_text()
