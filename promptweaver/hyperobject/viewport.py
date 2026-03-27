"""HyperobjectViewport — Textual widget for real-time 3D ASCII rendering.

This is the main integration point. It manages the render loop, scene
state, and responds to prompt generation events from the app.
"""

from __future__ import annotations

import time
from typing import Optional

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

from ..models import GeneratedPrompt
from ..palettes import Palette

from .scene import Scene, GeomKind
from .rasterizer import AsciiRasterizer
from .interpreter import TEMPLATE_GEOM, configure_scene, interpret_mesh_detail
from .state import VisualState
from .embedding_cache import EmbeddingCache
from .dynamics import compute_dynamics
from . import primitives


class HyperobjectViewport(Static):
    """3D ASCII renderer driven by prompt semantics.

    Renders at ~18fps using a timer. Responds to prompt generation
    events by transitioning between geometric forms.
    """

    DEFAULT_CSS = """
    HyperobjectViewport {
        height: 1fr;
        min-height: 3;
        overflow: hidden;
    }
    """

    TARGET_FPS: float = 18.0
    FRAME_INTERVAL: float = 1.0 / TARGET_FPS

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._timer: Optional[Timer] = None
        self._palette: Optional[Palette] = None
        self._scene: Optional[Scene] = None
        self._rasterizer: Optional[AsciiRasterizer] = None
        self._visual_state: Optional[VisualState] = None
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._frame_count: int = 0
        self._initialized: bool = False
        self._current_template: str = ""
        self._last_tick: float = 0.0

    def on_mount(self) -> None:
        self._timer = self.set_interval(self.FRAME_INTERVAL, self._tick)
        self._last_tick = time.monotonic()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # ── public API ────────────────────────────────────────────────────

    def set_palette(self, palette: Palette) -> None:
        """Update the color palette (called when template changes)."""
        self._palette = palette
        if self._scene is not None:
            self._scene.styles = (
                palette.bright,
                palette.primary,
                palette.rain_mid,
                palette.rain_dim,
            )

    def set_prompt(self, prompt: GeneratedPrompt) -> None:
        """Respond to a new prompt generation.

        Triggers a transition and reconfigures the scene based on
        the prompt's template and components.
        """
        self._ensure_initialized()

        scene = self._scene
        if scene is None:
            return

        # ALWAYS reset idle state on every prompt (not just template changes)
        scene.on_new_prompt()

        # Update visual state (component persistence)
        changed: set[str] = set()
        if self._visual_state is not None:
            changed = self._visual_state.apply_prompt(prompt)

        # Determine if we need a geometry change (template switch)
        template_changed = prompt.template_id != self._current_template
        self._current_template = prompt.template_id

        if template_changed:
            scene.capture_transition_source()
            self._build_geometry(prompt.template_id)

        # Apply all visual parameters from the accumulated state
        if self._visual_state is not None:
            configure_scene(scene, self._visual_state.slots, prompt.template_id)

        # Update palette styles on scene
        if self._palette is not None:
            p = self._palette
            scene.styles = (p.bright, p.primary, p.rain_mid, p.rain_dim)

        # Compute embedding dynamics
        if self._visual_state is not None:
            try:
                dyn = compute_dynamics(self._visual_state, self._embedding_cache)
                scene.anim.speed_scale *= dyn.energy * 1.5 + 0.5
            except Exception:
                pass

        if template_changed:
            scene.start_transition()

    # ── initialization ────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Set up scene, rasterizer, and visual state on first use."""
        if self._initialized:
            return
        self._initialized = True

        self._scene = Scene()

        # Load tesseract (always available for transitions + idle)
        try:
            verts, edges = primitives.make_tesseract()
            self._scene.tesseract_verts = verts
            self._scene.tesseract_edges = edges
            self._scene.geom_kind = GeomKind.TESSERACT
        except Exception:
            pass

        # Create rasterizer
        w, h = max(self.size.width, 10), max(self.size.height, 3)
        self._rasterizer = AsciiRasterizer(w, h)

        # Visual state accumulator
        try:
            self._visual_state = VisualState()
        except Exception:
            pass

        # Embedding cache (optional — works without it)
        try:
            self._embedding_cache = EmbeddingCache()
        except Exception:
            self._embedding_cache = None

        # Apply initial palette
        if self._palette is not None:
            p = self._palette
            self._scene.styles = (p.bright, p.primary, p.rain_mid, p.rain_dim)

    def _build_geometry(self, template_id: str) -> None:
        """Construct the geometry for a given template."""
        scene = self._scene
        if scene is None:
            return

        scene.clear_geometry()
        geom_kind = TEMPLATE_GEOM.get(template_id, GeomKind.MESH_FILLED)
        scene.geom_kind = geom_kind

        # Get mesh detail from subject_form if available
        subject_words = (
            self._visual_state.get("subject_form")
            if self._visual_state else []
        )

        try:
            if template_id == "material_study":
                detail = interpret_mesh_detail(subject_words)
                scene.mesh = primitives.make_icosahedron(
                    subdivisions=min(detail, 2)
                )

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
                scene.mesh = primitives.make_torus()

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
                scene.mesh = primitives.make_mobius_strip()

            elif template_id == "site_decay":
                scene.voxels = primitives.make_voxel_grid()

        except Exception:
            scene.geom_kind = GeomKind.TESSERACT

    # ── render loop ───────────────────────────────────────────────────

    def _tick(self) -> None:
        """Called every frame by the timer."""
        now = time.monotonic()
        dt = now - self._last_tick if self._last_tick > 0 else self.FRAME_INTERVAL
        self._last_tick = now
        dt = min(dt, 0.1)

        scene = self._scene
        rast = self._rasterizer
        if scene is None or rast is None:
            self._render_placeholder()
            return

        # Resize rasterizer if widget size changed
        w, h = max(self.size.width, 10), max(self.size.height, 3)
        rast.resize(w, h)

        # Advance animation + particles
        scene.tick(dt)

        # Render geometry + particles + postfx
        scene.render(rast)

        # Output
        self.update(rast.grid.to_rich_text())
        self._frame_count += 1

    def _render_placeholder(self) -> None:
        """Show a simple placeholder before initialization."""
        p = self._palette
        style = p.dim if p else "dim green"
        self.update(Text(
            "  // hyperobject viewport awaiting generation...", style=style
        ))
