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
from ..palettes import Palette, PALETTES, DEFAULT_PALETTE_NAME


# Late imports to avoid circular deps — these are resolved at runtime
# when the first prompt arrives.
_scene_mod = None
_rasterizer_mod = None
_interpreter_mod = None
_primitives_mod = None
_state_mod = None
_dynamics_mod = None


def _lazy_imports() -> bool:
    """Import heavy modules on first use."""
    global _scene_mod, _rasterizer_mod, _interpreter_mod
    global _primitives_mod, _state_mod, _dynamics_mod
    if _scene_mod is not None:
        return True
    try:
        from . import scene as _sm
        from . import rasterizer as _rm
        from . import interpreter as _im
        _scene_mod = _sm
        _rasterizer_mod = _rm
        _interpreter_mod = _im

        # Optional modules — fail gracefully
        try:
            from . import primitives as _pm
            _primitives_mod = _pm
        except Exception:
            _primitives_mod = None

        try:
            from . import state as _stm
            _state_mod = _stm
        except Exception:
            _state_mod = None

        try:
            from . import dynamics as _dm
            _dynamics_mod = _dm
        except Exception:
            _dynamics_mod = None

        return True
    except Exception:
        return False


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
        self._scene: object = None  # Scene (lazy)
        self._rasterizer: object = None  # AsciiRasterizer (lazy)
        self._visual_state: object = None  # VisualState (lazy)
        self._embedding_cache: object = None  # EmbeddingCache (lazy)
        self._last_frame_time: float = 0.0
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
        if self._scene is not None and _scene_mod is not None:
            scene = self._scene
            scene.styles = (  # type: ignore[attr-defined]
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
        if not _lazy_imports():
            return

        self._ensure_initialized()

        # Update visual state (component persistence)
        changed: set[str] = set()
        if _state_mod is not None and self._visual_state is not None:
            changed = self._visual_state.apply_prompt(prompt)  # type: ignore[union-attr]

        # Determine if we need a geometry change (template switch)
        template_changed = prompt.template_id != self._current_template
        self._current_template = prompt.template_id

        scene = self._scene
        assert scene is not None and _scene_mod is not None

        if template_changed:
            # Full transition: dissolve → tesseract → new geometry
            scene.start_transition()  # type: ignore[union-attr]
            self._build_geometry(prompt.template_id)

        # Apply visual parameters from components
        vs = self._visual_state
        if _interpreter_mod is not None and vs is not None:
            slots = vs.slots if hasattr(vs, "slots") else {}  # type: ignore[union-attr]
            _interpreter_mod.configure_scene(scene, slots, prompt.template_id)  # type: ignore[arg-type]

        # Update palette styles on scene
        if self._palette is not None:
            p = self._palette
            scene.styles = (p.bright, p.primary, p.rain_mid, p.rain_dim)  # type: ignore[attr-defined]

        # Compute embedding dynamics if available
        if _dynamics_mod is not None and vs is not None:
            try:
                dynamics = _dynamics_mod.compute_dynamics(vs, self._embedding_cache)
                scene.anim.speed_scale *= dynamics.energy * 1.5 + 0.5  # type: ignore[union-attr]
            except Exception:
                pass  # Embedding dynamics are optional

    # ── initialization ────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Set up scene, rasterizer, and visual state on first use."""
        if self._initialized:
            return
        self._initialized = True

        assert _scene_mod is not None and _rasterizer_mod is not None

        # Create scene with tesseract as initial geometry
        self._scene = _scene_mod.Scene()
        scene = self._scene

        # Load tesseract geometry (always available for transitions)
        if _primitives_mod is not None:
            try:
                verts, edges = _primitives_mod.make_tesseract()
                scene.tesseract_verts = verts  # type: ignore[union-attr]
                scene.tesseract_edges = edges  # type: ignore[union-attr]
                scene.geom_kind = _scene_mod.GeomKind.TESSERACT  # type: ignore[union-attr]
            except Exception:
                pass

        # Create rasterizer (sized to widget)
        w, h = max(self.size.width, 10), max(self.size.height, 3)
        self._rasterizer = _rasterizer_mod.AsciiRasterizer(w, h)

        # Create visual state
        if _state_mod is not None:
            try:
                self._visual_state = _state_mod.VisualState()
            except Exception:
                pass

        # Load embedding cache
        try:
            from .embedding_cache import EmbeddingCache
            self._embedding_cache = EmbeddingCache()
        except Exception:
            self._embedding_cache = None

        # Apply initial palette
        if self._palette is not None:
            p = self._palette
            scene.styles = (p.bright, p.primary, p.rain_mid, p.rain_dim)  # type: ignore[union-attr]

    def _build_geometry(self, template_id: str) -> None:
        """Construct the geometry for a given template."""
        if _primitives_mod is None or self._scene is None:
            return

        scene = self._scene
        assert _interpreter_mod is not None

        geom_kind = _interpreter_mod.TEMPLATE_GEOM.get(
            template_id, _scene_mod.GeomKind.MESH_FILLED  # type: ignore[union-attr]
        )
        scene.geom_kind = geom_kind  # type: ignore[union-attr]

        try:
            if template_id == "material_study":
                detail = _interpreter_mod.interpret_mesh_detail(
                    (self._visual_state.slots if self._visual_state and hasattr(self._visual_state, "slots") else {}).get("subject_form", [])  # type: ignore[union-attr]
                )
                scene.mesh = _primitives_mod.make_icosahedron(subdivisions=min(detail, 2))  # type: ignore[union-attr]

            elif template_id == "textural_macro":
                scene.heightmap = _primitives_mod.make_noise_surface()  # type: ignore[union-attr]
                scene.heightmap_mesh = None  # type: ignore[union-attr]

            elif template_id == "environmental":
                scene.heightmap = _primitives_mod.make_terrain()  # type: ignore[union-attr]
                scene.heightmap_mesh = None  # type: ignore[union-attr]

            elif template_id == "atmospheric_depth":
                scene.cloud = _primitives_mod.make_particle_nebula()  # type: ignore[union-attr]

            elif template_id == "process_state":
                scene.mesh = _primitives_mod.make_metaballs()  # type: ignore[union-attr]

            elif template_id == "material_collision":
                mesh_a, mesh_b = _primitives_mod.make_intersecting_solids()
                scene.mesh = mesh_a  # type: ignore[union-attr]
                scene.mesh_b = mesh_b  # type: ignore[union-attr]

            elif template_id == "specimen":
                scene.mesh = _primitives_mod.make_wireframe_organism()  # type: ignore[union-attr]

            elif template_id == "minimal_object":
                scene.mesh = _primitives_mod.make_torus()  # type: ignore[union-attr]

            elif template_id == "abstract_field":
                scene.cloud = _primitives_mod.make_lorenz_attractor()  # type: ignore[union-attr]

            elif template_id == "temporal_diptych":
                mesh_a, mesh_b = _primitives_mod.make_split_morph_pair()
                scene.mesh = mesh_a  # type: ignore[union-attr]
                scene.mesh_b = mesh_b  # type: ignore[union-attr]

            elif template_id == "liminal":
                scene.mesh = _primitives_mod.make_corridor()  # type: ignore[union-attr]

            elif template_id == "ruin_state":
                mesh, groups = _primitives_mod.make_fragmenting_solid()
                scene.mesh = mesh  # type: ignore[union-attr]
                scene.fragment_groups = groups  # type: ignore[union-attr]

            elif template_id == "essence":
                scene.mesh = _primitives_mod.make_mobius_strip()  # type: ignore[union-attr]

            elif template_id == "site_decay":
                scene.voxels = _primitives_mod.make_voxel_grid()  # type: ignore[union-attr]

        except Exception:
            # If any geometry factory fails, fall back to tesseract
            scene.geom_kind = _scene_mod.GeomKind.TESSERACT  # type: ignore[union-attr]

    # ── render loop ───────────────────────────────────────────────────

    def _tick(self) -> None:
        """Called every frame by the timer."""
        now = time.monotonic()
        dt = now - self._last_tick if self._last_tick > 0 else self.FRAME_INTERVAL
        self._last_tick = now

        # Cap dt to avoid huge jumps after pauses
        dt = min(dt, 0.1)

        if not _lazy_imports() or not self._initialized:
            self._render_placeholder()
            return

        scene = self._scene
        rast = self._rasterizer
        if scene is None or rast is None:
            self._render_placeholder()
            return

        # Resize rasterizer if widget size changed
        w, h = max(self.size.width, 10), max(self.size.height, 3)
        rast.resize(w, h)  # type: ignore[union-attr]

        # Advance animation
        scene.tick(dt)  # type: ignore[union-attr]

        # Render
        scene.render(rast)  # type: ignore[union-attr]

        # Convert to Rich Text and display
        text = rast.grid.to_rich_text()  # type: ignore[union-attr]
        self.update(text)

        self._frame_count += 1

    def _render_placeholder(self) -> None:
        """Show a simple placeholder before initialization."""
        p = self._palette
        style = p.dim if p else "dim green"
        text = Text(f"  // hyperobject viewport awaiting generation...", style=style)
        self.update(text)
