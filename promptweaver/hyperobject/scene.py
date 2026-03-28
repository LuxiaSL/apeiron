"""Scene graph — orchestrates geometry, camera, lighting, and animation.

A Scene holds everything needed to render one frame: the geometry (which
changes per template), the camera, the light, and the animation state.
The PromptInterpreter builds scenes from GeneratedPrompt data.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .lut import (
    Mat4,
    Vec3,
    Vec4,
    fast_sin,
    fast_cos,
    lerp_f,
    rotate_4d,
    project_4d_to_3d,
    smoothstep,
)
from .geometry import Mesh, PointCloud, VoxelGrid, HeightMap, noise3
from .rasterizer import (
    AsciiRasterizer,
    CharGrid,
    Light,
    DEFAULT_LIGHT,
    DONUT_LUMINANCE_RAMP,
    SurfaceSampler,
    TorusSampler,
    SphereSampler,
    MobiusSampler,
)
from .transform import Camera, ProjectionContext

# Late import to avoid circular deps
_postfx = None
_particles_mod = None
logger = logging.getLogger(__name__)


def _ensure_fx() -> None:
    global _postfx, _particles_mod
    if _postfx is None:
        try:
            from . import postfx as _pfx
            from . import particles as _ptc
            _postfx = _pfx
            _particles_mod = _ptc
        except Exception:
            logger.exception("Failed to import post-processing or particle modules")


# ── geometry type tag ───────────────────────────────────────────────────


class GeomKind(Enum):
    """What kind of geometry the scene holds."""

    MESH_FILLED = auto()
    MESH_WIREFRAME = auto()
    POINT_CLOUD = auto()
    VOXEL_GRID = auto()
    HEIGHTMAP = auto()
    TESSERACT = auto()
    DUAL_MESH = auto()       # two meshes (material_collision, temporal_diptych)
    SURFACE_DIRECT = auto()  # donut.c-style direct surface sampling


# ── animation state ─────────────────────────────────────────────────────


@dataclass
class AnimationState:
    """Per-frame animation accumulators."""

    time: float = 0.0        # total elapsed time (seconds)
    angle_x: float = 0.0     # rotation accumulator X
    angle_y: float = 0.0     # rotation accumulator Y
    angle_z: float = 0.0     # rotation accumulator Z
    angle_4d_xw: float = 0.0  # 4D rotation plane XW
    angle_4d_yz: float = 0.0  # 4D rotation plane YZ
    morph_t: float = 0.0     # morph interpolation [0, 1]
    breath: float = 0.0      # breathing oscillation [0, 1]
    phase: float = 0.0       # generic phase accumulator
    speed_scale: float = 1.0  # global speed multiplier (from prompt energy)

    def tick(self, dt: float) -> None:
        """Advance all accumulators by dt seconds."""
        self.time += dt
        scaled_dt = dt * self.speed_scale
        self.angle_y += 0.4 * scaled_dt
        self.angle_x += 0.15 * scaled_dt
        self.angle_4d_xw += 0.25 * scaled_dt
        self.angle_4d_yz += 0.18 * scaled_dt
        self.breath = (fast_sin(self.time * 1.5) + 1.0) * 0.5
        self.phase = self.time


# ── transition state ────────────────────────────────────────────────────


class TransitionPhase(Enum):
    NONE = auto()
    DISSOLVE = auto()     # current geometry fragmenting
    TESSERACT = auto()    # tesseract rotation
    FORM = auto()         # new geometry materializing


@dataclass
class TransitionState:
    """Manages the dissolve → tesseract → new form sequence."""

    phase: TransitionPhase = TransitionPhase.NONE
    progress: float = 0.0  # 0→1 within current phase
    total_frames: int = 126  # ~7 seconds at 18fps
    current_frame: int = 0

    # Phase boundaries (as fraction of total)
    # Dissolve: 20% (~1.4s) — geometry fades out gradually
    # Tesseract: 55% (~3.85s) — the hypercube holds court
    # Form: 25% (~1.75s) — new geometry materializes gradually
    dissolve_end: float = 0.20
    tesseract_end: float = 0.75
    # form phase runs from tesseract_end to 1.0

    @property
    def active(self) -> bool:
        return self.phase != TransitionPhase.NONE

    def start(self) -> None:
        self.phase = TransitionPhase.DISSOLVE
        self.progress = 0.0
        self.current_frame = 0

    def tick(self) -> None:
        if not self.active:
            return
        self.current_frame += 1
        self.progress = min(self.current_frame / max(self.total_frames, 1), 1.0)

        if self.progress < self.dissolve_end:
            self.phase = TransitionPhase.DISSOLVE
        elif self.progress < self.tesseract_end:
            self.phase = TransitionPhase.TESSERACT
        else:
            self.phase = TransitionPhase.FORM

        if self.progress >= 1.0:
            self.phase = TransitionPhase.NONE
            self.progress = 0.0
            self.current_frame = 0

    def phase_progress(self) -> float:
        """Progress within the current phase (0→1)."""
        if self.phase == TransitionPhase.DISSOLVE:
            return self.progress / self.dissolve_end
        elif self.phase == TransitionPhase.TESSERACT:
            return (self.progress - self.dissolve_end) / (self.tesseract_end - self.dissolve_end)
        elif self.phase == TransitionPhase.FORM:
            return (self.progress - self.tesseract_end) / (1.0 - self.tesseract_end)
        return 0.0


@dataclass
class SceneSnapshot:
    """Frozen render state for the outgoing side of a transition."""

    geom_kind: GeomKind
    mesh: Optional[Mesh]
    mesh_b: Optional[Mesh]
    cloud: Optional[PointCloud]
    voxels: Optional[VoxelGrid]
    heightmap: Optional[HeightMap]
    heightmap_mesh: Optional[Mesh]
    shader_chars: str
    light: Light
    camera: Camera
    styles: tuple[str, str, str, str]
    fragment_groups: list[list[int]]
    dual_mesh_mode: str = "overlay"
    surface_sampler: SurfaceSampler | None = None


@dataclass(slots=True)
class FragmentRenderCache:
    """Reusable exploded fragment topology for translational drift."""

    source_mesh: Mesh
    group_signature: tuple[tuple[int, ...], ...]
    base_vertices: tuple[Vec3, ...]
    vertex_group_indices: tuple[int, ...]
    faces: list[tuple[int, ...]]
    normals: list[Vec3]
    vertex_normals: list[Vec3]


# ── scene ───────────────────────────────────────────────────────────────


@dataclass
class Scene:
    """Complete scene state for one frame."""

    # Geometry (one of these will be active based on geom_kind)
    geom_kind: GeomKind = GeomKind.MESH_FILLED
    mesh: Optional[Mesh] = None
    mesh_b: Optional[Mesh] = None  # second mesh for DUAL_MESH
    cloud: Optional[PointCloud] = None
    voxels: Optional[VoxelGrid] = None
    heightmap: Optional[HeightMap] = None
    heightmap_mesh: Optional[Mesh] = None  # cached mesh from heightmap

    # Direct surface sampler (donut.c-style rendering)
    surface_sampler: SurfaceSampler | None = None

    # Tesseract (always available for transitions)
    tesseract_verts: list[Vec4] = field(default_factory=list)
    tesseract_edges: list[tuple[int, int]] = field(default_factory=list)

    # Rendering config — donut.c-inspired 13-char luminance ramp
    shader_chars: str = " .,-~:;=!*#$@"
    light: Light = field(default_factory=lambda: DEFAULT_LIGHT)
    camera: Camera = field(default_factory=lambda: Camera(
        position=Vec3(0.0, 0.3, 2.5),
        target=Vec3(0.0, 0.0, 0.0),
    ))

    # Palette styles (bright, primary, mid, dim)
    styles: tuple[str, str, str, str] = (
        "bright_green", "green", "#006600", "#003300",
    )

    # Animation
    anim: AnimationState = field(default_factory=AnimationState)
    transition: TransitionState = field(default_factory=TransitionState)

    # Post-processing effects (from medium_render)
    postfx_names: list[str] = field(default_factory=list)

    # Particle system (from atmosphere_field)
    particle_system: object = None  # ParticleSystem or None

    # Fragment groups for ruin_state dissolve
    fragment_groups: list[list[int]] = field(default_factory=list)
    dual_mesh_mode: str = "overlay"
    transition_source: SceneSnapshot | None = None

    # ── model matrix ──────────────────────────────────────────────────

    def model_matrix(self) -> Mat4:
        """Compute the current model transform from animation state."""
        rot_y = Mat4.rotation_y(self.anim.angle_y)
        rot_x = Mat4.rotation_x(self.anim.angle_x)
        return rot_y @ rot_x

    # ── render dispatch ───────────────────────────────────────────────

    def render(self, rast: AsciiRasterizer) -> None:
        """Render the current scene state into the rasterizer."""
        rast.clear()
        width, height = rast.width, rast.height

        if self.transition.active:
            self._render_transition(rast, width, height)
        else:
            self._render_geometry(rast, width, height)

        # Render particles on top of geometry
        self._render_particles(rast, width, height)

        # Apply post-processing effects
        self._apply_postfx(rast.grid)

    def clear_geometry(self) -> None:
        """Reset all geometry-bearing fields before loading a new template."""
        self.mesh = None
        self.mesh_b = None
        self.cloud = None
        self.voxels = None
        self.heightmap = None
        self.heightmap_mesh = None
        self.surface_sampler = None
        self.fragment_groups = []
        self.dual_mesh_mode = "overlay"
        self._fragment_cache = None

    def _render_geometry(self, rast: AsciiRasterizer, w: int, h: int) -> None:
        """Render the active geometry."""
        self._render_geometry_state(
            rast,
            w,
            h,
            self.geom_kind,
            self.mesh,
            self.mesh_b,
            self.cloud,
            self.voxels,
            self.heightmap,
            self.heightmap_mesh,
            self.shader_chars,
            self.light,
            self.styles,
            self.camera,
            self.fragment_groups,
            self.dual_mesh_mode,
            self.surface_sampler,
        )

    def _render_geometry_state(
        self,
        rast: AsciiRasterizer,
        w: int,
        h: int,
        geom_kind: GeomKind,
        mesh: Optional[Mesh],
        mesh_b: Optional[Mesh],
        cloud: Optional[PointCloud],
        voxels: Optional[VoxelGrid],
        heightmap: Optional[HeightMap],
        heightmap_mesh: Optional[Mesh],
        shader_chars: str,
        light: Light,
        styles: tuple[str, str, str, str],
        camera: Camera,
        fragment_groups: list[list[int]],
        dual_mesh_mode: str,
        surface_sampler: SurfaceSampler | None = None,
    ) -> None:
        """Render an explicit geometry state.

        This lets transitions render the preserved outgoing form instead of
        whatever geometry is currently loaded for the incoming prompt.
        """
        model = self.model_matrix()
        ctx = ProjectionContext.build(model, camera, w, h)

        if geom_kind == GeomKind.MESH_FILLED:
            render_mesh = self._mesh_for_render(mesh, fragment_groups)
            if render_mesh:
                rast.draw_mesh_filled(
                    render_mesh, ctx, shader_chars, light, styles,
                )

        elif geom_kind == GeomKind.MESH_WIREFRAME:
            if mesh:
                rast.draw_mesh_wireframe(
                    mesh, ctx,
                    edge_char="·",
                    styles=styles,
                    vertex_char="•",
                )

        elif geom_kind == GeomKind.POINT_CLOUD:
            if cloud:
                rast.draw_points(cloud, ctx, styles=styles)

        elif geom_kind == GeomKind.VOXEL_GRID:
            if voxels:
                rast.draw_voxels(voxels, ctx, styles=styles)

        elif geom_kind == GeomKind.HEIGHTMAP:
            render_heightmap_mesh = heightmap_mesh
            if heightmap and render_heightmap_mesh is None:
                render_heightmap_mesh = heightmap.to_mesh()
                if heightmap is self.heightmap:
                    self.heightmap_mesh = render_heightmap_mesh
            if render_heightmap_mesh:
                rast.draw_heightmap(
                    render_heightmap_mesh, ctx,
                    shader_chars, light, styles,
                )

        elif geom_kind == GeomKind.SURFACE_DIRECT:
            sampler = surface_sampler if surface_sampler is not None else self.surface_sampler
            if sampler is not None:
                rast.draw_surface_direct(
                    sampler, ctx, light, styles,
                    luminance_ramp=DONUT_LUMINANCE_RAMP,
                )

        elif geom_kind == GeomKind.TESSERACT:
            self._render_tesseract(rast, w, h)

        elif geom_kind == GeomKind.DUAL_MESH:
            if dual_mesh_mode == "morph" and mesh and mesh_b:
                morphed = self._morph_mesh(mesh, mesh_b, self.anim.morph_t)
                rast.draw_mesh_filled(
                    morphed, ctx, shader_chars, light, styles,
                )
            elif mesh:
                rast.draw_mesh_filled(
                    mesh, ctx, shader_chars, light, styles,
                )
            if dual_mesh_mode != "morph" and mesh_b:
                # Second mesh with slightly different rotation
                model_b = Mat4.rotation_y(self.anim.angle_y * 1.618)
                model_b = model_b @ Mat4.rotation_x(self.anim.angle_x * 0.7)
                ctx_b = ProjectionContext.build(model_b, camera, w, h)
                rast.draw_mesh_filled(
                    mesh_b, ctx_b, shader_chars, light, styles,
                )

    def _render_tesseract(self, rast: AsciiRasterizer, w: int, h: int) -> None:
        """Render the 4D tesseract."""
        if not self.tesseract_verts:
            return

        # 4D rotation
        rotated = [
            rotate_4d(v, self.anim.angle_4d_xw, self.anim.angle_4d_yz)
            for v in self.tesseract_verts
        ]

        # 4D → 3D projection
        verts_3d = [project_4d_to_3d(v) for v in rotated]

        # 3D → 2D via standard pipeline
        model = Mat4.rotation_y(self.anim.angle_y * 0.3)
        ctx = ProjectionContext.build(model, self.camera, w, h)
        rast.draw_tesseract_wireframe(
            verts_3d, self.tesseract_edges, ctx,
            edge_char="─",
            vertex_char="●",
            styles=self.styles,
        )

    def _render_transition(self, rast: AsciiRasterizer, w: int, h: int) -> None:
        """Render during a transition sequence."""
        phase = self.transition.phase
        t = self.transition.phase_progress()

        if phase == TransitionPhase.DISSOLVE:
            # Fade out current geometry (reduce brightness)
            if self.transition_source is not None:
                self._render_snapshot(rast, w, h, self.transition_source)
            else:
                self._render_geometry(rast, w, h)
            # Apply dissolve: randomly blank cells based on progress
            threshold = t * t  # accelerating
            for cell in rast.grid.cells:
                if cell.char != " " and random.random() < threshold:
                    cell.char = " "
                    cell.style = ""

        elif phase == TransitionPhase.TESSERACT:
            # Render tesseract with smooth fade-in/fade-out at edges
            self._render_tesseract(rast, w, h)
            # Fade envelope: ramp in over first 15%, ramp out over last 15%
            if t < 0.15:
                blank_prob = 1.0 - smoothstep(0.0, 1.0, t / 0.15)
            elif t > 0.85:
                blank_prob = smoothstep(0.0, 1.0, (t - 0.85) / 0.15)
            else:
                blank_prob = 0.0
            if blank_prob > 0.0:
                for cell in rast.grid.cells:
                    if cell.char != " " and random.random() < blank_prob:
                        cell.char = " "
                        cell.style = ""

        elif phase == TransitionPhase.FORM:
            # Render new geometry with progressive reveal
            self._render_geometry(rast, w, h)
            threshold = 1.0 - smoothstep(0.0, 1.0, t)
            for cell in rast.grid.cells:
                if cell.char != " " and random.random() < threshold:
                    cell.char = " "
                    cell.style = ""

    # ── particles ──────────────────────────────────────────────────────

    def _render_particles(self, rast: AsciiRasterizer, w: int, h: int) -> None:
        """Render ambient particles around the geometry."""
        if self.particle_system is None:
            return
        _ensure_fx()
        if _particles_mod is None:
            return

        model = Mat4.identity()  # particles are in world space
        ctx = ProjectionContext.build(model, self.camera, w, h)

        ps = self.particle_system
        if not hasattr(ps, 'particles'):
            return

        bright_s, primary_s, mid_s, dim_s = self.styles
        for p in ps.particles:
            sp = ctx.project_vertex(p.pos)
            if sp is None:
                continue
            # Depth behind main geometry (particles are background)
            adjusted_depth = min(sp.depth + 0.1, 1.0)
            style = dim_s if p.brightness < 0.5 else primary_s
            rast.grid.write(sp.col, sp.row, p.char, style, adjusted_depth)

    # ── post-processing ───────────────────────────────────────────────

    def _apply_postfx(self, grid: CharGrid) -> None:
        """Apply post-processing effects to the completed grid."""
        if not self.postfx_names:
            return
        _ensure_fx()
        if _postfx is None:
            return
        try:
            _postfx.apply_effects(grid, self.postfx_names)
        except Exception:
            logger.exception("Post-processing failed for effects=%s", self.postfx_names)

    # ── animation tick ────────────────────────────────────────────────

    # Heightmap animation throttle (avoids recomputing every frame)
    _hmap_accum: float = 0.0
    _HMAP_INTERVAL: float = 0.15  # ~6.7 updates/sec — smooth enough for noise

    # Voxel erosion state
    _voxel_timer: float = 0.0
    _voxel_eroding: bool = True
    _voxel_original_cells: list[bool] | None = None

    # Fragment drift state
    _fragment_offsets: list[Vec3] | None = None
    _fragment_timer: float = 0.0
    _fragment_cycle: float = 8.0  # seconds per ruin/rebuild cycle
    _fragment_cache: FragmentRenderCache | None = None

    # Lorenz integrator state (for abstract_field trail growth)
    _lorenz_state: tuple[float, float, float] | None = None

    def tick(self, dt: float) -> None:
        """Advance the scene by dt seconds."""
        was_transitioning = self.transition.active
        self.anim.tick(dt)
        self.anim.morph_t = 0.5 + 0.5 * fast_sin(self.anim.time * 0.7)
        if self.transition.active:
            self.transition.tick()
            if was_transitioning and not self.transition.active:
                self.transition_source = None

        # Tick particles
        if self.particle_system is not None and hasattr(self.particle_system, 'tick'):
            try:
                self.particle_system.tick(dt)
            except Exception:
                logger.exception("Particle system tick failed")

        # Per-geometry animations
        if self.geom_kind == GeomKind.HEIGHTMAP:
            self._animate_heightmap(dt)
        elif self.geom_kind == GeomKind.VOXEL_GRID:
            self._animate_voxels(dt)
        elif self.geom_kind == GeomKind.POINT_CLOUD and self.cloud is not None:
            self._animate_cloud(dt)

        # Fragment drift for ruin_state
        if self.fragment_groups and self.mesh is not None:
            self._animate_fragments(dt)

    # ── heightmap animation ───────────────────────────────────────────

    def _animate_heightmap(self, dt: float) -> None:
        """Scroll noise across the heightmap surface.

        Throttled to ~6-7 updates/sec to avoid recomputing the full
        noise field every frame (pure-Python noise3 is the hot path).
        """
        if self.heightmap is None:
            return
        self._hmap_accum += dt
        if self._hmap_accum < self._HMAP_INTERVAL:
            return
        self._hmap_accum = 0.0

        t = self.anim.time
        freq = 0.3
        amp = 0.4
        for z in range(self.heightmap.depth):
            for x in range(self.heightmap.width):
                h = noise3(
                    x * freq + t * 0.3,
                    0.0,
                    z * freq + t * 0.2,
                ) * amp
                self.heightmap.set(x, z, h)
        self.heightmap_mesh = None

    # ── voxel erosion cycle (site_decay) ──────────────────────────────

    def _animate_voxels(self, dt: float) -> None:
        """Erode and rebuild the voxel grid over time."""
        if self.voxels is None:
            return

        # Save original state on first call
        if self._voxel_original_cells is None:
            self._voxel_original_cells = list(self.voxels.cells)

        self._voxel_timer += dt
        step_interval = 0.12 / max(self.anim.speed_scale, 0.1)

        if self._voxel_timer < step_interval:
            return
        self._voxel_timer = 0.0

        if self._voxel_eroding:
            # Remove 1-3 random occupied cells per step
            occupied = [
                i for i, alive in enumerate(self.voxels.cells) if alive
            ]
            n_remove = min(len(occupied), random.randint(1, 3))
            for victim in random.sample(occupied, n_remove):
                self.voxels.cells[victim] = False

            # Switch to rebuild when mostly empty
            if self.voxels.fill_ratio() < 0.08:
                self._voxel_eroding = False
        else:
            # Rebuild: restore a random missing cell
            missing = [
                i for i, alive in enumerate(self.voxels.cells)
                if not alive and self._voxel_original_cells[i]
            ]
            if missing:
                restore = random.choice(missing)
                self.voxels.cells[restore] = True

            # Switch to eroding when mostly full
            if self.voxels.fill_ratio() > 0.65:
                self._voxel_eroding = True

    # ── fragment drift (ruin_state) ───────────────────────────────────

    def _animate_fragments(self, dt: float) -> None:
        """Drift face fragments outward, then snap back."""
        if not self.fragment_groups or self.mesh is None:
            return

        if self._fragment_offsets is None:
            # Initialize drift directions (outward from centroid)
            self._fragment_offsets = []
            for group in self.fragment_groups:
                if not group:
                    self._fragment_offsets.append(Vec3(0, 0, 0))
                    continue
                # Average face centroid for this group
                cx, cy, cz = 0.0, 0.0, 0.0
                count = 0
                for fi in group:
                    if fi < len(self.mesh.faces):
                        face = self.mesh.faces[fi]
                        for vi in face:
                            v = self.mesh.vertices[vi]
                            cx += v.x
                            cy += v.y
                            cz += v.z
                            count += 1
                if count > 0:
                    direction = Vec3(cx / count, cy / count, cz / count).normalized()
                else:
                    direction = Vec3(random.random() - 0.5, random.random() - 0.5, random.random() - 0.5).normalized()
                self._fragment_offsets.append(direction)

        # Cycle: 0→1 = drift out, 1→0 would snap back but we use a sawtooth
        self._fragment_timer += dt

    # ── point cloud animation (abstract_field, atmospheric_depth) ─────

    def _animate_cloud(self, dt: float) -> None:
        """Grow attractor trails / breathe nebulae."""
        if self.cloud is None:
            return

        # For attractor: keep integrating Lorenz system
        if self.cloud.count > 100:  # likely an attractor, not a nebula
            self._grow_attractor(dt)
        else:
            # Nebula: breathing scale
            pass

    def _grow_attractor(self, dt: float) -> None:
        """Add new points to the Lorenz attractor trail."""
        if self.cloud is None or self.cloud.count == 0:
            return

        # Initialize Lorenz state from the last point by reversing the
        # normalization that make_lorenz_attractor applied.
        if self._lorenz_state is None:
            last = self.cloud.points[-1]
            ns = self.cloud.norm_scale
            if ns < 1e-10:
                ns = 0.03  # safe fallback
            inv = 1.0 / ns
            self._lorenz_state = (last.x * inv, last.y * inv, last.z * inv)

        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        ldt = 0.005
        x, y, z = self._lorenz_state
        ns = self.cloud.norm_scale

        # Integrate a few steps per frame
        steps = max(1, int(dt * 200 * self.anim.speed_scale))
        for _ in range(min(steps, 10)):
            dx = sigma * (y - x) * ldt
            dy = (x * (rho - z) - y) * ldt
            dz = (x * y - beta * z) * ldt
            x, y, z = x + dx, y + dy, z + dz

        self._lorenz_state = (x, y, z)

        # Normalize using the same scale as the original cloud
        self.cloud.add(Vec3(x * ns, y * ns, z * ns), bright=1.0)

        # Fade old points
        max_pts = 5000
        if self.cloud.count > max_pts:
            self.cloud.trim(max_pts)
        for i in range(len(self.cloud.brightness)):
            age = 1.0 - (i / max(self.cloud.count, 1))
            self.cloud.brightness[i] = max(0.1, 1.0 - age * 0.9)

    # ── lifecycle ─────────────────────────────────────────────────────

    def start_transition(self) -> None:
        """Begin the dissolve → tesseract → form sequence."""
        self.transition.start()
        # Reset per-geometry animation state for the outgoing geometry
        self._fragment_offsets = None
        self._fragment_timer = 0.0
        self._fragment_cache = None
        self._lorenz_state = None
        self._voxel_timer = 0.0
        self._voxel_eroding = True
        self._voxel_original_cells = None

    def capture_transition_source(self) -> None:
        """Freeze the current renderable form for the dissolve phase."""
        self.transition_source = SceneSnapshot(
            geom_kind=self.geom_kind,
            mesh=copy.deepcopy(self.mesh),
            mesh_b=copy.deepcopy(self.mesh_b),
            cloud=copy.deepcopy(self.cloud),
            voxels=copy.deepcopy(self.voxels),
            heightmap=copy.deepcopy(self.heightmap),
            heightmap_mesh=copy.deepcopy(self.heightmap_mesh),
            shader_chars=self.shader_chars,
            light=copy.deepcopy(self.light),
            camera=copy.deepcopy(self.camera),
            styles=self.styles,
            fragment_groups=copy.deepcopy(self.fragment_groups),
            dual_mesh_mode=self.dual_mesh_mode,
            surface_sampler=self.surface_sampler,  # immutable sample caches are safe to share
        )

    def _render_snapshot(
        self,
        rast: AsciiRasterizer,
        w: int,
        h: int,
        snapshot: SceneSnapshot,
    ) -> None:
        self._render_geometry_state(
            rast,
            w,
            h,
            snapshot.geom_kind,
            snapshot.mesh,
            snapshot.mesh_b,
            snapshot.cloud,
            snapshot.voxels,
            snapshot.heightmap,
            snapshot.heightmap_mesh,
            snapshot.shader_chars,
            snapshot.light,
            snapshot.styles,
            snapshot.camera,
            snapshot.fragment_groups,
            snapshot.dual_mesh_mode,
            snapshot.surface_sampler,
        )

    def _mesh_for_render(
        self,
        mesh: Optional[Mesh],
        fragment_groups: list[list[int]],
    ) -> Optional[Mesh]:
        if mesh is None:
            return None
        if fragment_groups and self._fragment_offsets is not None:
            drift = self._fragment_drift_amount(fragment_groups)
            if drift > 0.0:
                return self._build_fragment_mesh(mesh, fragment_groups, drift)
        return mesh

    def _fragment_drift_amount(self, fragment_groups: list[list[int]]) -> float:
        if not fragment_groups:
            return 0.0
        cycle_t = (self._fragment_timer % self._fragment_cycle) / self._fragment_cycle
        if cycle_t >= 0.8:
            return 0.0
        return (cycle_t / 0.8) ** 2 * 1.5

    def _build_fragment_mesh(
        self,
        mesh: Mesh,
        fragment_groups: list[list[int]],
        drift: float,
    ) -> Mesh:
        if self._fragment_offsets is None:
            return mesh

        cache = self._fragment_cache_for(mesh, fragment_groups)
        if cache is None:
            return mesh

        scaled_offsets = [offset * drift for offset in self._fragment_offsets]
        vertices = [
            Vec3(
                base.x + scaled_offsets[group_index].x,
                base.y + scaled_offsets[group_index].y,
                base.z + scaled_offsets[group_index].z,
            )
            for base, group_index in zip(cache.base_vertices, cache.vertex_group_indices)
        ]
        return Mesh(
            vertices=vertices,
            faces=cache.faces,
            normals=cache.normals,
            vertex_normals=cache.vertex_normals,
        )

    def _fragment_cache_for(
        self,
        mesh: Mesh,
        fragment_groups: list[list[int]],
    ) -> FragmentRenderCache | None:
        group_signature = tuple(tuple(group) for group in fragment_groups)
        cache = self._fragment_cache
        if (
            cache is not None
            and cache.source_mesh is mesh
            and cache.group_signature == group_signature
        ):
            return cache

        vertices: list[Vec3] = []
        vertex_group_indices: list[int] = []
        faces: list[tuple[int, ...]] = []

        for group_index, group in enumerate(fragment_groups):
            for face_index in group:
                if face_index >= len(mesh.faces):
                    continue
                face = mesh.faces[face_index]
                render_face: list[int] = []
                for vertex_index in face:
                    vertices.append(mesh.vertices[vertex_index])
                    vertex_group_indices.append(group_index)
                    render_face.append(len(vertices) - 1)
                faces.append(tuple(render_face))

        if not faces:
            return None

        exploded = Mesh(vertices=vertices, faces=faces)
        exploded.compute_normals()
        cache = FragmentRenderCache(
            source_mesh=mesh,
            group_signature=group_signature,
            base_vertices=tuple(vertices),
            vertex_group_indices=tuple(vertex_group_indices),
            faces=faces,
            normals=exploded.normals,
            vertex_normals=exploded.vertex_normals,
        )
        self._fragment_cache = cache
        return cache

    def _morph_mesh(self, mesh_a: Mesh, mesh_b: Mesh, t: float) -> Mesh:
        if mesh_a.vertex_count != mesh_b.vertex_count:
            return mesh_a
        morphed = Mesh(
            vertices=[
                a.lerp(b, smoothstep(0.0, 1.0, t))
                for a, b in zip(mesh_a.vertices, mesh_b.vertices)
            ],
            faces=list(mesh_a.faces),
        )
        morphed.compute_normals()
        return morphed
