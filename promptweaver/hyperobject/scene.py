"""Scene graph — orchestrates geometry, camera, lighting, and animation.

A Scene holds everything needed to render one frame: the geometry (which
changes per template), the camera, the light, and the animation state.
The PromptInterpreter builds scenes from GeneratedPrompt data.
"""

from __future__ import annotations

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
from .geometry import Mesh, PointCloud, VoxelGrid, HeightMap
from .rasterizer import AsciiRasterizer, CharGrid, Light, DEFAULT_LIGHT
from .transform import Camera, ProjectionContext


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
    total_frames: int = 36  # ~2 seconds at 18fps
    current_frame: int = 0

    # Phase boundaries (as fraction of total)
    dissolve_end: float = 0.25
    tesseract_end: float = 0.55
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

    # Tesseract (always available for transitions)
    tesseract_verts: list[Vec4] = field(default_factory=list)
    tesseract_edges: list[tuple[int, int]] = field(default_factory=list)

    # Rendering config
    shader_chars: str = " ░▒▓█░▒▓██"
    light: Light = field(default_factory=lambda: DEFAULT_LIGHT)
    camera: Camera = field(default_factory=lambda: Camera(
        position=Vec3(0.0, 0.5, 3.5),
        target=Vec3(0.0, 0.0, 0.0),
    ))

    # Palette styles (bright, primary, mid, dim)
    styles: tuple[str, str, str, str] = (
        "bright_green", "green", "#006600", "#003300",
    )

    # Animation
    anim: AnimationState = field(default_factory=AnimationState)
    transition: TransitionState = field(default_factory=TransitionState)

    # Fragment groups for ruin_state dissolve
    fragment_groups: list[list[int]] = field(default_factory=list)

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

    def _render_geometry(self, rast: AsciiRasterizer, w: int, h: int) -> None:
        """Render the active geometry."""
        model = self.model_matrix()
        ctx = ProjectionContext.build(model, self.camera, w, h)

        if self.geom_kind == GeomKind.MESH_FILLED:
            if self.mesh:
                rast.draw_mesh_filled(
                    self.mesh, ctx, self.shader_chars, self.light, self.styles,
                )

        elif self.geom_kind == GeomKind.MESH_WIREFRAME:
            if self.mesh:
                rast.draw_mesh_wireframe(
                    self.mesh, ctx,
                    edge_char="·",
                    styles=self.styles,
                    vertex_char="•",
                )

        elif self.geom_kind == GeomKind.POINT_CLOUD:
            if self.cloud:
                rast.draw_points(self.cloud, ctx, styles=self.styles)

        elif self.geom_kind == GeomKind.VOXEL_GRID:
            if self.voxels:
                rast.draw_voxels(self.voxels, ctx, styles=self.styles)

        elif self.geom_kind == GeomKind.HEIGHTMAP:
            if self.heightmap and self.heightmap_mesh is None:
                self.heightmap_mesh = self.heightmap.to_mesh()
            if self.heightmap_mesh:
                rast.draw_heightmap(
                    self.heightmap_mesh, ctx,
                    self.shader_chars, self.light, self.styles,
                )

        elif self.geom_kind == GeomKind.TESSERACT:
            self._render_tesseract(rast, w, h)

        elif self.geom_kind == GeomKind.DUAL_MESH:
            if self.mesh:
                rast.draw_mesh_filled(
                    self.mesh, ctx, self.shader_chars, self.light, self.styles,
                )
            if self.mesh_b:
                # Second mesh with slightly different rotation
                model_b = Mat4.rotation_y(self.anim.angle_y * 1.618)
                model_b = model_b @ Mat4.rotation_x(self.anim.angle_x * 0.7)
                ctx_b = ProjectionContext.build(model_b, self.camera, w, h)
                rast.draw_mesh_filled(
                    self.mesh_b, ctx_b, self.shader_chars, self.light, self.styles,
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
            self._render_geometry(rast, w, h)
            # Apply dissolve: randomly blank cells based on progress
            threshold = t * t  # accelerating
            for i, cell in enumerate(rast.grid.cells):
                if cell.char != " " and random.random() < threshold:
                    cell.char = " "
                    cell.style = ""

        elif phase == TransitionPhase.TESSERACT:
            # Render tesseract with fade-in/fade-out
            self._render_tesseract(rast, w, h)
            # Fade edges based on phase progress
            fade = 1.0 - abs(t - 0.5) * 2.0  # peak at t=0.5
            if fade < 0.8:
                threshold = 1.0 - fade
                for i, cell in enumerate(rast.grid.cells):
                    if cell.char != " " and random.random() < threshold * 0.5:
                        cell.char = " "
                        cell.style = ""

        elif phase == TransitionPhase.FORM:
            # Render new geometry with progressive reveal
            self._render_geometry(rast, w, h)
            threshold = 1.0 - smoothstep(0.0, 1.0, t)
            for i, cell in enumerate(rast.grid.cells):
                if cell.char != " " and random.random() < threshold:
                    cell.char = " "
                    cell.style = ""

    # ── animation tick ────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Advance the scene by dt seconds."""
        self.anim.tick(dt)
        if self.transition.active:
            self.transition.tick()

    def start_transition(self) -> None:
        """Begin the dissolve → tesseract → form sequence."""
        self.transition.start()
