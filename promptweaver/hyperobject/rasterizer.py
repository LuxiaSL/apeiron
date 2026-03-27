"""Z-buffered ASCII rasterizer.

Converts 3D geometry into a character grid using depth-tested rendering.
Supports four modes: filled faces (Gouraud-shaded), wireframe edges,
point clouds, and donut.c-style direct surface sampling.
Produces a CharGrid that can be post-processed and converted to Rich Text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol

from rich.text import Text

from .lut import Vec3, clamp, lerp_f
from .geometry import Mesh, PointCloud, VoxelGrid
from .shaders import SHADER_PRESETS
from .transform import Camera, ProjectionContext, ScreenPoint, bresenham


# donut.c-style 12-character luminance ramp (dark → bright).
# Used as the default when no shader chars are supplied.
DONUT_LUMINANCE_RAMP: str = " .,-~:;=!*#$@"


# ── surface sampler protocol (for donut.c-style rendering) ─────────────


class SurfaceSampler(Protocol):
    """Protocol for parametric surfaces that can be directly sampled.

    Implementations yield (position, normal) pairs for every sample
    point on the surface.  The rasterizer projects each point and
    computes per-point brightness — the core donut.c technique.
    """

    def samples(self) -> Iterator[tuple[Vec3, Vec3]]:
        """Yield (world_position, surface_normal) for each sample."""
        ...


class TorusSampler:
    """donut.c-style torus sampler.

    Sweeps θ (tube angle) and φ (ring angle) and yields position +
    analytically-computed surface normal at each sample.  Dense
    sampling means every character cell gets unique shading.
    """

    def __init__(
        self,
        R: float = 1.0,
        r: float = 0.5,
        theta_step: float = 0.09,
        phi_step: float = 0.03,
    ) -> None:
        self.R = R
        self.r = r
        self.theta_step = theta_step
        self.phi_step = phi_step

    def samples(self) -> Iterator[tuple[Vec3, Vec3]]:
        R, r = self.R, self.r
        TWO_PI = 2.0 * math.pi
        theta = 0.0
        while theta < TWO_PI:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            phi = 0.0
            while phi < TWO_PI:
                cos_p = math.cos(phi)
                sin_p = math.sin(phi)

                # Point on torus surface
                ring = R + r * cos_t
                x = ring * cos_p
                z = ring * sin_p
                y = r * sin_t

                # Surface normal (outward from tube center)
                nx = cos_t * cos_p
                nz = cos_t * sin_p
                ny = sin_t

                yield Vec3(x, y, z), Vec3(nx, ny, nz)
                phi += self.phi_step
            theta += self.theta_step


class SphereSampler:
    """Direct sphere sampler with analytical normals."""

    def __init__(
        self,
        radius: float = 1.0,
        theta_step: float = 0.07,
        phi_step: float = 0.04,
    ) -> None:
        self.radius = radius
        self.theta_step = theta_step
        self.phi_step = phi_step

    def samples(self) -> Iterator[tuple[Vec3, Vec3]]:
        r = self.radius
        PI = math.pi
        TWO_PI = 2.0 * PI
        theta = 0.0
        while theta < PI:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            phi = 0.0
            while phi < TWO_PI:
                cos_p = math.cos(phi)
                sin_p = math.sin(phi)

                nx = sin_t * cos_p
                ny = cos_t
                nz = sin_t * sin_p

                yield Vec3(nx * r, ny * r, nz * r), Vec3(nx, ny, nz)
                phi += self.phi_step
            theta += self.theta_step


class MobiusSampler:
    """Direct Möbius strip sampler with numerical normals.

    The Möbius strip is non-orientable, so surface normals flip across
    the surface — this is correct and gives the strip its characteristic
    shading pattern.
    """

    def __init__(
        self,
        u_step: float = 0.05,
        v_steps: int = 14,
        v_min: float = -0.4,
        v_max: float = 0.4,
    ) -> None:
        self.u_step = u_step
        self.v_steps = v_steps
        self.v_min = v_min
        self.v_max = v_max
        # Pre-compute normalization scale (match make_mobius_strip)
        self._scale = self._compute_scale()

    def _compute_scale(self) -> float:
        """Find bounding radius and compute scale to radius ~1.0."""
        max_r = 0.0
        TWO_PI = 2.0 * math.pi
        v_range = self.v_max - self.v_min
        u = 0.0
        while u < TWO_PI:
            for j in range(self.v_steps + 1):
                v = self.v_min + v_range * j / self.v_steps
                pt = self._raw_point(u, v)
                r = pt.length()
                if r > max_r:
                    max_r = r
            u += self.u_step
        return 1.0 / max_r if max_r > 1e-10 else 1.0

    def _raw_point(self, u: float, v: float) -> Vec3:
        half_v = v / 2.0
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_u2 = math.cos(u / 2.0)
        sin_u2 = math.sin(u / 2.0)
        x = (1.0 + half_v * cos_u2) * cos_u
        y = (1.0 + half_v * cos_u2) * sin_u
        z = half_v * sin_u2
        return Vec3(x, y, z)

    def samples(self) -> Iterator[tuple[Vec3, Vec3]]:
        TWO_PI = 2.0 * math.pi
        eps = 0.01
        v_range = self.v_max - self.v_min
        scale = self._scale

        u = 0.0
        while u < TWO_PI:
            for j in range(self.v_steps + 1):
                v = self.v_min + v_range * j / self.v_steps
                pt = self._raw_point(u, v) * scale

                # Numerical normal via cross product of partial derivatives
                du = self._raw_point(u + eps, v) - self._raw_point(u - eps, v)
                dv = self._raw_point(u, v + eps) - self._raw_point(u, v - eps)
                normal = du.cross(dv).normalized()

                yield pt, normal
            u += self.u_step


# ── character grid (shared output type) ─────────────────────────────────


@dataclass(slots=True)
class Cell:
    """Single character cell in the render output."""

    char: str = " "
    style: str = ""
    depth: float = 1.0  # for z-ordering in compositing


@dataclass
class CharGrid:
    """2D grid of character cells — the render target.

    Flat array, row-major: index = row * width + col.
    """

    width: int
    height: int
    cells: list[Cell] = field(default_factory=list)
    zbuf: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.width * self.height
        if not self.cells:
            self.cells = [Cell() for _ in range(n)]
        if not self.zbuf:
            self.zbuf = [1.0] * n

    def _idx(self, col: int, row: int) -> int:
        return row * self.width + col

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def get(self, col: int, row: int) -> Cell:
        if not self.in_bounds(col, row):
            return Cell()
        return self.cells[self._idx(col, row)]

    def set(self, col: int, row: int, cell: Cell) -> None:
        if self.in_bounds(col, row):
            self.cells[self._idx(col, row)] = cell

    def write(self, col: int, row: int, char: str, style: str, depth: float) -> bool:
        """Write a character if it passes the z-test. Returns True if written."""
        if not self.in_bounds(col, row):
            return False
        idx = self._idx(col, row)
        if depth < self.zbuf[idx]:
            self.zbuf[idx] = depth
            cell = self.cells[idx]
            cell.char = char
            cell.style = style
            cell.depth = depth
            return True
        return False

    def clear(self) -> None:
        """Reset to blank state."""
        for cell in self.cells:
            cell.char = " "
            cell.style = ""
            cell.depth = 1.0
        for i in range(len(self.zbuf)):
            self.zbuf[i] = 1.0

    def to_rich_text(self) -> Text:
        """Convert the grid to a Rich Text object for Textual rendering."""
        text = Text()
        append = text.append
        width = self.width
        for row in range(self.height):
            base = row * width
            col = 0
            while col < width:
                cell = self.cells[base + col]
                style = cell.style
                chars = [cell.char]
                col += 1
                while col < width:
                    next_cell = self.cells[base + col]
                    if next_cell.style != style:
                        break
                    chars.append(next_cell.char)
                    col += 1
                append("".join(chars), style=style or None)
            if row < self.height - 1:
                append("\n")
        return text


# ── style mapping ──────────────────────────────────────────────────────


def depth_to_style(
    depth: float,
    bright: str,
    primary: str,
    mid: str,
    dim: str,
) -> str:
    """Map normalized depth [0, 1] to one of four palette styles.

    Legacy function kept for wireframe/voxel/point renderers where
    brightness isn't computed per-pixel.
    """
    if depth < 0.35:
        return bright
    elif depth < 0.55:
        return primary
    elif depth < 0.75:
        return mid
    else:
        return dim


def brightness_to_style(
    brightness: float,
    bright: str,
    primary: str,
    mid: str,
    dim: str,
) -> str:
    """Map brightness [0, 1] to one of four palette styles.

    Bright surfaces get bright colors regardless of depth, making
    objects look vivid rather than washed out. Thresholds are tuned
    so that the distribution is roughly even across the four bands
    for typical Lambert-lit geometry with ambient.
    """
    if brightness > 0.50:
        return bright
    elif brightness > 0.28:
        return primary
    elif brightness > 0.10:
        return mid
    else:
        return dim


# ── lighting ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Light:
    """Directional light for face shading."""

    direction: Vec3  # normalized, points FROM light TO scene
    intensity: float = 1.0
    wrap: float = 0.0  # 0 = hard Lambert, 0.5 = subsurface wrap
    ambient: float = 0.18  # minimum brightness so dark sides are clearly visible

    def shade(self, normal: Vec3) -> float:
        """Compute brightness for a face with the given normal.

        ``direction`` points FROM light TO scene, so we negate for the
        standard Lambert N·L where L points FROM surface TO light.
        ``ambient`` provides a floor so shapes remain visible even on
        the unlit side.
        """
        ndotl = -(normal.dot(self.direction))
        if self.wrap > 0.0:
            brightness = (ndotl + self.wrap) / (1.0 + self.wrap)
        else:
            brightness = max(0.0, ndotl)
        # Mix diffuse with ambient
        result = self.ambient + (1.0 - self.ambient) * brightness * self.intensity
        return clamp(result, 0.0, 1.0)


DEFAULT_LIGHT = Light(
    direction=Vec3(0.3, -0.8, 0.5).normalized(),
    intensity=1.2,
)


# ── rasterizer ──────────────────────────────────────────────────────────


class AsciiRasterizer:
    """The core rendering engine.

    Usage:
        rast = AsciiRasterizer(width=80, height=24)
        ctx = ProjectionContext.build(model, camera, width, height)
        rast.clear()
        rast.draw_mesh_filled(mesh, ctx, shader_chars, light, styles)
        rast.draw_mesh_wireframe(mesh, ctx, edge_char, style)
        rast.draw_points(cloud, ctx, point_chars, styles)
        grid = rast.grid
    """

    def __init__(self, width: int, height: int) -> None:
        self.grid = CharGrid(width=width, height=height)

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    def resize(self, width: int, height: int) -> None:
        """Resize the render target (creates new grid)."""
        if width != self.grid.width or height != self.grid.height:
            self.grid = CharGrid(width=width, height=height)

    def clear(self) -> None:
        self.grid.clear()

    # ── filled face rendering ─────────────────────────────────────────

    def draw_mesh_filled(
        self,
        mesh: Mesh,
        ctx: ProjectionContext,
        shader_chars: str,
        light: Light,
        styles: tuple[str, str, str, str],
    ) -> None:
        """Render filled faces with per-pixel Gouraud shading.

        Interpolates vertex normals across each triangle so every
        character cell gets its own brightness — the key technique that
        makes donut.c look so good.

        shader_chars: brightness ramp (dark→light), e.g. " .,-~:;=!*#$@"
        styles: (bright, primary, mid, dim) palette style strings
        """
        if not mesh.normals:
            mesh.compute_normals()
        has_vnormals = bool(mesh.vertex_normals)

        n_chars = len(shader_chars)
        bright_s, primary_s, mid_s, dim_s = styles

        # Project all vertices once
        projected: list[tuple[float, float, float] | None] = [
            ctx.project_vertex_unclamped(v) for v in mesh.vertices
        ]

        # Pre-compute per-vertex brightness for Gouraud interpolation
        vert_brightness: list[float] = []
        if has_vnormals:
            for vn in mesh.vertex_normals:
                view_n = ctx.transform_normal(vn)
                vert_brightness.append(light.shade(view_n))
        else:
            vert_brightness = [0.5] * len(mesh.vertices)

        for fi, face in enumerate(mesh.faces):
            # Skip faces where any vertex is behind camera
            face_proj = [projected[vi] for vi in face]
            if any(p is None for p in face_proj):
                continue

            # Type narrow — we know none are None after the check
            pts = [p for p in face_proj if p is not None]
            if len(pts) < 3:
                continue

            # Backface cull (screen-space winding order)
            ax, ay, _ = pts[0]
            bx, by, _ = pts[1]
            cx, cy, _ = pts[2]
            cross_z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross_z > 0:
                continue  # backfacing

            # Gather per-vertex brightness for this face
            face_bright = [vert_brightness[vi] for vi in face]

            # Rasterize triangulated face with per-pixel shading
            for ti in range(1, len(pts) - 1):
                self._fill_triangle_gouraud(
                    pts[0], pts[ti], pts[ti + 1],
                    face_bright[0], face_bright[ti], face_bright[ti + 1],
                    shader_chars, n_chars, styles,
                )

    def _fill_triangle_gouraud(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        bright_a: float,
        bright_b: float,
        bright_c: float,
        shader_chars: str,
        n_chars: int,
        styles: tuple[str, str, str, str],
    ) -> None:
        """Scan-convert a triangle with per-pixel brightness interpolation.

        Each character cell gets its own brightness from barycentric
        interpolation of vertex brightnesses, producing smooth gradients
        instead of flat-colored faces.
        """
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c

        bright_s, primary_s, mid_s, dim_s = styles
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if -1e-10 < area < 1e-10:
            return
        orient = 1.0 if area > 0.0 else -1.0
        inv_area = 1.0 / abs(area)

        # Bounding box (clamped to grid)
        min_col = max(0, int(min(ax, bx, cx)))
        max_col = min(self.width - 1, int(max(ax, bx, cx)) + 1)
        min_row = max(0, int(min(ay, by, cy)))
        max_row = min(self.height - 1, int(max(ay, by, cy)) + 1)

        # Skip tiny or off-screen triangles
        if min_col > max_col or min_row > max_row:
            return

        width = self.width
        zbuf = self.grid.zbuf
        cells = self.grid.cells
        shader_last = n_chars - 1
        eps = -0.001

        e0_dx = (by - cy) * orient
        e0_dy = (cx - bx) * orient
        e1_dx = (cy - ay) * orient
        e1_dy = (ax - cx) * orient
        e2_dx = (ay - by) * orient
        e2_dy = (bx - ax) * orient

        start_x = min_col + 0.5
        start_y = min_row + 0.5
        row_e0 = ((cx - bx) * (start_y - by) - (cy - by) * (start_x - bx)) * orient
        row_e1 = ((ax - cx) * (start_y - cy) - (ay - cy) * (start_x - cx)) * orient
        row_e2 = ((bx - ax) * (start_y - ay) - (by - ay) * (start_x - ax)) * orient

        for row in range(min_row, max_row + 1):
            e0 = row_e0
            e1 = row_e1
            e2 = row_e2
            idx = row * width + min_col
            for col in range(min_col, max_col + 1):
                if e0 >= eps and e1 >= eps and e2 >= eps:
                    u = e0 * inv_area
                    v = e1 * inv_area
                    w = 1.0 - u - v

                    depth = u * az + v * bz + w * cz
                    if depth < zbuf[idx]:
                        brightness = u * bright_a + v * bright_b + w * bright_c
                        char_idx = int(brightness * shader_last)
                        if char_idx < 0:
                            char_idx = 0
                        elif char_idx > shader_last:
                            char_idx = shader_last
                        char = shader_chars[char_idx]
                        if char != " ":
                            style_b = brightness ** 0.55
                            style = brightness_to_style(
                                style_b, bright_s, primary_s, mid_s, dim_s
                            )
                            zbuf[idx] = depth
                            cell = cells[idx]
                            cell.char = char
                            cell.style = style
                            cell.depth = depth
                idx += 1
                e0 += e0_dx
                e1 += e1_dx
                e2 += e2_dx
            row_e0 += e0_dy
            row_e1 += e1_dy
            row_e2 += e2_dy

    # ── wireframe rendering ───────────────────────────────────────────

    def draw_mesh_wireframe(
        self,
        mesh: Mesh,
        ctx: ProjectionContext,
        edge_char: str = "·",
        styles: tuple[str, str, str, str] = ("bright_white", "white", "grey70", "grey50"),
        vertex_char: str = "",
    ) -> None:
        """Render mesh edges as lines, optionally with vertex dots."""
        bright_s, primary_s, mid_s, dim_s = styles

        projected: list[ScreenPoint | None] = [
            ctx.project_vertex(v) for v in mesh.vertices
        ]

        # Draw edges
        if not mesh.edges:
            mesh.compute_edges_from_faces()

        for i0, i1 in mesh.edges:
            p0, p1 = projected[i0], projected[i1]
            if p0 is None or p1 is None:
                continue

            points = bresenham(p0.col, p0.row, p1.col, p1.row)
            n = max(len(points) - 1, 1)
            for pi, (col, row) in enumerate(points):
                t = pi / n
                depth = lerp_f(p0.depth, p1.depth, t)
                style = depth_to_style(depth, bright_s, primary_s, mid_s, dim_s)
                self.grid.write(col, row, edge_char, style, depth)

        # Draw vertices
        if vertex_char:
            for sp in projected:
                if sp is not None:
                    style = depth_to_style(sp.depth, bright_s, primary_s, mid_s, dim_s)
                    self.grid.write(sp.col, sp.row, vertex_char, style, sp.depth)

    # ── point cloud rendering ─────────────────────────────────────────

    def draw_points(
        self,
        cloud: PointCloud,
        ctx: ProjectionContext,
        point_chars: str = "·∙•●",
        styles: tuple[str, str, str, str] = ("bright_white", "white", "grey70", "grey50"),
    ) -> None:
        """Render a point cloud with depth+brightness-based characters."""
        bright_s, primary_s, mid_s, dim_s = styles
        n_chars = len(point_chars)

        for i, pt in enumerate(cloud.points):
            sp = ctx.project_vertex(pt)
            if sp is None:
                continue

            b = cloud.brightness[i] if i < len(cloud.brightness) else 0.5
            # Combine brightness with inverse depth for character selection
            combined = b * (1.0 - sp.depth * 0.5)
            char_idx = int(combined * (n_chars - 1))
            char_idx = max(0, min(char_idx, n_chars - 1))
            char = point_chars[char_idx]

            style = depth_to_style(sp.depth, bright_s, primary_s, mid_s, dim_s)
            self.grid.write(sp.col, sp.row, char, style, sp.depth)

    # ── voxel grid rendering ──────────────────────────────────────────

    def draw_voxels(
        self,
        voxels: VoxelGrid,
        ctx: ProjectionContext,
        block_char: str = "█",
        styles: tuple[str, str, str, str] = ("bright_white", "white", "grey70", "grey50"),
    ) -> None:
        """Render a voxel grid as filled blocks."""
        bright_s, primary_s, mid_s, dim_s = styles

        for z in range(voxels.size_z):
            for y in range(voxels.size_y):
                for x in range(voxels.size_x):
                    if not voxels.get(x, y, z):
                        continue
                    center = voxels.cell_center(x, y, z)
                    sp = ctx.project_vertex(center)
                    if sp is None:
                        continue
                    style = depth_to_style(sp.depth, bright_s, primary_s, mid_s, dim_s)
                    self.grid.write(sp.col, sp.row, block_char, style, sp.depth)

    # ── tesseract wireframe (4D → 3D → 2D) ───────────────────────────

    def draw_tesseract_wireframe(
        self,
        vertices_3d: list[Vec3],
        edges: list[tuple[int, int]],
        ctx: ProjectionContext,
        edge_char: str = "─",
        vertex_char: str = "●",
        styles: tuple[str, str, str, str] = ("bright_white", "white", "grey70", "grey50"),
    ) -> None:
        """Render a pre-projected tesseract (4D→3D already done, we do 3D→2D)."""
        bright_s, primary_s, mid_s, dim_s = styles

        projected: list[ScreenPoint | None] = [
            ctx.project_vertex(v) for v in vertices_3d
        ]

        # Edges
        for i0, i1 in edges:
            p0, p1 = projected[i0], projected[i1]
            if p0 is None or p1 is None:
                continue
            points = bresenham(p0.col, p0.row, p1.col, p1.row)
            n = max(len(points) - 1, 1)
            for pi, (col, row) in enumerate(points):
                t = pi / n
                depth = lerp_f(p0.depth, p1.depth, t)
                style = depth_to_style(depth, bright_s, primary_s, mid_s, dim_s)
                self.grid.write(col, row, edge_char, style, depth)

        # Vertices (bright dots)
        for sp in projected:
            if sp is not None:
                self.grid.write(sp.col, sp.row, vertex_char, bright_s, sp.depth * 0.9)

    # ── heightmap surface rendering ───────────────────────────────────

    def draw_heightmap(
        self,
        mesh: Mesh,
        ctx: ProjectionContext,
        shader_chars: str,
        light: Light,
        styles: tuple[str, str, str, str],
    ) -> None:
        """Render a heightmap-derived mesh. Delegates to filled face rendering."""
        self.draw_mesh_filled(mesh, ctx, shader_chars, light, styles)

    # ── donut.c-style direct surface renderer ─────────────────────────

    def draw_surface_direct(
        self,
        surface: SurfaceSampler,
        ctx: ProjectionContext,
        light: Light,
        styles: tuple[str, str, str, str],
        luminance_ramp: str = DONUT_LUMINANCE_RAMP,
    ) -> None:
        """Render a parametric surface by direct point sampling.

        This is the donut.c technique: instead of meshing and triangle-
        rasterizing, sample the surface densely and plot each point as a
        single character whose brightness is computed from the surface
        normal at that exact point.  Uses depth buffering for correct
        ordering.

        The hot loop is manually inlined for performance — extracting
        the MVP matrix elements once and avoiding object creation per
        sample point.
        """
        bright_s, primary_s, mid_s, dim_s = styles
        n_chars = len(luminance_ramp)
        w, h = self.width, self.height
        max_col = max(w - 1, 0)
        max_row = max(h - 1, 0)

        m0, m1, m2, m3 = ctx.m0, ctx.m1, ctx.m2, ctx.m3
        m4, m5, m6, m7 = ctx.m4, ctx.m5, ctx.m6, ctx.m7
        m8, m9, m10, m11 = ctx.m8, ctx.m9, ctx.m10, ctx.m11
        m12, m13, m14, m15 = ctx.m12, ctx.m13, ctx.m14, ctx.m15
        nm0, nm1, nm2 = ctx.nm0, ctx.nm1, ctx.nm2
        nm4, nm5, nm6 = ctx.nm4, ctx.nm5, ctx.nm6
        nm8, nm9, nm10 = ctx.nm8, ctx.nm9, ctx.nm10

        # Light direction (pre-negate for Lambert)
        ldx = -light.direction.x
        ldy = -light.direction.y
        ldz = -light.direction.z
        l_intensity = light.intensity
        l_wrap = light.wrap
        l_ambient = light.ambient

        zbuf = self.grid.zbuf
        cells = self.grid.cells

        for pos, normal in surface.samples():
            px, py, pz = pos.x, pos.y, pos.z

            # MVP transform (inlined, w=1)
            clip_w = m12 * px + m13 * py + m14 * pz + m15
            if clip_w <= 0.0:
                continue

            inv_w = 1.0 / clip_w
            ndc_x = (m0 * px + m1 * py + m2 * pz + m3) * inv_w
            ndc_y = (m4 * px + m5 * py + m6 * pz + m7) * inv_w

            # Quick frustum cull
            if ndc_x < -1.2 or ndc_x > 1.2 or ndc_y < -1.2 or ndc_y > 1.2:
                continue

            # NDC → screen (inlined)
            col = int((ndc_x * 0.5 + 0.5) * max_col + 0.5)
            row = int(((1.0 - ndc_y) * 0.5) * max_row + 0.5)

            if col < 0 or col >= w or row < 0 or row >= h:
                continue

            # Depth for z-buffer
            ndc_z = (m8 * px + m9 * py + m10 * pz + m11) * inv_w
            depth = (ndc_z + 1.0) * 0.5
            if depth < 0.0:
                depth = 0.0
            elif depth > 1.0:
                depth = 1.0

            idx = row * w + col
            if depth >= zbuf[idx]:
                continue

            # Normal transform (inlined model.transform_direction + normalize)
            nx, ny, nz = normal.x, normal.y, normal.z
            wnx = nm0 * nx + nm1 * ny + nm2 * nz
            wny = nm4 * nx + nm5 * ny + nm6 * nz
            wnz = nm8 * nx + nm9 * ny + nm10 * nz
            inv_len = wnx * wnx + wny * wny + wnz * wnz
            if inv_len > 1e-20:
                inv_len = 1.0 / (inv_len ** 0.5)
                wnx *= inv_len
                wny *= inv_len
                wnz *= inv_len

            # Lambert shading (inlined light.shade)
            ndotl = wnx * ldx + wny * ldy + wnz * ldz
            if l_wrap > 0.0:
                brightness = (ndotl + l_wrap) / (1.0 + l_wrap)
            else:
                brightness = ndotl if ndotl > 0.0 else 0.0
            brightness = l_ambient + (1.0 - l_ambient) * brightness * l_intensity
            if brightness < 0.0:
                brightness = 0.0
            elif brightness > 1.0:
                brightness = 1.0

            char_idx = int(brightness * (n_chars - 1))
            if char_idx < 0:
                char_idx = 0
            elif char_idx >= n_chars:
                char_idx = n_chars - 1
            char = luminance_ramp[char_idx]

            if char == " ":
                continue

            # Gamma-lifted brightness for style selection (pushes more
            # cells into brighter color bands)
            style_b = brightness ** 0.55
            if style_b > 0.50:
                style = bright_s
            elif style_b > 0.28:
                style = primary_s
            elif style_b > 0.10:
                style = mid_s
            else:
                style = dim_s

            zbuf[idx] = depth
            cell = cells[idx]
            cell.char = char
            cell.style = style
            cell.depth = depth

    # ── composite helper ──────────────────────────────────────────────

    def overlay(self, other: CharGrid) -> None:
        """Composite another grid on top (non-space characters only)."""
        for i, cell in enumerate(other.cells):
            if cell.char != " " and cell.depth < self.grid.zbuf[i]:
                self.grid.cells[i] = cell
                self.grid.zbuf[i] = cell.depth
