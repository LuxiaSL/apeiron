"""Z-buffered ASCII rasterizer.

Converts 3D geometry into a character grid using depth-tested rendering.
Supports four modes: filled faces (Gouraud-shaded), wireframe edges,
point clouds, and donut.c-style direct surface sampling.
Produces a CharGrid that can be post-processed and converted to Rich Text.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol

from rich.text import Text

from .lut import Vec3, clamp
from .geometry import Mesh, PointCloud, VoxelGrid
from .shaders import SHADER_PRESETS
from .transform import Camera, ProjectionContext, ScreenPoint

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)


# donut.c-style 12-character luminance ramp (dark → bright).
# Used as the default when no shader chars are supplied.
DONUT_LUMINANCE_RAMP: str = " .,-~:;=!*#$@"
_STYLE_BRIGHT_CUTOFF: float = 0.50 ** (1.0 / 0.55)
_STYLE_PRIMARY_CUTOFF: float = 0.28 ** (1.0 / 0.55)
_STYLE_MID_CUTOFF: float = 0.10 ** (1.0 / 0.55)


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
        self._samples = self._build_samples()
        self._np_positions: object = None  # np.ndarray | None (lazy)
        self._np_normals: object = None  # np.ndarray | None (lazy)

    def _build_samples(self) -> tuple[tuple[Vec3, Vec3], ...]:
        samples: list[tuple[Vec3, Vec3]] = []
        R, r = self.R, self.r
        two_pi = 2.0 * math.pi
        theta = 0.0
        while theta < two_pi:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            phi = 0.0
            while phi < two_pi:
                cos_p = math.cos(phi)
                sin_p = math.sin(phi)

                ring = R + r * cos_t
                x = ring * cos_p
                z = ring * sin_p
                y = r * sin_t

                nx = cos_t * cos_p
                nz = cos_t * sin_p
                ny = sin_t

                samples.append((Vec3(x, y, z), Vec3(nx, ny, nz)))
                phi += self.phi_step
            theta += self.theta_step
        return tuple(samples)

    def samples(self) -> Iterator[tuple[Vec3, Vec3]]:
        return iter(self._samples)

    def np_arrays(self) -> tuple[object, object]:
        """Return (positions, normals) as (N,3) numpy arrays. Cached."""
        if self._np_positions is not None:
            return self._np_positions, self._np_normals
        if not _HAS_NUMPY:
            raise RuntimeError("numpy not available")
        n = len(self._samples)
        positions = np.empty((n, 3), dtype=np.float64)
        normals = np.empty((n, 3), dtype=np.float64)
        for i, (pos, nrm) in enumerate(self._samples):
            positions[i, 0] = pos.x
            positions[i, 1] = pos.y
            positions[i, 2] = pos.z
            normals[i, 0] = nrm.x
            normals[i, 1] = nrm.y
            normals[i, 2] = nrm.z
        self._np_positions = positions
        self._np_normals = normals
        return positions, normals


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
        self._samples = self._build_samples()
        self._np_positions: object = None
        self._np_normals: object = None

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
        return iter(self._samples)

    def _build_samples(self) -> tuple[tuple[Vec3, Vec3], ...]:
        samples: list[tuple[Vec3, Vec3]] = []
        two_pi = 2.0 * math.pi
        eps = 0.01
        v_range = self.v_max - self.v_min
        scale = self._scale
        raw_point = self._raw_point

        u = 0.0
        while u < two_pi:
            for j in range(self.v_steps + 1):
                v = self.v_min + v_range * j / self.v_steps
                pt = raw_point(u, v) * scale

                # Numerical normal via cross product of partial derivatives
                du = raw_point(u + eps, v) - raw_point(u - eps, v)
                dv = raw_point(u, v + eps) - raw_point(u, v - eps)
                normal = du.cross(dv).normalized()

                samples.append((pt, normal))
            u += self.u_step
        return tuple(samples)

    def np_arrays(self) -> tuple[object, object]:
        """Return (positions, normals) as (N,3) numpy arrays. Cached."""
        if self._np_positions is not None:
            return self._np_positions, self._np_normals
        if not _HAS_NUMPY:
            raise RuntimeError("numpy not available")
        n = len(self._samples)
        positions = np.empty((n, 3), dtype=np.float64)
        normals = np.empty((n, 3), dtype=np.float64)
        for i, (pos, nrm) in enumerate(self._samples):
            positions[i, 0] = pos.x
            positions[i, 1] = pos.y
            positions[i, 2] = pos.z
            normals[i, 0] = nrm.x
            normals[i, 1] = nrm.y
            normals[i, 2] = nrm.z
        self._np_positions = positions
        self._np_normals = normals
        return positions, normals


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
    fx_scratch: list[Cell] = field(default_factory=list, repr=False)
    time: float = 0.0  # scene time, set before postfx for temporal effects

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
    objects look vivid rather than washed out. The thresholds are the
    raw-brightness equivalents of the renderer's gamma-lifted style
    bands, so callers can avoid a per-sample power operation.
    """
    if brightness > _STYLE_BRIGHT_CUTOFF:
        return bright
    elif brightness > _STYLE_PRIMARY_CUTOFF:
        return primary
    elif brightness > _STYLE_MID_CUTOFF:
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
        has_vnormals = bool(mesh.vertex_normals)
        if not mesh.normals and not has_vnormals:
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
                            if brightness > _STYLE_BRIGHT_CUTOFF:
                                style = bright_s
                            elif brightness > _STYLE_PRIMARY_CUTOFF:
                                style = primary_s
                            elif brightness > _STYLE_MID_CUTOFF:
                                style = mid_s
                            else:
                                style = dim_s
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

    def _draw_projected_line(
        self,
        p0: ScreenPoint,
        p1: ScreenPoint,
        char: str,
        styles: tuple[str, str, str, str],
    ) -> None:
        """Stream a depth-interpolated Bresenham line directly to the grid."""
        bright_s, primary_s, mid_s, dim_s = styles
        width = self.width
        height = self.height
        zbuf = self.grid.zbuf
        cells = self.grid.cells

        x0, y0, depth0 = p0.col, p0.row, p0.depth
        x1, y1, depth1 = p1.col, p1.row, p1.depth
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        steps = dx if dx > dy else dy
        if steps < 1:
            steps = 1
        depth_delta = depth1 - depth0
        step = 0
        endpoints_in_bounds = (
            0 <= x0 < width
            and 0 <= y0 < height
            and 0 <= x1 < width
            and 0 <= y1 < height
        )

        if endpoints_in_bounds:
            while True:
                depth = depth0 + depth_delta * (step / steps)
                idx = y0 * width + x0
                if depth < zbuf[idx]:
                    if depth < 0.35:
                        style = bright_s
                    elif depth < 0.55:
                        style = primary_s
                    elif depth < 0.75:
                        style = mid_s
                    else:
                        style = dim_s
                    zbuf[idx] = depth
                    cell = cells[idx]
                    cell.char = char
                    cell.style = style
                    cell.depth = depth

                if x0 == x1 and y0 == y1:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
                step += 1
            return

        while True:
            if 0 <= x0 < width and 0 <= y0 < height:
                depth = depth0 + depth_delta * (step / steps)
                idx = y0 * width + x0
                if depth < zbuf[idx]:
                    if depth < 0.35:
                        style = bright_s
                    elif depth < 0.55:
                        style = primary_s
                    elif depth < 0.75:
                        style = mid_s
                    else:
                        style = dim_s
                    zbuf[idx] = depth
                    cell = cells[idx]
                    cell.char = char
                    cell.style = style
                    cell.depth = depth

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            step += 1

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
            self._draw_projected_line(p0, p1, edge_char, styles)

        # Draw vertices
        width = self.width
        height = self.height
        zbuf = self.grid.zbuf
        cells = self.grid.cells
        if vertex_char:
            for sp in projected:
                if sp is not None:
                    col, row, depth = sp.col, sp.row, sp.depth
                    if 0 <= col < width and 0 <= row < height:
                        idx = row * width + col
                        if depth < zbuf[idx]:
                            style = depth_to_style(
                                depth, bright_s, primary_s, mid_s, dim_s
                            )
                            zbuf[idx] = depth
                            cell = cells[idx]
                            cell.char = vertex_char
                            cell.style = style
                            cell.depth = depth

    # ── point cloud rendering ─────────────────────────────────────────

    def draw_points(
        self,
        cloud: PointCloud,
        ctx: ProjectionContext,
        point_chars: str = "·∙•●",
        styles: tuple[str, str, str, str] = ("bright_white", "white", "grey70", "grey50"),
    ) -> None:
        """Render a point cloud with depth+brightness-based characters."""
        if not point_chars:
            return

        bright_s, primary_s, mid_s, dim_s = styles
        last_char_index = len(point_chars) - 1
        points = cloud.points
        brightness = cloud.brightness
        brightness_count = len(brightness)
        width = self.width
        height = self.height
        max_col = ctx.max_col
        max_row = ctx.max_row
        clip_margin = ctx.clip_margin
        depth_margin = ctx.depth_margin

        m0, m1, m2, m3 = ctx.m0, ctx.m1, ctx.m2, ctx.m3
        m4, m5, m6, m7 = ctx.m4, ctx.m5, ctx.m6, ctx.m7
        m8, m9, m10, m11 = ctx.m8, ctx.m9, ctx.m10, ctx.m11
        m12, m13, m14, m15 = ctx.m12, ctx.m13, ctx.m14, ctx.m15

        zbuf = self.grid.zbuf
        cells = self.grid.cells

        for i, pt in enumerate(points):
            px, py, pz = pt.x, pt.y, pt.z
            clip_x = m0 * px + m1 * py + m2 * pz + m3
            clip_y = m4 * px + m5 * py + m6 * pz + m7
            clip_z = m8 * px + m9 * py + m10 * pz + m11
            clip_w = m12 * px + m13 * py + m14 * pz + m15
            if clip_w <= 0.0:
                continue

            inv_w = 1.0 / clip_w
            ndc_x = clip_x * inv_w
            ndc_y = clip_y * inv_w
            ndc_z = clip_z * inv_w
            if (
                abs(ndc_x) > clip_margin
                or abs(ndc_y) > clip_margin
                or abs(ndc_z) > depth_margin
            ):
                continue

            col = int(((ndc_x * 0.5) + 0.5) * max_col + 0.5)
            row = int(((1.0 - ndc_y) * 0.5) * max_row + 0.5)
            if col < 0 or col >= width or row < 0 or row >= height:
                continue

            depth = clamp((ndc_z + 1.0) * 0.5, 0.0, 1.0)
            idx = row * width + col
            if depth >= zbuf[idx]:
                continue

            point_brightness = brightness[i] if i < brightness_count else 0.5
            combined = point_brightness * (1.0 - depth * 0.5)
            char_idx = int(combined * last_char_index)
            if char_idx < 0:
                char_idx = 0
            elif char_idx > last_char_index:
                char_idx = last_char_index
            char = point_chars[char_idx]

            if depth < 0.35:
                style = bright_s
            elif depth < 0.55:
                style = primary_s
            elif depth < 0.75:
                style = mid_s
            else:
                style = dim_s

            zbuf[idx] = depth
            cell = cells[idx]
            cell.char = char
            cell.style = style
            cell.depth = depth

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
            self._draw_projected_line(p0, p1, edge_char, styles)

        # Vertices (bright dots)
        width = self.width
        height = self.height
        zbuf = self.grid.zbuf
        cells = self.grid.cells
        for sp in projected:
            if sp is not None:
                col, row, depth = sp.col, sp.row, sp.depth * 0.9
                if 0 <= col < width and 0 <= row < height:
                    idx = row * width + col
                    if depth < zbuf[idx]:
                        zbuf[idx] = depth
                        cell = cells[idx]
                        cell.char = vertex_char
                        cell.style = bright_s
                        cell.depth = depth

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

        Uses numpy for batch transform/shade when available, falling
        back to a pure-Python inner loop otherwise.
        """
        if _HAS_NUMPY and hasattr(surface, 'np_arrays'):
            try:
                self._draw_surface_direct_numpy(
                    surface, ctx, light, styles, luminance_ramp,
                )
                return
            except Exception:
                logger.debug(
                    "numpy surface render failed, falling back to scalar",
                    exc_info=True,
                )

        self._draw_surface_direct_scalar(
            surface, ctx, light, styles, luminance_ramp,
        )

    def _draw_surface_direct_numpy(
        self,
        surface: SurfaceSampler,
        ctx: ProjectionContext,
        light: Light,
        styles: tuple[str, str, str, str],
        luminance_ramp: str,
    ) -> None:
        """Vectorized surface render using numpy for batch math.

        All transform, culling, normal transform, and shading is done in
        batch numpy operations. Only the z-buffer write loop remains in
        Python (sequential due to depth ordering).
        """
        positions, normals = surface.np_arrays()  # (N,3) each, cached

        bright_s, primary_s, mid_s, dim_s = styles
        n_chars = len(luminance_ramp)
        w, h = self.width, self.height
        max_col = float(max(w - 1, 0))
        max_row = float(max(h - 1, 0))

        # ── batch MVP transform ───────────────────────────────────────
        # MVP as 3x4 slice: clip = positions @ mvp34.T + translation row
        # Equivalent to homogeneous multiply but avoids (N,4) allocation.
        mvp34 = np.array([
            [ctx.m0,  ctx.m1,  ctx.m2],
            [ctx.m4,  ctx.m5,  ctx.m6],
            [ctx.m8,  ctx.m9,  ctx.m10],
            [ctx.m12, ctx.m13, ctx.m14],
        ], dtype=np.float64)  # (4, 3)
        mvp_t = np.array([ctx.m3, ctx.m7, ctx.m11, ctx.m15], dtype=np.float64)

        # clip_xyzw = positions @ mvp34.T + mvp_t  → (N, 4)
        clip = positions @ mvp34.T + mvp_t  # broadcast (N,4)

        clip_w = clip[:, 3]

        # ── cull behind camera ────────────────────────────────────────
        visible = clip_w > 0.0

        # Safe reciprocal (only where visible)
        safe_w = np.where(visible, clip_w, 1.0)
        inv_w = 1.0 / safe_w
        inv_w[~visible] = 0.0

        ndc_x = clip[:, 0] * inv_w
        ndc_y = clip[:, 1] * inv_w
        ndc_z = clip[:, 2] * inv_w

        # ── frustum cull ──────────────────────────────────────────────
        visible &= (ndc_x >= -1.2) & (ndc_x <= 1.2)
        visible &= (ndc_y >= -1.2) & (ndc_y <= 1.2)

        # ── NDC → screen coords ──────────────────────────────────────
        cols = ((ndc_x * 0.5 + 0.5) * max_col + 0.5).astype(np.int32)
        rows = (((1.0 - ndc_y) * 0.5) * max_row + 0.5).astype(np.int32)

        visible &= (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)

        # ── depth ─────────────────────────────────────────────────────
        depth = np.clip((ndc_z + 1.0) * 0.5, 0.0, 1.0)

        # ── normal transform (batch) ─────────────────────────────────
        nm = np.array([
            [ctx.nm0, ctx.nm1, ctx.nm2],
            [ctx.nm4, ctx.nm5, ctx.nm6],
            [ctx.nm8, ctx.nm9, ctx.nm10],
        ], dtype=np.float64)

        world_normals = normals @ nm.T  # (N, 3)
        norms_sq = np.einsum('ij,ij->i', world_normals, world_normals)
        inv_norms = np.where(norms_sq > 1e-20, 1.0 / np.sqrt(norms_sq), 0.0)
        world_normals = world_normals * inv_norms[:, np.newaxis]

        # ── Lambert shading (batch) ──────────────────────────────────
        light_dir = np.array(
            [-light.direction.x, -light.direction.y, -light.direction.z],
            dtype=np.float64,
        )
        ndotl = world_normals @ light_dir  # (N,)

        if light.wrap > 0.0:
            brightness = (ndotl + light.wrap) / (1.0 + light.wrap)
        else:
            np.maximum(ndotl, 0.0, out=ndotl)
            brightness = ndotl

        brightness = light.ambient + (1.0 - light.ambient) * brightness * light.intensity
        np.clip(brightness, 0.0, 1.0, out=brightness)

        # ── char index + style selection (batch) ─────────────────────
        char_indices = np.clip(
            (brightness * (n_chars - 1)).astype(np.int32), 0, n_chars - 1,
        )

        # Filter out space characters (index 0 for standard ramp)
        # Build a lookup: which char indices map to space?
        space_mask = np.array(
            [c == " " for c in luminance_ramp], dtype=bool,
        )  # tiny (13 elements)
        visible &= ~space_mask[char_indices]

        # ── compact to visible-only and convert to Python lists ─────
        vis_indices = np.nonzero(visible)[0]
        if len(vis_indices) == 0:
            return

        # Convert to flat Python lists (.tolist()) — avoids expensive
        # per-element numpy scalar → Python conversions in the loop.
        flat_idx = (rows[vis_indices] * w + cols[vis_indices]).tolist()
        flat_depth = depth[vis_indices].tolist()
        flat_char_idx = char_indices[vis_indices].tolist()
        flat_bright = brightness[vis_indices].tolist()

        # Pre-build char list from ramp
        ramp_arr = list(luminance_ramp)

        # ── z-buffer write loop (Python — sequential) ────────────────
        zbuf = self.grid.zbuf
        cells = self.grid.cells
        _bc = _STYLE_BRIGHT_CUTOFF
        _pc = _STYLE_PRIMARY_CUTOFF
        _mc = _STYLE_MID_CUTOFF

        for k in range(len(flat_idx)):
            d = flat_depth[k]
            idx = flat_idx[k]
            if d < zbuf[idx]:
                zbuf[idx] = d
                cell = cells[idx]
                cell.char = ramp_arr[flat_char_idx[k]]
                b = flat_bright[k]
                if b > _bc:
                    cell.style = bright_s
                elif b > _pc:
                    cell.style = primary_s
                elif b > _mc:
                    cell.style = mid_s
                else:
                    cell.style = dim_s
                cell.depth = d

    def _draw_surface_direct_scalar(
        self,
        surface: SurfaceSampler,
        ctx: ProjectionContext,
        light: Light,
        styles: tuple[str, str, str, str],
        luminance_ramp: str,
    ) -> None:
        """Pure-Python fallback for surface rendering.

        Manually inlined inner loop — used when numpy is unavailable.
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
                inv_len = 1.0 / math.sqrt(inv_len)
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

            if brightness > _STYLE_BRIGHT_CUTOFF:
                style = bright_s
            elif brightness > _STYLE_PRIMARY_CUTOFF:
                style = primary_s
            elif brightness > _STYLE_MID_CUTOFF:
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
