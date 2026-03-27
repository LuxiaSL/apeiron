"""Z-buffered ASCII rasterizer.

Converts 3D geometry into a character grid using depth-tested rendering.
Supports three modes: filled faces, wireframe edges, and point clouds.
Produces a CharGrid that can be post-processed and converted to Rich Text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.text import Text

from .lut import Vec3, clamp, lerp_f
from .geometry import Mesh, PointCloud, VoxelGrid
from .transform import (
    Camera,
    ProjectionContext,
    ScreenPoint,
    bresenham,
    point_in_triangle,
    barycentric,
)


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
            self.cells[idx] = Cell(char=char, style=style, depth=depth)
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
        for row in range(self.height):
            for col in range(self.width):
                cell = self.cells[row * self.width + col]
                if cell.style:
                    text.append(cell.char, style=cell.style)
                else:
                    text.append(cell.char)
            if row < self.height - 1:
                text.append("\n")
        return text


# ── depth → style mapping ──────────────────────────────────────────────


def depth_to_style(
    depth: float,
    bright: str,
    primary: str,
    mid: str,
    dim: str,
) -> str:
    """Map normalized depth [0, 1] to one of four palette styles."""
    if depth < 0.3:
        return bright
    elif depth < 0.5:
        return primary
    elif depth < 0.7:
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

    def shade(self, normal: Vec3) -> float:
        """Compute brightness for a face with the given normal."""
        ndotl = normal.dot(self.direction)
        if self.wrap > 0.0:
            brightness = (ndotl + self.wrap) / (1.0 + self.wrap)
        else:
            brightness = max(0.0, ndotl)
        return clamp(brightness * self.intensity, 0.0, 1.0)


DEFAULT_LIGHT = Light(
    direction=Vec3(0.3, -0.8, 0.5).normalized(),
    intensity=1.0,
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
        """Render filled faces with face-normal shading.

        shader_chars: brightness ramp (dark→light), e.g. " .:-=+*#%@"
        styles: (bright, primary, mid, dim) palette style strings
        """
        if not mesh.normals:
            mesh.compute_normals()

        n_chars = len(shader_chars)
        bright_s, primary_s, mid_s, dim_s = styles

        # Project all vertices once
        projected: list[tuple[float, float, float] | None] = [
            ctx.project_vertex_unclamped(v) for v in mesh.vertices
        ]

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

            # Compute face brightness from normal
            normal = mesh.normals[fi] if fi < len(mesh.normals) else Vec3(0, 1, 0)
            view_normal = ctx.transform_normal(normal)
            brightness = light.shade(view_normal)
            char_idx = int(brightness * (n_chars - 1))
            char_idx = max(0, min(char_idx, n_chars - 1))
            char = shader_chars[char_idx]

            if char == " ":
                continue  # fully dark, skip

            # Average depth for style selection
            avg_depth = sum(p[2] for p in pts) / len(pts)
            style = depth_to_style(avg_depth, bright_s, primary_s, mid_s, dim_s)

            # Rasterize triangulated face
            # For quads+, fan-triangulate from vertex 0
            for ti in range(1, len(pts) - 1):
                self._fill_triangle(
                    pts[0], pts[ti], pts[ti + 1],
                    char, style,
                )

    def _fill_triangle(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        char: str,
        style: str,
    ) -> None:
        """Scan-convert a screen-space triangle with depth interpolation."""
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c

        # Bounding box (clamped to grid)
        min_col = max(0, int(min(ax, bx, cx)))
        max_col = min(self.width - 1, int(max(ax, bx, cx)) + 1)
        min_row = max(0, int(min(ay, by, cy)))
        max_row = min(self.height - 1, int(max(ay, by, cy)) + 1)

        # Skip tiny or off-screen triangles
        if min_col > max_col or min_row > max_row:
            return

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                px, py = col + 0.5, row + 0.5
                bary = barycentric(px, py, ax, ay, bx, by, cx, cy)
                if bary is None:
                    continue
                u, v, w = bary
                if u < -0.001 or v < -0.001 or w < -0.001:
                    continue

                # Interpolate depth
                depth = u * az + v * bz + w * cz
                self.grid.write(col, row, char, style, depth)

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

    # ── composite helper ──────────────────────────────────────────────

    def overlay(self, other: CharGrid) -> None:
        """Composite another grid on top (non-space characters only)."""
        for i, cell in enumerate(other.cells):
            if cell.char != " " and cell.depth < self.grid.zbuf[i]:
                self.grid.cells[i] = cell
                self.grid.zbuf[i] = cell.depth
