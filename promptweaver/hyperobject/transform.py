"""Transform pipeline: model -> world -> view -> clip -> screen.

Handles the full vertex transformation chain including aspect-ratio
correction for terminal characters (~2:1 height:width).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .lut import Mat4, Vec3, Vec4, clamp


# Terminal characters are approximately 2x taller than wide.
# We compress Y by this factor during projection so circles look circular.
CHAR_ASPECT: float = 0.5


@dataclass(slots=True)
class ScreenPoint:
    """A projected point in screen (character cell) coordinates."""

    col: int   # x: column (0 = left)
    row: int   # y: row (0 = top)
    depth: float  # normalized depth (0 = near, 1 = far) for z-buffer


@dataclass(slots=True)
class Camera:
    """Simple orbit/look-at camera."""

    position: Vec3
    target: Vec3
    fov: float = 1.2  # ~69 degrees, radians
    near: float = 0.1
    far: float = 50.0

    def view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, Vec3(0.0, 1.0, 0.0))

    def projection_matrix(self, width: int, height: int) -> Mat4:
        """Perspective projection with terminal aspect correction."""
        # Effective aspect ratio accounts for character cell shape
        char_aspect = (width / max(height, 1)) * CHAR_ASPECT
        return Mat4.perspective(self.fov, char_aspect, self.near, self.far)


@dataclass
class ProjectionContext:
    """Pre-computed matrices and viewport dimensions for one frame.

    Created once per frame, passed to the rasterizer to avoid
    recomputing matrices per-vertex.
    """

    model: Mat4
    view: Mat4
    projection: Mat4
    mvp: Mat4
    width: int
    height: int
    half_w: float = 0.0
    half_h: float = 0.0
    max_col: float = 0.0
    max_row: float = 0.0
    clip_margin: float = 1.25
    unclamped_margin: float = 2.0
    depth_margin: float = 1.05
    m0: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    m5: float = 0.0
    m6: float = 0.0
    m7: float = 0.0
    m8: float = 0.0
    m9: float = 0.0
    m10: float = 0.0
    m11: float = 0.0
    m12: float = 0.0
    m13: float = 0.0
    m14: float = 0.0
    m15: float = 0.0
    nm0: float = 0.0
    nm1: float = 0.0
    nm2: float = 0.0
    nm4: float = 0.0
    nm5: float = 0.0
    nm6: float = 0.0
    nm8: float = 0.0
    nm9: float = 0.0
    nm10: float = 0.0

    def __post_init__(self) -> None:
        self.half_w = self.width / 2.0
        self.half_h = self.height / 2.0
        self.max_col = float(max(self.width - 1, 0))
        self.max_row = float(max(self.height - 1, 0))
        mvp = self.mvp.m
        self.m0, self.m1, self.m2, self.m3 = mvp[0], mvp[1], mvp[2], mvp[3]
        self.m4, self.m5, self.m6, self.m7 = mvp[4], mvp[5], mvp[6], mvp[7]
        self.m8, self.m9, self.m10, self.m11 = mvp[8], mvp[9], mvp[10], mvp[11]
        self.m12, self.m13, self.m14, self.m15 = mvp[12], mvp[13], mvp[14], mvp[15]
        model = self.model.m
        self.nm0, self.nm1, self.nm2 = model[0], model[1], model[2]
        self.nm4, self.nm5, self.nm6 = model[4], model[5], model[6]
        self.nm8, self.nm9, self.nm10 = model[8], model[9], model[10]

    @staticmethod
    def build(
        model: Mat4,
        camera: Camera,
        width: int,
        height: int,
    ) -> ProjectionContext:
        view = camera.view_matrix()
        proj = camera.projection_matrix(width, height)
        mvp = proj @ view @ model
        return ProjectionContext(
            model=model,
            view=view,
            projection=proj,
            mvp=mvp,
            width=width,
            height=height,
        )

    def project_vertex(self, v: Vec3) -> ScreenPoint | None:
        """Transform a world-space vertex to screen coordinates.

        Returns None if the vertex is behind the camera or outside
        the view frustum.
        """
        vx, vy, vz = v.x, v.y, v.z
        clip_x = self.m0 * vx + self.m1 * vy + self.m2 * vz + self.m3
        clip_y = self.m4 * vx + self.m5 * vy + self.m6 * vz + self.m7
        clip_z = self.m8 * vx + self.m9 * vy + self.m10 * vz + self.m11
        clip_w = self.m12 * vx + self.m13 * vy + self.m14 * vz + self.m15

        # Behind camera
        if clip_w <= 0.0:
            return None

        inv_w = 1.0 / clip_w
        ndc_x = clip_x * inv_w
        ndc_y = clip_y * inv_w
        ndc_z = clip_z * inv_w

        # Frustum cull (with margin for edge drawing)
        margin = self.clip_margin
        if abs(ndc_x) > margin or abs(ndc_y) > margin or abs(ndc_z) > self.depth_margin:
            return None

        # NDC → screen
        col = int(((ndc_x * 0.5) + 0.5) * self.max_col + 0.5)
        row = int(((1.0 - ndc_y) * 0.5) * self.max_row + 0.5)

        # Depth: map from [-1, 1] to [0, 1]
        depth = clamp((ndc_z + 1.0) * 0.5, 0.0, 1.0)

        return ScreenPoint(col=col, row=row, depth=depth)

    def project_vertex_unclamped(self, v: Vec3) -> tuple[float, float, float] | None:
        """Like project_vertex but returns float coords for line interpolation.

        Returns (col, row, depth) as floats, or None if behind camera.
        """
        vx, vy, vz = v.x, v.y, v.z
        clip_x = self.m0 * vx + self.m1 * vy + self.m2 * vz + self.m3
        clip_y = self.m4 * vx + self.m5 * vy + self.m6 * vz + self.m7
        clip_z = self.m8 * vx + self.m9 * vy + self.m10 * vz + self.m11
        clip_w = self.m12 * vx + self.m13 * vy + self.m14 * vz + self.m15
        if clip_w <= 0.0:
            return None

        inv_w = 1.0 / clip_w
        ndc_x = clip_x * inv_w
        ndc_y = clip_y * inv_w
        ndc_z = clip_z * inv_w

        if abs(ndc_x) > self.unclamped_margin or abs(ndc_y) > self.unclamped_margin or abs(ndc_z) > self.depth_margin:
            return None

        col = ((ndc_x * 0.5) + 0.5) * self.max_col
        row = ((1.0 - ndc_y) * 0.5) * self.max_row
        depth = clamp((ndc_z + 1.0) * 0.5, 0.0, 1.0)

        return (col, row, depth)

    def transform_normal(self, n: Vec3) -> Vec3:
        """Transform a face normal to view space for lighting."""
        # For uniform scale, model matrix suffices.
        # For non-uniform scale, we'd need the inverse transpose.
        nx, ny, nz = n.x, n.y, n.z
        tx = self.nm0 * nx + self.nm1 * ny + self.nm2 * nz
        ty = self.nm4 * nx + self.nm5 * ny + self.nm6 * nz
        tz = self.nm8 * nx + self.nm9 * ny + self.nm10 * nz
        length_sq = tx * tx + ty * ty + tz * tz
        if length_sq < 1e-20:
            return Vec3(0.0, 0.0, 0.0)
        inv_len = 1.0 / math.sqrt(length_sq)
        return Vec3(tx * inv_len, ty * inv_len, tz * inv_len)


# ── screen-space helpers ────────────────────────────────────────────────


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm. Returns list of (col, row) points."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def edge_function(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """2D cross product for triangle rasterization (signed area).

    Positive if (cx, cy) is on the left side of edge A→B.
    """
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def point_in_triangle(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> bool:
    """Test if point P is inside triangle ABC using edge functions."""
    e0 = edge_function(ax, ay, bx, by, px, py)
    e1 = edge_function(bx, by, cx, cy, px, py)
    e2 = edge_function(cx, cy, ax, ay, px, py)
    # All same sign (or zero) means inside
    return (e0 >= 0 and e1 >= 0 and e2 >= 0) or (e0 <= 0 and e1 <= 0 and e2 <= 0)


def barycentric(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> tuple[float, float, float] | None:
    """Compute barycentric coordinates of P w.r.t. triangle ABC.

    Returns (u, v, w) where P = u*A + v*B + w*C, or None if degenerate.
    """
    area = edge_function(ax, ay, bx, by, cx, cy)
    if abs(area) < 1e-10:
        return None
    inv_area = 1.0 / area
    u = edge_function(bx, by, cx, cy, px, py) * inv_area
    v = edge_function(cx, cy, ax, ay, px, py) * inv_area
    w = 1.0 - u - v
    return (u, v, w)
