"""Mesh containers and geometry operations for the hyperobject renderer.

A Mesh is a simple indexed polygon soup: vertices + edges + faces.
All geometry factories (primitives.py) produce Mesh instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .lut import Vec3


@dataclass
class Mesh:
    """Indexed triangle/polygon mesh.

    vertices: model-space positions
    edges:    index pairs (for wireframe rendering)
    faces:    index tuples (triangles or quads — renderer handles both)
    normals:  per-face normals (auto-computed if empty)
    """

    vertices: list[Vec3] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)

    def compute_normals(self) -> None:
        """Compute per-face normals from vertex positions.

        For triangles: cross product of two edges.
        For quads+: uses first three vertices as the triangle.
        """
        self.normals = []
        for face in self.faces:
            if len(face) < 3:
                self.normals.append(Vec3(0.0, 1.0, 0.0))
                continue
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = e1.cross(e2).normalized()
            self.normals.append(normal)

    def compute_edges_from_faces(self) -> None:
        """Derive unique edge list from face connectivity."""
        edge_set: set[tuple[int, int]] = set()
        for face in self.faces:
            n = len(face)
            for i in range(n):
                a, b = face[i], face[(i + 1) % n]
                edge = (min(a, b), max(a, b))
                edge_set.add(edge)
        self.edges = sorted(edge_set)

    def translate(self, offset: Vec3) -> Mesh:
        """Return a translated copy."""
        return Mesh(
            vertices=[v + offset for v in self.vertices],
            edges=list(self.edges),
            faces=list(self.faces),
            normals=list(self.normals),
        )

    def scale_uniform(self, factor: float) -> Mesh:
        """Return a uniformly scaled copy (about the origin)."""
        return Mesh(
            vertices=[v * factor for v in self.vertices],
            edges=list(self.edges),
            faces=list(self.faces),
            normals=list(self.normals),
        )

    def centroid(self) -> Vec3:
        """Average position of all vertices."""
        if not self.vertices:
            return Vec3(0.0, 0.0, 0.0)
        n = len(self.vertices)
        sx = sum(v.x for v in self.vertices)
        sy = sum(v.y for v in self.vertices)
        sz = sum(v.z for v in self.vertices)
        return Vec3(sx / n, sy / n, sz / n)

    def bounding_radius(self) -> float:
        """Max distance from origin to any vertex."""
        if not self.vertices:
            return 0.0
        return max(v.length() for v in self.vertices)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


# ── point cloud (for attractors, particles, nebulae) ────────────────────


@dataclass
class PointCloud:
    """Unstructured set of 3D points with per-point brightness.

    Used for strange attractors, particle nebulae, and similar
    non-mesh geometries. Each point is rendered as a single character.
    """

    points: list[Vec3] = field(default_factory=list)
    brightness: list[float] = field(default_factory=list)  # 0.0–1.0 per point

    def add(self, point: Vec3, bright: float = 1.0) -> None:
        self.points.append(point)
        self.brightness.append(bright)

    def trim(self, max_points: int) -> None:
        """Remove oldest points if over budget."""
        if len(self.points) > max_points:
            excess = len(self.points) - max_points
            self.points = self.points[excess:]
            self.brightness = self.brightness[excess:]

    @property
    def count(self) -> int:
        return len(self.points)


# ── voxel grid (for site_decay) ─────────────────────────────────────────


@dataclass
class VoxelGrid:
    """Boolean occupancy grid with per-cell state.

    Used by the site_decay geometry (eroding architectural skeleton).
    Each voxel is rendered as a filled block if occupied.
    """

    size_x: int
    size_y: int
    size_z: int
    spacing: float = 0.3
    cells: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = [True] * (self.size_x * self.size_y * self.size_z)

    def _idx(self, x: int, y: int, z: int) -> int:
        return x + y * self.size_x + z * self.size_x * self.size_y

    def get(self, x: int, y: int, z: int) -> bool:
        if 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z:
            return self.cells[self._idx(x, y, z)]
        return False

    def set(self, x: int, y: int, z: int, alive: bool) -> None:
        if 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z:
            self.cells[self._idx(x, y, z)] = alive

    def alive_count(self) -> int:
        return sum(self.cells)

    def total_count(self) -> int:
        return self.size_x * self.size_y * self.size_z

    def center_offset(self) -> Vec3:
        """Offset to center the grid at the origin."""
        return Vec3(
            -self.size_x * self.spacing / 2.0,
            -self.size_y * self.spacing / 2.0,
            -self.size_z * self.spacing / 2.0,
        )

    def cell_center(self, x: int, y: int, z: int) -> Vec3:
        """World-space center of a cell (with centering offset)."""
        off = self.center_offset()
        return Vec3(
            (x + 0.5) * self.spacing + off.x,
            (y + 0.5) * self.spacing + off.y,
            (z + 0.5) * self.spacing + off.z,
        )

    def fill_ratio(self) -> float:
        total = self.total_count()
        if total == 0:
            return 0.0
        return self.alive_count() / total


# ── heightmap surface (for textural_macro / environmental) ──────────────


@dataclass
class HeightMap:
    """2D grid of height values, rendered as a surface mesh.

    Used by textural_macro (undulating surface) and environmental
    (terrain flyover). Heights drive Y displacement of a flat grid.
    """

    width: int
    depth: int
    spacing: float = 0.2
    heights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.heights:
            self.heights = [0.0] * (self.width * self.depth)

    def get(self, x: int, z: int) -> float:
        if 0 <= x < self.width and 0 <= z < self.depth:
            return self.heights[x + z * self.width]
        return 0.0

    def set(self, x: int, z: int, h: float) -> None:
        if 0 <= x < self.width and 0 <= z < self.depth:
            self.heights[x + z * self.width] = h

    def to_mesh(self) -> Mesh:
        """Convert heightmap to triangle mesh for rendering."""
        verts: list[Vec3] = []
        cx = (self.width - 1) * self.spacing / 2.0
        cz = (self.depth - 1) * self.spacing / 2.0

        for z in range(self.depth):
            for x in range(self.width):
                px = x * self.spacing - cx
                py = self.get(x, z)
                pz = z * self.spacing - cz
                verts.append(Vec3(px, py, pz))

        faces: list[tuple[int, ...]] = []
        for z in range(self.depth - 1):
            for x in range(self.width - 1):
                i00 = x + z * self.width
                i10 = (x + 1) + z * self.width
                i01 = x + (z + 1) * self.width
                i11 = (x + 1) + (z + 1) * self.width
                faces.append((i00, i10, i11))
                faces.append((i00, i11, i01))

        mesh = Mesh(vertices=verts, faces=faces)
        mesh.compute_normals()
        mesh.compute_edges_from_faces()
        return mesh


# ── simple noise (avoids external deps) ─────────────────────────────────

# Permutation table for gradient noise
_PERM = list(range(256))
# Deterministic shuffle using a simple LCG
_seed = 42
for _i in range(255, 0, -1):
    _seed = (_seed * 1103515245 + 12345) & 0x7FFFFFFF
    _j = _seed % (_i + 1)
    _PERM[_i], _PERM[_j] = _PERM[_j], _PERM[_i]
_PERM = _PERM + _PERM  # Double for wrapping

_GRAD3 = [
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
]


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grad3(h: int, x: float, y: float, z: float) -> float:
    g = _GRAD3[h % 12]
    return g[0] * x + g[1] * y + g[2] * z


def noise3(x: float, y: float, z: float) -> float:
    """3D Perlin noise, returns value in approximately [-1, 1]."""
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    zi = int(math.floor(z)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    zf = z - math.floor(z)

    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    p = _PERM
    aaa = p[p[p[xi] + yi] + zi]
    aba = p[p[p[xi] + yi + 1] + zi]
    aab = p[p[p[xi] + yi] + zi + 1]
    abb = p[p[p[xi] + yi + 1] + zi + 1]
    baa = p[p[p[xi + 1] + yi] + zi]
    bba = p[p[p[xi + 1] + yi + 1] + zi]
    bab = p[p[p[xi + 1] + yi] + zi + 1]
    bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]

    x1 = _grad3(aaa, xf, yf, zf) + (_grad3(baa, xf - 1, yf, zf) - _grad3(aaa, xf, yf, zf)) * u
    x2 = _grad3(aba, xf, yf - 1, zf) + (_grad3(bba, xf - 1, yf - 1, zf) - _grad3(aba, xf, yf - 1, zf)) * u
    y1 = x1 + (x2 - x1) * v

    x1 = _grad3(aab, xf, yf, zf - 1) + (_grad3(bab, xf - 1, yf, zf - 1) - _grad3(aab, xf, yf, zf - 1)) * u
    x2 = _grad3(abb, xf, yf - 1, zf - 1) + (_grad3(bbb, xf - 1, yf - 1, zf - 1) - _grad3(abb, xf, yf - 1, zf - 1)) * u
    y2 = x1 + (x2 - x1) * v

    return y1 + (y2 - y1) * w


def fbm(x: float, y: float, z: float, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
    """Fractal Brownian Motion — layered noise for natural terrain."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * noise3(x * frequency, y * frequency, z * frequency)
        amplitude *= gain
        frequency *= lacunarity
    return value
