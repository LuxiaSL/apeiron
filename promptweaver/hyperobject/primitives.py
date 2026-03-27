"""Geometry factories for Hyperobject Mode.

Each factory returns a Mesh, PointCloud, VoxelGrid, HeightMap, or raw
vertex/edge data (tesseract).  All geometry is centered at the origin
with a bounding radius of roughly 1.0-1.5; the camera/viewport handles
final scaling.

15 geometries total: 14 template-mapped forms + the tesseract.
"""

from __future__ import annotations

import math
import random
from itertools import product
from typing import Sequence

from .lut import Vec3, Vec4
from .geometry import HeightMap, Mesh, PointCloud, VoxelGrid, fbm, noise3


# =====================================================================
# 1. Tesseract (4-D hypercube)
# =====================================================================

def make_tesseract() -> tuple[list[Vec4], list[tuple[int, int]]]:
    """Build a 4-D hypercube: 16 vertices, 32 edges.

    Returns
    -------
    vertices : list[Vec4]
        All 16 vertices at (+/-1, +/-1, +/-1, +/-1).
    edges : list[tuple[int, int]]
        32 edges connecting vertex pairs that differ in exactly one
        coordinate.
    """
    signs: list[float] = [-1.0, 1.0]
    vertices: list[Vec4] = [
        Vec4(x, y, z, w)
        for x, y, z, w in product(signs, repeat=4)
    ]

    edges: list[tuple[int, int]] = []
    n = len(vertices)
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            # Two vertices are connected if they differ in exactly one
            # coordinate (Manhattan distance == 2 since values are +/-1).
            diff = (
                abs(vi.x - vj.x)
                + abs(vi.y - vj.y)
                + abs(vi.z - vj.z)
                + abs(vi.w - vj.w)
            )
            if abs(diff - 2.0) < 1e-9:
                edges.append((i, j))

    return vertices, edges


# =====================================================================
# 2. Icosahedron  (material_study)
# =====================================================================

def make_icosahedron(subdivisions: int = 0) -> Mesh:
    """Golden-ratio icosahedron with optional Loop-style subdivision.

    Parameters
    ----------
    subdivisions : int
        Number of subdivision passes.  Each pass splits every triangle
        into 4, projecting new vertices onto the unit sphere.

    Returns a Mesh with computed normals and edges.
    """
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio

    # 12 vertices of a regular icosahedron (on the unit sphere after
    # normalization).
    raw: list[Vec3] = [
        Vec3(-1.0,  phi, 0.0),
        Vec3( 1.0,  phi, 0.0),
        Vec3(-1.0, -phi, 0.0),
        Vec3( 1.0, -phi, 0.0),
        Vec3(0.0, -1.0,  phi),
        Vec3(0.0,  1.0,  phi),
        Vec3(0.0, -1.0, -phi),
        Vec3(0.0,  1.0, -phi),
        Vec3( phi, 0.0, -1.0),
        Vec3( phi, 0.0,  1.0),
        Vec3(-phi, 0.0, -1.0),
        Vec3(-phi, 0.0,  1.0),
    ]
    vertices: list[Vec3] = [v.normalized() for v in raw]

    faces: list[tuple[int, ...]] = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    # Subdivision loop
    for _ in range(subdivisions):
        vertices, faces = _subdivide_sphere(vertices, faces)

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


def _subdivide_sphere(
    vertices: list[Vec3],
    faces: list[tuple[int, ...]],
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """One pass of icosphere subdivision.

    Each triangle (a, b, c) is split into 4 by inserting midpoints on
    each edge and projecting them onto the unit sphere.
    """
    midpoint_cache: dict[tuple[int, int], int] = {}

    def _midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = ((vertices[i] + vertices[j]) * 0.5).normalized()
        idx = len(vertices)
        vertices.append(mid)
        midpoint_cache[key] = idx
        return idx

    new_faces: list[tuple[int, ...]] = []
    for a, b, c in faces:
        ab = _midpoint(a, b)
        bc = _midpoint(b, c)
        ca = _midpoint(c, a)
        new_faces.extend([
            (a, ab, ca),
            (b, bc, ab),
            (c, ca, bc),
            (ab, bc, ca),
        ])
    return vertices, new_faces


# =====================================================================
# 3. Noise Surface  (textural_macro)
# =====================================================================

def make_noise_surface(
    width: int = 24,
    depth: int = 24,
    freq: float = 0.3,
    amplitude: float = 0.4,
) -> HeightMap:
    """Flat grid displaced by fractal Perlin noise.

    Returns a HeightMap centered at the origin.
    """
    hmap = HeightMap(width=width, depth=depth)
    for z in range(depth):
        for x in range(width):
            h = fbm(x * freq, 0.0, z * freq, octaves=4) * amplitude
            hmap.set(x, z, h)
    return hmap


# =====================================================================
# 4. Terrain  (environmental)
# =====================================================================

def make_terrain(
    width: int = 32,
    depth: int = 32,
    freq: float = 0.15,
    amplitude: float = 0.8,
) -> HeightMap:
    """Larger heightmap with multi-octave noise for landscape feel.

    Higher octave count and lacunarity produce more dramatic ridges.
    """
    hmap = HeightMap(width=width, depth=depth)
    for z in range(depth):
        for x in range(width):
            h = fbm(
                x * freq, 0.0, z * freq,
                octaves=6, lacunarity=2.2, gain=0.45,
            ) * amplitude
            hmap.set(x, z, h)
    return hmap


# =====================================================================
# 5. Particle Nebula  (atmospheric_depth)
# =====================================================================

def make_particle_nebula(
    count: int = 300,
    spread: float = 1.2,
) -> PointCloud:
    """3-D Gaussian particle cloud.

    Brightness falls off with distance from the center so the nebula
    has a bright core and dim halo.
    """
    rng = random.Random(42)
    cloud = PointCloud()
    for _ in range(count):
        # Box-Muller for 3-D Gaussian
        x = rng.gauss(0.0, spread * 0.4)
        y = rng.gauss(0.0, spread * 0.4)
        z = rng.gauss(0.0, spread * 0.4)
        pt = Vec3(x, y, z)
        dist = pt.length()
        bright = max(0.05, 1.0 - dist / (spread * 1.5))
        cloud.add(pt, bright)
    return cloud


# =====================================================================
# 6. Metaballs  (process_state)
# =====================================================================

def make_metaballs(
    n_blobs: int = 3,
    resolution: int = 30,
) -> Mesh:
    """Isosurface contour of a 2-D metaball field via marching squares.

    Blob centers are equally spaced around the origin at radius ~0.5.
    The field is sampled on a [-1.5, 1.5] grid at z=0 and thresholded
    to produce a mesh contour.
    """
    # Place blob centers evenly around a circle
    centers: list[tuple[float, float]] = []
    for i in range(n_blobs):
        angle = 2.0 * math.pi * i / n_blobs
        centers.append((0.5 * math.cos(angle), 0.5 * math.sin(angle)))

    extent = 1.5
    step = 2.0 * extent / resolution

    def _field(px: float, py: float) -> float:
        total = 0.0
        for cx, cy in centers:
            dx = px - cx
            dy = py - cy
            r2 = dx * dx + dy * dy
            total += 1.0 / (r2 + 0.01)
        return total

    threshold = 4.0

    # Sample scalar field on a grid
    grid: list[list[float]] = []
    for iy in range(resolution + 1):
        row_vals: list[float] = []
        for ix in range(resolution + 1):
            px = -extent + ix * step
            py = -extent + iy * step
            row_vals.append(_field(px, py))
        grid.append(row_vals)

    # Marching squares to extract contour segments
    vertices: list[Vec3] = []
    edges: list[tuple[int, int]] = []
    vert_cache: dict[tuple[int, int, int, int], int] = {}

    def _interp_edge(
        x0: float, y0: float, v0: float,
        x1: float, y1: float, v1: float,
    ) -> Vec3:
        """Linearly interpolate threshold crossing on an edge."""
        denom = v1 - v0
        if abs(denom) < 1e-12:
            t = 0.5
        else:
            t = (threshold - v0) / denom
            t = max(0.0, min(1.0, t))
        return Vec3(x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.0)

    def _get_or_add(edge_key: tuple[int, int, int, int], pt: Vec3) -> int:
        if edge_key in vert_cache:
            return vert_cache[edge_key]
        idx = len(vertices)
        vertices.append(pt)
        vert_cache[edge_key] = idx
        return idx

    # Marching-squares edge table (4 bits -> list of edge pairs)
    # Edge numbering: 0=bottom, 1=right, 2=top, 3=left
    _MS_EDGES: dict[int, list[tuple[int, int]]] = {
        0: [], 15: [],
        1: [(3, 0)], 14: [(3, 0)],
        2: [(0, 1)], 13: [(0, 1)],
        3: [(3, 1)], 12: [(3, 1)],
        4: [(1, 2)], 11: [(1, 2)],
        5: [(3, 0), (1, 2)],
        6: [(0, 2)], 9: [(0, 2)],
        7: [(3, 2)], 8: [(3, 2)],
        10: [(0, 1), (2, 3)],
    }

    for iy in range(resolution):
        for ix in range(resolution):
            # Corner values (BL, BR, TR, TL)
            v_bl = grid[iy][ix]
            v_br = grid[iy][ix + 1]
            v_tr = grid[iy + 1][ix + 1]
            v_tl = grid[iy + 1][ix]

            # Corner coords
            x0 = -extent + ix * step
            x1 = x0 + step
            y0 = -extent + iy * step
            y1 = y0 + step

            # Classify corners
            case = 0
            if v_bl >= threshold:
                case |= 1
            if v_br >= threshold:
                case |= 2
            if v_tr >= threshold:
                case |= 4
            if v_tl >= threshold:
                case |= 8

            edge_segments = _MS_EDGES.get(case, [])
            for e_a, e_b in edge_segments:
                # Compute interpolated vertex on each edge
                def _edge_pt(e: int) -> tuple[tuple[int, int, int, int], Vec3]:
                    if e == 0:  # bottom: BL -> BR
                        key = (ix, iy, ix + 1, iy)
                        return key, _interp_edge(x0, y0, v_bl, x1, y0, v_br)
                    elif e == 1:  # right: BR -> TR
                        key = (ix + 1, iy, ix + 1, iy + 1)
                        return key, _interp_edge(x1, y0, v_br, x1, y1, v_tr)
                    elif e == 2:  # top: TL -> TR
                        key = (ix, iy + 1, ix + 1, iy + 1)
                        return key, _interp_edge(x0, y1, v_tl, x1, y1, v_tr)
                    else:  # left: BL -> TL
                        key = (ix, iy, ix, iy + 1)
                        return key, _interp_edge(x0, y0, v_bl, x0, y1, v_tl)

                key_a, pt_a = _edge_pt(e_a)
                key_b, pt_b = _edge_pt(e_b)
                idx_a = _get_or_add(key_a, pt_a)
                idx_b = _get_or_add(key_b, pt_b)
                edges.append((idx_a, idx_b))

    # Build faces from closed contour loops (approximate with triangles
    # to the centroid) so the mesh can also be rendered as filled.
    faces: list[tuple[int, ...]] = []
    if vertices:
        centroid = Vec3(0.0, 0.0, 0.0)
        c_idx = len(vertices)
        vertices.append(centroid)
        for a, b in edges:
            faces.append((c_idx, a, b))

    mesh = Mesh(vertices=vertices, edges=edges, faces=faces)
    mesh.compute_normals()
    return mesh


# =====================================================================
# 7. Intersecting Solids  (material_collision)
# =====================================================================

def make_intersecting_solids() -> tuple[Mesh, Mesh]:
    """Return a (cube, octahedron) pair, both centered at the origin.

    Both have a bounding radius of ~1.0.  The renderer rotates them
    independently to produce the intersection effect.
    """
    cube = _make_cube(1.0)
    octa = _make_octahedron(1.0)
    return cube, octa


def _make_cube(radius: float) -> Mesh:
    """Axis-aligned cube with bounding radius *radius*.

    ``radius`` is the half-diagonal of the cube (distance from origin
    to a vertex).  Edge half-length = radius / sqrt(3).
    """
    s = radius / math.sqrt(3.0)
    signs: list[float] = [-s, s]
    vertices = [Vec3(x, y, z) for x, y, z in product(signs, repeat=3)]
    # 8 vertices ordered: 0(-,-,-) 1(-,-,+) 2(-,+,-) 3(-,+,+)
    #                      4(+,-,-) 5(+,-,+) 6(+,+,-) 7(+,+,+)
    faces: list[tuple[int, ...]] = [
        (0, 4, 6, 2),  # front  (-z face)
        (5, 1, 3, 7),  # back   (+z face)
        (0, 1, 5, 4),  # bottom (-y face)
        (2, 6, 7, 3),  # top    (+y face)
        (0, 2, 3, 1),  # left   (-x face)
        (4, 5, 7, 6),  # right  (+x face)
    ]
    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


def _make_octahedron(radius: float) -> Mesh:
    """Regular octahedron with vertices at distance *radius*."""
    r = radius
    vertices = [
        Vec3(r, 0.0, 0.0),
        Vec3(-r, 0.0, 0.0),
        Vec3(0.0, r, 0.0),
        Vec3(0.0, -r, 0.0),
        Vec3(0.0, 0.0, r),
        Vec3(0.0, 0.0, -r),
    ]
    faces: list[tuple[int, ...]] = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


# =====================================================================
# 8. Wireframe Organism  (specimen)
# =====================================================================

def make_wireframe_organism(subdivisions: int = 2) -> Mesh:
    """Subdivided icosphere intended for wireframe-only rendering.

    The returned mesh has ``edges_only`` set to ``True`` as a hint
    to the renderer (no face fill).
    """
    mesh = make_icosahedron(subdivisions=subdivisions)
    # Mark as wireframe hint (duck-typed attribute)
    mesh.edges_only = True  # type: ignore[attr-defined]
    return mesh


# =====================================================================
# 9. Torus  (minimal_object)
# =====================================================================

def make_torus(
    R: float = 0.8,
    r: float = 0.3,
    u_segments: int = 24,
    v_segments: int = 16,
) -> Mesh:
    """Parametric torus with face-shaded quads (split into triangles).

    Parameters
    ----------
    R : float
        Major radius (center of tube to center of torus).
    r : float
        Minor radius (radius of the tube).
    u_segments : int
        Slices around the torus ring.
    v_segments : int
        Slices around the tube cross-section.
    """
    vertices: list[Vec3] = []
    for i in range(u_segments):
        u = 2.0 * math.pi * i / u_segments
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        for j in range(v_segments):
            v = 2.0 * math.pi * j / v_segments
            cos_v = math.cos(v)
            sin_v = math.sin(v)
            x = (R + r * cos_v) * cos_u
            y = (R + r * cos_v) * sin_u
            z = r * sin_v
            vertices.append(Vec3(x, y, z))

    faces: list[tuple[int, ...]] = []
    for i in range(u_segments):
        i_next = (i + 1) % u_segments
        for j in range(v_segments):
            j_next = (j + 1) % v_segments
            a = i * v_segments + j
            b = i_next * v_segments + j
            c = i_next * v_segments + j_next
            d = i * v_segments + j_next
            # Two triangles per quad
            faces.append((a, b, c))
            faces.append((a, c, d))

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


# =====================================================================
# 10. Lorenz Attractor  (abstract_field)
# =====================================================================

def make_lorenz_attractor(
    steps: int = 5000,
    dt: float = 0.005,
) -> PointCloud:
    """Integrate the Lorenz system and return a normalized point cloud.

    Trail brightness: newest points = 1.0, oldest = 0.1.
    The entire cloud is normalized to fit within radius ~1.5.
    """
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    x, y, z = 1.0, 1.0, 1.0
    points: list[Vec3] = []
    for _ in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        points.append(Vec3(x, y, z))

    # Find bounding radius and normalize to ~1.5
    max_r = max((p.length() for p in points), default=1.0)
    if max_r < 1e-10:
        max_r = 1.0
    scale = 1.5 / max_r

    cloud = PointCloud()
    n = len(points)
    for i, pt in enumerate(points):
        bright = 0.1 + 0.9 * (i / max(n - 1, 1))
        cloud.add(pt * scale, bright)
    return cloud


# =====================================================================
# 11. Split Morph Pair  (temporal_diptych)
# =====================================================================

def make_split_morph_pair() -> tuple[Mesh, Mesh]:
    """Return (sphere, cube) with matched vertex counts.

    The UV sphere is constructed with a vertex count that matches a
    subdivided cube so that the renderer can smoothly interpolate
    vertex positions between the two shapes.
    """
    # Use a UV sphere with rings x sectors to match the cube vertex count.
    # A subdivided cube with 2 subdivisions per edge gives (2+1)^2 * 6 = 54
    # vertices.  We match that with a UV sphere of 6 rings x 9 sectors = 54
    # + 2 poles = 56.  For practical vertex matching we'll use 6x9 = 54
    # non-pole vertices + 2 poles = 56.
    # Simplify: use n_rings=7, n_sectors=8 => 7*8 + 2 = 58 vertices.
    # Then the cube has the same count via careful subdivision.

    n_rings = 7
    n_sectors = 8
    sphere = _make_uv_sphere(n_rings, n_sectors, radius=1.0)
    target_count = sphere.vertex_count
    cube = _make_subdivided_cube_matched(target_count, radius=1.0)
    return sphere, cube


def _make_uv_sphere(n_rings: int, n_sectors: int, radius: float) -> Mesh:
    """UV sphere with poles.  Total verts = n_rings * n_sectors + 2."""
    vertices: list[Vec3] = []
    # North pole
    vertices.append(Vec3(0.0, radius, 0.0))
    for i in range(1, n_rings + 1):
        phi = math.pi * i / (n_rings + 1)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for j in range(n_sectors):
            theta = 2.0 * math.pi * j / n_sectors
            x = radius * sin_phi * math.cos(theta)
            y = radius * cos_phi
            z = radius * sin_phi * math.sin(theta)
            vertices.append(Vec3(x, y, z))
    # South pole
    vertices.append(Vec3(0.0, -radius, 0.0))

    faces: list[tuple[int, ...]] = []
    # Top cap (north pole to first ring)
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors
        faces.append((0, 1 + j, 1 + j_next))
    # Middle bands
    for i in range(n_rings - 1):
        ring_start = 1 + i * n_sectors
        next_ring_start = ring_start + n_sectors
        for j in range(n_sectors):
            j_next = (j + 1) % n_sectors
            a = ring_start + j
            b = ring_start + j_next
            c = next_ring_start + j_next
            d = next_ring_start + j
            faces.append((a, d, c))
            faces.append((a, c, b))
    # Bottom cap (last ring to south pole)
    south = len(vertices) - 1
    last_ring = 1 + (n_rings - 1) * n_sectors
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors
        faces.append((last_ring + j, south, last_ring + j_next))

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


def _make_subdivided_cube_matched(target_count: int, radius: float) -> Mesh:
    """Cube with vertices redistributed to roughly match *target_count*.

    Vertices are placed on the surface of a cube with bounding radius
    *radius*, then projected onto the enclosing sphere of the same
    radius so lerp between sphere<->cube is visually smooth.
    """
    s = radius / math.sqrt(3.0)
    # Determine subdivision level per face to approach target_count.
    # Each face of a cube gets (n+1)^2 vertices; 6 faces share edges,
    # but for simplicity we oversample slightly.
    n = max(1, int(math.sqrt(target_count / 6.0)))

    vertices: list[Vec3] = []
    vert_map: dict[tuple[float, float, float], int] = {}

    def _add(v: Vec3) -> int:
        key = (round(v.x, 6), round(v.y, 6), round(v.z, 6))
        if key in vert_map:
            return vert_map[key]
        idx = len(vertices)
        vertices.append(v)
        vert_map[key] = idx
        return idx

    faces: list[tuple[int, ...]] = []

    # Six face orientations: (normal_axis, sign)
    face_defs: list[tuple[int, float]] = [
        (0,  1.0), (0, -1.0),  # +X, -X
        (1,  1.0), (1, -1.0),  # +Y, -Y
        (2,  1.0), (2, -1.0),  # +Z, -Z
    ]

    for axis, sign in face_defs:
        for i in range(n):
            for j in range(n):
                # Parametric (u, v) on this face in [-s, s]
                u0 = -s + 2.0 * s * i / n
                u1 = -s + 2.0 * s * (i + 1) / n
                v0 = -s + 2.0 * s * j / n
                v1 = -s + 2.0 * s * (j + 1) / n

                def _face_pt(u: float, v: float) -> Vec3:
                    coords = [0.0, 0.0, 0.0]
                    coords[axis] = s * sign
                    # Map u, v to the other two axes
                    other = [k for k in range(3) if k != axis]
                    coords[other[0]] = u
                    coords[other[1]] = v
                    return Vec3(coords[0], coords[1], coords[2])

                p00 = _face_pt(u0, v0)
                p10 = _face_pt(u1, v0)
                p11 = _face_pt(u1, v1)
                p01 = _face_pt(u0, v1)

                a = _add(p00)
                b = _add(p10)
                c = _add(p11)
                d = _add(p01)
                faces.append((a, b, c))
                faces.append((a, c, d))

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


# =====================================================================
# 12. Corridor  (liminal)
# =====================================================================

def make_corridor(
    n_frames: int = 12,
    depth_spacing: float = 0.3,
) -> Mesh:
    """Nested rectangular frames receding along Z.

    Wireframe only -- no faces.  Successive frames are connected by
    corner-to-corner longitudinal edges.
    """
    # Each frame is a rectangle in XY at a particular Z depth.
    # Frame size shrinks slightly with depth to amplify perspective.
    vertices: list[Vec3] = []
    edges: list[tuple[int, int]] = []

    half_w = 1.0
    half_h = 0.7
    shrink = 0.92  # per-frame size decay

    total_depth = n_frames * depth_spacing
    z_start = -total_depth / 2.0

    for i in range(n_frames):
        z = z_start + i * depth_spacing
        w = half_w * (shrink ** i)
        h = half_h * (shrink ** i)
        base = len(vertices)
        # 4 corners: TL, TR, BR, BL
        vertices.append(Vec3(-w,  h, z))  # 0 TL
        vertices.append(Vec3( w,  h, z))  # 1 TR
        vertices.append(Vec3( w, -h, z))  # 2 BR
        vertices.append(Vec3(-w, -h, z))  # 3 BL

        # Frame rectangle edges
        edges.append((base + 0, base + 1))
        edges.append((base + 1, base + 2))
        edges.append((base + 2, base + 3))
        edges.append((base + 3, base + 0))

        # Connect to previous frame's corners
        if i > 0:
            prev_base = base - 4
            for c in range(4):
                edges.append((prev_base + c, base + c))

    mesh = Mesh(vertices=vertices, edges=edges)
    return mesh


# =====================================================================
# 13. Fragmenting Solid  (ruin_state)
# =====================================================================

def make_fragmenting_solid(
    n_fragments: int = 8,
) -> tuple[Mesh, list[list[int]]]:
    """Cube mesh + list of face-groups (fragments).

    Each fragment is a list of face indices grouped by the spatial
    position of their centroid.  The renderer applies per-fragment
    offsets to create the shattering effect.
    """
    cube = _make_cube(1.0)

    # Subdivide each quad face into smaller triangles so there are
    # enough faces to meaningfully fragment.
    subdivided = _subdivide_mesh_flat(cube, splits=2)

    # Group faces into fragments by spatial clustering of their
    # centroid.  We use a simple angular+height partitioning.
    face_centroids: list[Vec3] = []
    for face in subdivided.faces:
        cx = sum(subdivided.vertices[i].x for i in face) / len(face)
        cy = sum(subdivided.vertices[i].y for i in face) / len(face)
        cz = sum(subdivided.vertices[i].z for i in face) / len(face)
        face_centroids.append(Vec3(cx, cy, cz))

    # Assign each face to a fragment bucket via hash of rounded centroid
    fragments: list[list[int]] = [[] for _ in range(n_fragments)]
    for fi, cen in enumerate(face_centroids):
        angle = math.atan2(cen.z, cen.x)
        # Map angle + height into a bucket index
        bucket = int(
            ((angle + math.pi) / (2.0 * math.pi) * (n_fragments // 2))
            + (0 if cen.y < 0 else n_fragments // 2)
        ) % n_fragments
        fragments[bucket].append(fi)

    # Ensure every fragment has at least one face
    orphaned: list[int] = []
    for fi, frag in enumerate(fragments):
        if not frag:
            orphaned.append(fi)
    if orphaned:
        # Redistribute from the largest fragment
        for empty_idx in orphaned:
            largest = max(range(n_fragments), key=lambda k: len(fragments[k]))
            if len(fragments[largest]) > 1:
                fragments[empty_idx].append(fragments[largest].pop())

    return subdivided, fragments


def _subdivide_mesh_flat(mesh: Mesh, splits: int = 1) -> Mesh:
    """Subdivide every triangle/quad face into smaller triangles.

    Unlike icosphere subdivision, new vertices are NOT projected onto
    a sphere -- they stay on the original face plane.
    """
    vertices = list(mesh.vertices)
    faces: list[tuple[int, ...]] = list(mesh.faces)

    for _ in range(splits):
        new_faces: list[tuple[int, ...]] = []
        midpoint_cache: dict[tuple[int, int], int] = {}

        def _mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                return midpoint_cache[key]
            mid = (vertices[a] + vertices[b]) * 0.5
            idx = len(vertices)
            vertices.append(mid)
            midpoint_cache[key] = idx
            return idx

        for face in faces:
            if len(face) == 3:
                a, b, c = face
                ab = _mid(a, b)
                bc = _mid(b, c)
                ca = _mid(c, a)
                new_faces.extend([
                    (a, ab, ca),
                    (ab, b, bc),
                    (ca, bc, c),
                    (ab, bc, ca),
                ])
            elif len(face) == 4:
                a, b, c, d = face
                # Split quad into 4 quads via midpoints + center
                ab = _mid(a, b)
                bc = _mid(b, c)
                cd = _mid(c, d)
                da = _mid(d, a)
                center_pt = (
                    (vertices[a] + vertices[b] + vertices[c] + vertices[d])
                    * 0.25
                )
                center_idx = len(vertices)
                vertices.append(center_pt)
                # 4 sub-quads as triangle pairs
                new_faces.extend([
                    (a, ab, center_idx),
                    (a, center_idx, da),
                    (ab, b, center_idx),
                    (b, bc, center_idx),
                    (bc, c, center_idx),
                    (c, cd, center_idx),
                    (cd, d, center_idx),
                    (d, da, center_idx),
                ])
            else:
                # Keep as-is for degenerate faces
                new_faces.append(face)
        faces = new_faces

    result = Mesh(vertices=vertices, faces=faces)
    result.compute_normals()
    result.compute_edges_from_faces()
    return result


# =====================================================================
# 14. Mobius Strip  (essence)
# =====================================================================

def make_mobius_strip(
    u_segments: int = 48,
    v_segments: int = 8,
) -> Mesh:
    """Parametric Mobius strip as a triangle-strip mesh.

    x = (1 + (v/2)*cos(u/2)) * cos(u)
    y = (1 + (v/2)*cos(u/2)) * sin(u)
    z = (v/2) * sin(u/2)

    u in [0, 2*pi], v in [-0.4, 0.4].
    Scaled to bounding radius ~1.0.
    """
    v_min = -0.4
    v_max = 0.4

    vertices: list[Vec3] = []
    for i in range(u_segments):
        u = 2.0 * math.pi * i / u_segments
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_u2 = math.cos(u / 2.0)
        sin_u2 = math.sin(u / 2.0)
        for j in range(v_segments + 1):
            v = v_min + (v_max - v_min) * j / v_segments
            half_v = v / 2.0
            x = (1.0 + half_v * cos_u2) * cos_u
            y = (1.0 + half_v * cos_u2) * sin_u
            z = half_v * sin_u2
            vertices.append(Vec3(x, y, z))

    # Scale to bounding radius ~1.0
    max_r = max((v.length() for v in vertices), default=1.0)
    if max_r > 1e-10:
        scale = 1.0 / max_r
        vertices = [v * scale for v in vertices]

    row_len = v_segments + 1
    faces: list[tuple[int, ...]] = []
    for i in range(u_segments):
        i_next = (i + 1) % u_segments
        for j in range(v_segments):
            a = i * row_len + j
            b = i_next * row_len + j

            if i_next == 0:
                # Wrap: the Mobius twist means the v-parameterization
                # is reversed for the closing seam.
                b_twisted = i_next * row_len + (v_segments - j)
                c_twisted = i_next * row_len + (v_segments - (j + 1))
                d = i * row_len + (j + 1)
                faces.append((a, b_twisted, c_twisted))
                faces.append((a, c_twisted, d))
            else:
                c = i_next * row_len + (j + 1)
                d = i * row_len + (j + 1)
                faces.append((a, b, c))
                faces.append((a, c, d))

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()
    mesh.compute_edges_from_faces()
    return mesh


# =====================================================================
# 15. Voxel Grid  (site_decay)
# =====================================================================

def make_voxel_grid(
    sx: int = 8,
    sy: int = 6,
    sz: int = 8,
    fill: float = 0.7,
) -> VoxelGrid:
    """Boolean occupancy grid with random fill at a deterministic seed.

    The seed is derived from the grid dimensions so identical
    parameters always produce the same grid.
    """
    seed = sx * 10000 + sy * 100 + sz
    rng = random.Random(seed)
    total = sx * sy * sz
    cells = [rng.random() < fill for _ in range(total)]
    return VoxelGrid(
        size_x=sx,
        size_y=sy,
        size_z=sz,
        cells=cells,
    )


# =====================================================================
# Template -> Geometry Registry
# =====================================================================

TEMPLATE_GEOMETRY: dict[str, str] = {
    "material_study": "icosahedron",
    "textural_macro": "noise_surface",
    "environmental": "terrain",
    "atmospheric_depth": "particle_nebula",
    "process_state": "metaballs",
    "material_collision": "intersecting_solids",
    "specimen": "wireframe_organism",
    "minimal_object": "torus",
    "abstract_field": "lorenz_attractor",
    "temporal_diptych": "split_morph_pair",
    "liminal": "corridor",
    "ruin_state": "fragmenting_solid",
    "essence": "mobius_strip",
    "site_decay": "voxel_grid",
}
"""Maps each template_id to the geometry factory short name.

The factory function is ``make_{name}`` (e.g. ``make_icosahedron``).
"""

# Convenience mapping from short name -> callable
GEOMETRY_FACTORIES: dict[str, object] = {
    "tesseract": make_tesseract,
    "icosahedron": make_icosahedron,
    "noise_surface": make_noise_surface,
    "terrain": make_terrain,
    "particle_nebula": make_particle_nebula,
    "metaballs": make_metaballs,
    "intersecting_solids": make_intersecting_solids,
    "wireframe_organism": make_wireframe_organism,
    "torus": make_torus,
    "lorenz_attractor": make_lorenz_attractor,
    "split_morph_pair": make_split_morph_pair,
    "corridor": make_corridor,
    "fragmenting_solid": make_fragmenting_solid,
    "mobius_strip": make_mobius_strip,
    "voxel_grid": make_voxel_grid,
}
"""Maps geometry short name -> factory callable for programmatic lookup."""
