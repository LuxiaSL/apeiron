"""Math primitives and lookup tables for the hyperobject renderer.

Provides Vec3, Vec4, Mat4 types and pre-computed trig tables. Every module
in the hyperobject package imports its math from here.

Design: pure-Python with __slots__ for minimal overhead. Optional numpy
batch acceleration is handled at the call site, not in these types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── trig lookup tables ──────────────────────────────────────────────────

_LUT_SIZE: int = 4096
_TWO_PI: float = 2.0 * math.pi
_LUT_SCALE: float = _LUT_SIZE / _TWO_PI
_SIN_LUT: list[float] = [math.sin(i / _LUT_SCALE) for i in range(_LUT_SIZE)]
_COS_LUT: list[float] = [math.cos(i / _LUT_SCALE) for i in range(_LUT_SIZE)]


def fast_sin(angle: float) -> float:
    """Lookup-table sine. ~10x faster than math.sin for bulk transforms."""
    return _SIN_LUT[int(angle * _LUT_SCALE) % _LUT_SIZE]


def fast_cos(angle: float) -> float:
    """Lookup-table cosine."""
    return _COS_LUT[int(angle * _LUT_SCALE) % _LUT_SIZE]


# ── Vec3 ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Vec3:
    """3-component vector (x, y, z)."""

    x: float
    y: float
    z: float

    # ── arithmetic ────────────────────────────────────────────────────

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, s: float) -> Vec3:
        inv = 1.0 / s
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    # ── products ──────────────────────────────────────────────────────

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    # ── magnitude ─────────────────────────────────────────────────────

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> Vec3:
        sq = self.length_sq()
        if sq < 1e-20:
            return Vec3(0.0, 0.0, 0.0)
        inv = 1.0 / math.sqrt(sq)
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    # ── interpolation ─────────────────────────────────────────────────

    def lerp(self, other: Vec3, t: float) -> Vec3:
        u = 1.0 - t
        return Vec3(
            self.x * u + other.x * t,
            self.y * u + other.y * t,
            self.z * u + other.z * t,
        )

    # ── conversion ────────────────────────────────────────────────────

    def to_vec4(self, w: float = 1.0) -> Vec4:
        return Vec4(self.x, self.y, self.z, w)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ── Vec4 ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Vec4:
    """4-component vector (x, y, z, w). Used for homogeneous coordinates
    and 4D hypercube geometry."""

    x: float
    y: float
    z: float
    w: float

    def __add__(self, other: Vec4) -> Vec4:
        return Vec4(
            self.x + other.x, self.y + other.y,
            self.z + other.z, self.w + other.w,
        )

    def __sub__(self, other: Vec4) -> Vec4:
        return Vec4(
            self.x - other.x, self.y - other.y,
            self.z - other.z, self.w - other.w,
        )

    def __mul__(self, s: float) -> Vec4:
        return Vec4(self.x * s, self.y * s, self.z * s, self.w * s)

    def __rmul__(self, s: float) -> Vec4:
        return Vec4(self.x * s, self.y * s, self.z * s, self.w * s)

    def __neg__(self) -> Vec4:
        return Vec4(-self.x, -self.y, -self.z, -self.w)

    def dot(self, other: Vec4) -> float:
        return (
            self.x * other.x + self.y * other.y
            + self.z * other.z + self.w * other.w
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec4:
        sq = self.dot(self)
        if sq < 1e-20:
            return Vec4(0.0, 0.0, 0.0, 0.0)
        inv = 1.0 / math.sqrt(sq)
        return Vec4(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    def lerp(self, other: Vec4, t: float) -> Vec4:
        u = 1.0 - t
        return Vec4(
            self.x * u + other.x * t,
            self.y * u + other.y * t,
            self.z * u + other.z * t,
            self.w * u + other.w * t,
        )

    def to_vec3(self) -> Vec3:
        """Drop w component."""
        return Vec3(self.x, self.y, self.z)

    def perspective_divide(self) -> Vec3:
        """Perspective divide: (x/w, y/w, z/w). Returns Vec3."""
        if abs(self.w) < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        inv = 1.0 / self.w
        return Vec3(self.x * inv, self.y * inv, self.z * inv)


# ── Mat4 ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Mat4:
    """4x4 matrix, row-major, stored as flat 16-element list.

    Convention: column vectors, right-multiply (M @ v).
    Composition: (A @ B) applies B first, then A.
    """

    m: list[float]

    # ── constructors ──────────────────────────────────────────────────

    @staticmethod
    def identity() -> Mat4:
        return Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def translation(tx: float, ty: float, tz: float) -> Mat4:
        return Mat4([
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, ty,
            0.0, 0.0, 1.0, tz,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> Mat4:
        return Mat4([
            sx,  0.0, 0.0, 0.0,
            0.0, sy,  0.0, 0.0,
            0.0, 0.0, sz,  0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def rotation_x(angle: float) -> Mat4:
        s, c = fast_sin(angle), fast_cos(angle)
        return Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, c,   -s,  0.0,
            0.0, s,    c,  0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def rotation_y(angle: float) -> Mat4:
        s, c = fast_sin(angle), fast_cos(angle)
        return Mat4([
            c,   0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
            -s,  0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def rotation_z(angle: float) -> Mat4:
        s, c = fast_sin(angle), fast_cos(angle)
        return Mat4([
            c,   -s,  0.0, 0.0,
            s,    c,  0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
        """Perspective projection matrix.

        fov_y: vertical field of view in radians
        aspect: width / height (in character cells, after aspect correction)
        """
        f = 1.0 / math.tan(fov_y / 2.0)
        nf = 1.0 / (near - far)
        return Mat4([
            f / aspect, 0.0, 0.0,              0.0,
            0.0,        f,   0.0,              0.0,
            0.0,        0.0, (far + near) * nf, 2.0 * far * near * nf,
            0.0,        0.0, -1.0,              0.0,
        ])

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
        """View matrix (world → camera)."""
        fwd = (target - eye).normalized()
        right = fwd.cross(up).normalized()
        true_up = right.cross(fwd)
        return Mat4([
            right.x,    right.y,    right.z,    -right.dot(eye),
            true_up.x,  true_up.y,  true_up.z,  -true_up.dot(eye),
            -fwd.x,     -fwd.y,     -fwd.z,     fwd.dot(eye),
            0.0,        0.0,        0.0,         1.0,
        ])

    # ── operations ────────────────────────────────────────────────────

    def __matmul__(self, other: object) -> Mat4 | Vec4:
        if isinstance(other, Mat4):
            return self._mul_mat(other)
        if isinstance(other, Vec4):
            return self._mul_vec(other)
        return NotImplemented  # type: ignore[return-value]

    def _mul_mat(self, other: Mat4) -> Mat4:
        a, b = self.m, other.m
        r = [0.0] * 16
        for row in range(4):
            r0 = row * 4
            a0, a1, a2, a3 = a[r0], a[r0 + 1], a[r0 + 2], a[r0 + 3]
            for col in range(4):
                r[r0 + col] = (
                    a0 * b[col]
                    + a1 * b[4 + col]
                    + a2 * b[8 + col]
                    + a3 * b[12 + col]
                )
        return Mat4(r)

    def _mul_vec(self, v: Vec4) -> Vec4:
        m = self.m
        return Vec4(
            m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
            m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
            m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w,
            m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w,
        )

    def transform_point(self, v: Vec3) -> Vec3:
        """Transform a point (w=1) and perspective-divide back to Vec3."""
        r = self._mul_vec(v.to_vec4(1.0))
        return r.perspective_divide()

    def transform_direction(self, v: Vec3) -> Vec3:
        """Transform a direction (w=0), no perspective divide."""
        r = self._mul_vec(v.to_vec4(0.0))
        return r.to_vec3()


# ── 4D rotation (for tesseract) ─────────────────────────────────────────


def rotate_4d(v: Vec4, angle_xw: float, angle_yz: float) -> Vec4:
    """Rotate a 4D point in two independent planes (XW and YZ).

    These two rotations create the characteristic 'inside-out fold' of
    the tesseract projection.
    """
    s_xw, c_xw = fast_sin(angle_xw), fast_cos(angle_xw)
    s_yz, c_yz = fast_sin(angle_yz), fast_cos(angle_yz)

    # XW plane rotation
    x1 = v.x * c_xw - v.w * s_xw
    w1 = v.x * s_xw + v.w * c_xw

    # YZ plane rotation
    y1 = v.y * c_yz - v.z * s_yz
    z1 = v.y * s_yz + v.z * c_yz

    return Vec4(x1, y1, z1, w1)


def project_4d_to_3d(v: Vec4, distance: float = 2.5) -> Vec3:
    """Stereographic-like projection from 4D to 3D.

    Points with larger w are projected smaller (farther in 4D).
    """
    denom = distance - v.w
    if abs(denom) < 1e-10:
        denom = 1e-10
    factor = 1.0 / denom
    return Vec3(v.x * factor, v.y * factor, v.z * factor)


# ── utility ─────────────────────────────────────────────────────────────

ORIGIN = Vec3(0.0, 0.0, 0.0)
UP = Vec3(0.0, 1.0, 0.0)
FORWARD = Vec3(0.0, 0.0, -1.0)
RIGHT = Vec3(1.0, 0.0, 0.0)


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def lerp_f(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
