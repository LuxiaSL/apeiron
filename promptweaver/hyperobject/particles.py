"""Ambient particle systems for the hyperobject renderer.

Particles float around the main geometry in 3D space. The renderer
projects them to screen coordinates and composites them behind (or in
front of) the solid geometry.

Each factory function returns a configured ``ParticleSystem`` (or
subclass) with type-specific spawn and tick behaviour.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .lut import Vec3, clamp, fast_cos, fast_sin


# ── particle ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class Particle:
    """A single ambient particle in 3D space."""

    pos: Vec3
    vel: Vec3
    life: float        # remaining lifetime, 0.0-1.0 (decreasing)
    brightness: float  # visual brightness, 0.0-1.0
    char: str          # character to render


# ── base system ────────────────────────────────────────────────────────


def _random_on_shell(r_min: float = 1.5, r_max: float = 3.0) -> Vec3:
    """Return a random point in the spherical shell [r_min, r_max]."""
    # Uniform random direction.
    theta = random.uniform(0.0, 2.0 * math.pi)
    phi = math.acos(random.uniform(-1.0, 1.0))
    r = random.uniform(r_min, r_max)
    sp = math.sin(phi)
    return Vec3(
        r * sp * math.cos(theta),
        r * sp * math.sin(theta),
        r * math.cos(phi),
    )


@dataclass
class ParticleSystem:
    """Base particle system with spawn / tick lifecycle.

    Subclasses override ``spawn()`` for custom particle behaviour and
    ``_update_particle()`` for per-frame motion.
    """

    particles: list[Particle] = field(default_factory=list)
    max_particles: int = 100
    spawn_rate: float = 2.0  # particles per tick

    # Internal accumulator for fractional spawn counts.
    _spawn_accum: float = field(default=0.0, init=False, repr=False)

    # ── lifecycle ──────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Advance the system by *dt* seconds.

        Updates all live particles, removes dead ones, and spawns new
        ones according to ``spawn_rate``.
        """
        dt = max(dt, 0.0)

        # Update existing particles.
        alive: list[Particle] = []
        for p in self.particles:
            p.life -= dt * 0.3  # base decay rate
            if p.life <= 0.0:
                continue
            self._update_particle(p, dt)
            alive.append(p)
        self.particles = alive

        # Spawn new particles.
        self._spawn_accum += self.spawn_rate * dt
        while self._spawn_accum >= 1.0 and len(self.particles) < self.max_particles:
            try:
                self.particles.append(self.spawn())
            except Exception:
                break
            self._spawn_accum -= 1.0
        # Clamp accumulator to avoid runaway spawning after long pauses.
        self._spawn_accum = min(self._spawn_accum, 3.0)

    def spawn(self) -> Particle:
        """Create one new particle.  Override in subclasses."""
        return Particle(
            pos=_random_on_shell(),
            vel=Vec3(0.0, 0.0, 0.0),
            life=1.0,
            brightness=1.0,
            char="\u00b7",  # ·
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        """Per-frame update for a single particle.  Override for custom motion."""
        p.pos = p.pos + p.vel * dt
        p.brightness = clamp(p.life, 0.0, 1.0)


# ── fog ────────────────────────────────────────────────────────────────


class _FogSystem(ParticleSystem):
    """Slow upward drift, concentrated at bottom."""

    _chars: str = "\u2591\u2592"  # ░▒

    def spawn(self) -> Particle:
        pos = _random_on_shell(1.5, 3.0)
        # Bias downward: most fog sits near the bottom.
        pos = Vec3(pos.x, -abs(pos.y) * 0.8, pos.z)
        return Particle(
            pos=pos,
            vel=Vec3(
                random.uniform(-0.02, 0.02),
                random.uniform(0.05, 0.15),
                random.uniform(-0.02, 0.02),
            ),
            life=random.uniform(0.6, 1.0),
            brightness=random.uniform(0.2, 0.5),
            char=random.choice(self._chars),
        )


class _SmokeSystem(ParticleSystem):
    """Turbulent drift with noise added to velocity each frame."""

    _chars: str = "\u00b7\u2219\u00b0"  # ·∙°

    def spawn(self) -> Particle:
        pos = _random_on_shell(1.5, 2.5)
        return Particle(
            pos=pos,
            vel=Vec3(
                random.uniform(-0.1, 0.1),
                random.uniform(0.08, 0.2),
                random.uniform(-0.1, 0.1),
            ),
            life=random.uniform(0.5, 1.0),
            brightness=random.uniform(0.3, 0.7),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        # Turbulence: random velocity perturbation each frame.
        turbulence = Vec3(
            random.gauss(0.0, 0.15),
            random.gauss(0.0, 0.08),
            random.gauss(0.0, 0.15),
        )
        p.vel = p.vel + turbulence * dt
        p.pos = p.pos + p.vel * dt
        p.brightness = clamp(p.life * 0.7, 0.0, 1.0)


class _DustSystem(ParticleSystem):
    """Very slow Brownian motion with occasional brightness flash."""

    _chars: str = "\u00b7\u2219"  # ·∙

    def spawn(self) -> Particle:
        return Particle(
            pos=_random_on_shell(1.5, 3.0),
            vel=Vec3(
                random.gauss(0.0, 0.01),
                random.gauss(0.0, 0.01),
                random.gauss(0.0, 0.01),
            ),
            life=random.uniform(0.7, 1.0),
            brightness=random.uniform(0.1, 0.3),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        # Brownian motion: tiny random displacement.
        jitter = Vec3(
            random.gauss(0.0, 0.03),
            random.gauss(0.0, 0.03),
            random.gauss(0.0, 0.03),
        )
        p.vel = jitter
        p.pos = p.pos + p.vel * dt
        # Occasional brightness flash.
        if random.random() < 0.02:
            p.brightness = clamp(random.uniform(0.6, 1.0), 0.0, 1.0)
        else:
            p.brightness = clamp(p.life * 0.3, 0.0, 1.0)


class _EmberSystem(ParticleSystem):
    """Fast upward with deceleration; bright head that fades."""

    _chars: str = "\u00b7\u2219\u2022"  # ·∙•

    def spawn(self) -> Particle:
        pos = _random_on_shell(1.0, 2.0)
        # Bias downward origin for upward travel.
        pos = Vec3(pos.x, -abs(pos.y), pos.z)
        return Particle(
            pos=pos,
            vel=Vec3(
                random.uniform(-0.05, 0.05),
                random.uniform(0.5, 1.2),
                random.uniform(-0.05, 0.05),
            ),
            life=random.uniform(0.5, 1.0),
            brightness=1.0,
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        # Deceleration: drag on the Y (upward) component.
        drag = 0.92
        p.vel = Vec3(p.vel.x * drag, p.vel.y * drag, p.vel.z * drag)
        p.pos = p.pos + p.vel * dt
        # Bright head, fading tail.
        p.brightness = clamp(p.life, 0.0, 1.0)
        # Ember chars: shrink as they cool.
        if p.life < 0.3:
            p.char = "\u00b7"  # ·
        elif p.life < 0.6:
            p.char = "\u2219"  # ∙


class _RainSystem(ParticleSystem):
    """Fast downward streaks."""

    _chars: str = "\u2502:"  # │:

    def spawn(self) -> Particle:
        # Spawn above the scene, spread across X and Z.
        return Particle(
            pos=Vec3(
                random.uniform(-3.0, 3.0),
                random.uniform(2.5, 4.0),
                random.uniform(-3.0, 3.0),
            ),
            vel=Vec3(
                random.uniform(-0.02, 0.02),
                random.uniform(-2.5, -1.5),
                random.uniform(-0.02, 0.02),
            ),
            life=random.uniform(0.4, 0.8),
            brightness=random.uniform(0.5, 1.0),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        p.pos = p.pos + p.vel * dt
        p.brightness = clamp(p.life * 0.8, 0.0, 1.0)


class _SnowSystem(ParticleSystem):
    """Slow diagonal drift with sine oscillation on X."""

    _chars: str = "\u00b7*"  # ·*
    # Per-particle phase stored in vel.z (repurposed for oscillation phase).

    def spawn(self) -> Particle:
        phase = random.uniform(0.0, 2.0 * math.pi)
        return Particle(
            pos=Vec3(
                random.uniform(-3.0, 3.0),
                random.uniform(2.5, 4.0),
                random.uniform(-2.0, 2.0),
            ),
            vel=Vec3(
                random.uniform(-0.1, 0.1),
                random.uniform(-0.4, -0.15),
                phase,  # abusing vel.z as oscillation phase
            ),
            life=random.uniform(0.6, 1.0),
            brightness=random.uniform(0.4, 0.8),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        phase = p.vel.z + dt * 2.0
        p.vel = Vec3(p.vel.x, p.vel.y, phase)
        lateral = fast_sin(phase) * 0.3
        p.pos = Vec3(
            p.pos.x + (p.vel.x + lateral) * dt,
            p.pos.y + p.vel.y * dt,
            p.pos.z,
        )
        p.brightness = clamp(p.life * 0.6, 0.0, 1.0)


class _SporeSystem(ParticleSystem):
    """Very slow random walk."""

    _chars: str = "\u00b7\u2218\u25cb"  # ·∘○

    def spawn(self) -> Particle:
        return Particle(
            pos=_random_on_shell(1.5, 2.5),
            vel=Vec3(0.0, 0.0, 0.0),
            life=random.uniform(0.7, 1.0),
            brightness=random.uniform(0.2, 0.6),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        # Pure random walk: tiny step each frame.
        step = Vec3(
            random.gauss(0.0, 0.05),
            random.gauss(0.0, 0.05),
            random.gauss(0.0, 0.05),
        )
        p.pos = p.pos + step * dt
        p.brightness = clamp(p.life * 0.5, 0.0, 1.0)


class _DataSystem(ParticleSystem):
    """Vertical streams of hex characters (matrix-rain-like, 3D positioned)."""

    _chars: str = "0123456789abcdef"

    def spawn(self) -> Particle:
        return Particle(
            pos=Vec3(
                random.uniform(-3.0, 3.0),
                random.uniform(2.5, 4.0),
                random.uniform(-2.0, 2.0),
            ),
            vel=Vec3(0.0, random.uniform(-1.5, -0.5), 0.0),
            life=random.uniform(0.3, 0.8),
            brightness=random.uniform(0.5, 1.0),
            char=random.choice(self._chars),
        )

    def _update_particle(self, p: Particle, dt: float) -> None:
        p.pos = p.pos + p.vel * dt
        p.brightness = clamp(p.life, 0.0, 1.0)
        # Occasionally swap to a new hex char for flicker effect.
        if random.random() < 0.15:
            p.char = random.choice(self._chars)


# ── factory functions ──────────────────────────────────────────────────


def make_fog_particles(max_p: int = 80) -> ParticleSystem:
    """Create a fog particle system (slow upward drift, bottom-heavy)."""
    return _FogSystem(max_particles=max_p, spawn_rate=15.0)


def make_smoke_particles(max_p: int = 60) -> ParticleSystem:
    """Create a smoke particle system (turbulent drift)."""
    return _SmokeSystem(max_particles=max_p, spawn_rate=12.0)


def make_dust_particles(max_p: int = 40) -> ParticleSystem:
    """Create a dust particle system (Brownian motion, occasional flash)."""
    return _DustSystem(max_particles=max_p, spawn_rate=8.0)


def make_ember_particles(max_p: int = 50) -> ParticleSystem:
    """Create an ember particle system (fast upward, decelerating, bright head)."""
    return _EmberSystem(max_particles=max_p, spawn_rate=15.0)


def make_rain_particles(max_p: int = 100) -> ParticleSystem:
    """Create a rain particle system (fast downward streaks)."""
    return _RainSystem(max_particles=max_p, spawn_rate=30.0)


def make_snow_particles(max_p: int = 50) -> ParticleSystem:
    """Create a snow particle system (slow diagonal with sine oscillation)."""
    return _SnowSystem(max_particles=max_p, spawn_rate=10.0)


def make_spore_particles(max_p: int = 30) -> ParticleSystem:
    """Create a spore particle system (very slow random walk)."""
    return _SporeSystem(max_particles=max_p, spawn_rate=6.0)


def make_data_particles(max_p: int = 60) -> ParticleSystem:
    """Create a data-stream particle system (vertical hex streams)."""
    return _DataSystem(max_particles=max_p, spawn_rate=20.0)


# ── word -> particle system mapping ────────────────────────────────────

_FACTORIES: list[tuple[str, object]] = [
    ("fog", make_fog_particles),
    ("smoke", make_smoke_particles),
    ("dust", make_dust_particles),
    ("ember", make_ember_particles),
    ("rain", make_rain_particles),
    ("snow", make_snow_particles),
    ("spore", make_spore_particles),
    ("data", make_data_particles),
]


def _stable_hash(word: str) -> int:
    """FNV-1a hash for deterministic cross-platform results."""
    h: int = 0x811C9DC5
    for ch in word:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def particle_system_for_word(word: str) -> ParticleSystem:
    """Deterministically select a particle system for an ``atmosphere_field`` word.

    Known keywords (e.g. ``"fog"``, ``"rain"``) are matched directly.
    Unknown words fall through to a hash-based selection.
    """
    key = word.lower().strip()

    # Direct keyword matches.
    _keyword_map: dict[str, int] = {
        "fog": 0,
        "smoke": 1,
        "dust": 2,
        "dust motes": 2,
        "dust_motes": 2,
        "ember": 3,
        "embers": 3,
        "rain": 4,
        "snow": 5,
        "spore": 6,
        "spores": 6,
        "data": 7,
        "data_stream": 7,
        "data stream": 7,
    }

    idx = _keyword_map.get(key)
    if idx is None:
        idx = _stable_hash(key) % len(_FACTORIES)

    _name, factory = _FACTORIES[idx]
    return factory()  # type: ignore[operator]
