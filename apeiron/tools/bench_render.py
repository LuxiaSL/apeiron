#!/usr/bin/env python3
"""Render pipeline benchmark.

Profiles per-frame render time across all geometry types at representative
terminal sizes. Reports median, p95, and breakdown by stage.

Usage:
    uv run python -m apeiron.tools.bench_render
    uv run python -m apeiron.tools.bench_render --width 200 --height 60 --frames 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apeiron.hyperobject.scene import Scene, GeomKind
from apeiron.hyperobject.rasterizer import (
    AsciiRasterizer,
    TorusSampler,
    MobiusSampler,
)
from apeiron.hyperobject.transform import Camera, ProjectionContext
from apeiron.hyperobject.lut import Vec3, Mat4
from apeiron.hyperobject import primitives


# ── benchmark config ──────────────────────────────────────────────────────


@dataclass
class BenchResult:
    """Timing results for a single geometry type."""

    name: str
    frame_times_ms: list[float] = field(default_factory=list)
    clear_ms: list[float] = field(default_factory=list)
    render_ms: list[float] = field(default_factory=list)
    postfx_ms: list[float] = field(default_factory=list)
    to_text_ms: list[float] = field(default_factory=list)

    @property
    def total_median(self) -> float:
        return statistics.median(self.frame_times_ms) if self.frame_times_ms else 0.0

    @property
    def total_p95(self) -> float:
        if not self.frame_times_ms:
            return 0.0
        sorted_times = sorted(self.frame_times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def render_median(self) -> float:
        return statistics.median(self.render_ms) if self.render_ms else 0.0

    @property
    def to_text_median(self) -> float:
        return statistics.median(self.to_text_ms) if self.to_text_ms else 0.0

    @property
    def fps_at_median(self) -> float:
        med = self.total_median
        return 1000.0 / med if med > 0 else float("inf")


# ── geometry setup ────────────────────────────────────────────────────────


def make_scene_torus() -> Scene:
    """SURFACE_DIRECT — torus (densest sample count)."""
    scene = Scene()
    scene.geom_kind = GeomKind.SURFACE_DIRECT
    scene.surface_sampler = TorusSampler(R=1.0, r=0.5)
    return scene


def make_scene_mobius() -> Scene:
    """SURFACE_DIRECT — Möbius strip."""
    scene = Scene()
    scene.geom_kind = GeomKind.SURFACE_DIRECT
    scene.surface_sampler = MobiusSampler()
    return scene


def make_scene_icosahedron() -> Scene:
    """MESH_FILLED — subdivided icosahedron."""
    scene = Scene()
    scene.geom_kind = GeomKind.MESH_FILLED
    scene.mesh = primitives.make_icosahedron(subdivisions=2)
    return scene


def make_scene_terrain() -> Scene:
    """HEIGHTMAP — terrain."""
    scene = Scene()
    scene.geom_kind = GeomKind.HEIGHTMAP
    scene.heightmap = primitives.make_terrain()
    return scene


def make_scene_lorenz() -> Scene:
    """POINT_CLOUD — Lorenz attractor."""
    scene = Scene()
    scene.geom_kind = GeomKind.POINT_CLOUD
    scene.cloud = primitives.make_lorenz_attractor()
    return scene


def make_scene_tesseract() -> Scene:
    """TESSERACT — 4D hypercube wireframe."""
    scene = Scene()
    scene.geom_kind = GeomKind.TESSERACT
    verts, edges = primitives.make_tesseract()
    scene.tesseract_verts = verts
    scene.tesseract_edges = edges
    return scene


def make_scene_wireframe() -> Scene:
    """MESH_WIREFRAME — specimen organism."""
    scene = Scene()
    scene.geom_kind = GeomKind.MESH_WIREFRAME
    scene.mesh = primitives.make_wireframe_organism()
    return scene


def make_scene_voxels() -> Scene:
    """VOXEL_GRID — site_decay."""
    scene = Scene()
    scene.geom_kind = GeomKind.VOXEL_GRID
    scene.voxels = primitives.make_voxel_grid()
    return scene


SCENARIOS: dict[str, callable] = {
    "torus (surface_direct)": make_scene_torus,
    "möbius (surface_direct)": make_scene_mobius,
    "icosahedron (mesh_filled)": make_scene_icosahedron,
    "terrain (heightmap)": make_scene_terrain,
    "lorenz (point_cloud)": make_scene_lorenz,
    "tesseract (wireframe_4d)": make_scene_tesseract,
    "organism (wireframe)": make_scene_wireframe,
    "voxels (voxel_grid)": make_scene_voxels,
}


# ── benchmark runner ──────────────────────────────────────────────────────


def bench_scenario(
    name: str,
    scene: Scene,
    width: int,
    height: int,
    n_frames: int,
    warmup: int = 5,
) -> BenchResult:
    """Time n_frames of rendering for a given scene."""
    rast = AsciiRasterizer(width, height)
    result = BenchResult(name=name)
    dt = 1.0 / 18.0  # simulate 18fps tick

    # Warmup (let caches settle, JIT-like effects in CPython)
    for _ in range(warmup):
        scene.anim.tick(dt)
        rast.clear()
        scene.render(rast)
        rast.grid.to_rich_text()

    for _ in range(n_frames):
        scene.anim.tick(dt)

        t0 = time.perf_counter()
        rast.clear()
        t1 = time.perf_counter()
        scene.render(rast)
        t2 = time.perf_counter()
        text = rast.grid.to_rich_text()
        t3 = time.perf_counter()

        result.clear_ms.append((t1 - t0) * 1000)
        result.render_ms.append((t2 - t1) * 1000)
        result.to_text_ms.append((t3 - t2) * 1000)
        result.frame_times_ms.append((t3 - t0) * 1000)

    return result


# ── display ───────────────────────────────────────────────────────────────


def print_results(results: list[BenchResult], width: int, height: int) -> None:
    cells = width * height
    print(f"\n{'=' * 90}")
    print(f"RENDER BENCHMARK  —  {width}×{height} = {cells:,} cells")
    print(f"{'=' * 90}")
    print(
        f"{'Scenario':<30} {'Med ms':>8} {'P95 ms':>8} {'FPS':>7}"
        f" {'Render':>8} {'ToText':>8} {'Clear':>8}"
    )
    print("-" * 90)

    for r in sorted(results, key=lambda x: x.total_median, reverse=True):
        print(
            f"{r.name:<30} {r.total_median:>8.2f} {r.total_p95:>8.2f}"
            f" {r.fps_at_median:>7.1f}"
            f" {r.render_median:>8.2f} {r.to_text_median:>8.2f}"
            f" {statistics.median(r.clear_ms):>8.2f}"
        )

    # Budget analysis at 18fps
    budget_ms = 1000.0 / 18.0
    print(f"\n18fps budget: {budget_ms:.1f}ms per frame")
    for r in sorted(results, key=lambda x: x.total_median, reverse=True):
        headroom = budget_ms - r.total_median
        status = "OK" if headroom > 0 else "OVER"
        print(f"  {r.name:<30} {status} ({headroom:+.1f}ms headroom)")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pipeline benchmark")
    parser.add_argument("--width", "-W", type=int, default=120)
    parser.add_argument("--height", "-H", type=int, default=40)
    parser.add_argument("--frames", "-n", type=int, default=100)
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated scenario substrings to run (e.g. 'torus,terrain')",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.only:
        filters = [f.strip().lower() for f in args.only.split(",")]
        scenarios = {
            k: v
            for k, v in scenarios.items()
            if any(f in k.lower() for f in filters)
        }

    if not scenarios:
        print("No matching scenarios.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(scenarios)} scenarios, {args.frames} frames each "
          f"at {args.width}×{args.height}...")

    results: list[BenchResult] = []
    for name, factory in scenarios.items():
        scene = factory()
        print(f"  {name}...", end="", flush=True)
        r = bench_scenario(name, scene, args.width, args.height, args.frames)
        print(f" {r.total_median:.2f}ms median")
        results.append(r)

    print_results(results, args.width, args.height)


if __name__ == "__main__":
    main()
