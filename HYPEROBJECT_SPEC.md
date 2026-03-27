# HYPEROBJECT MODE — Design Specification

> *"A hyperobject is something massively distributed in time and space
> relative to humans." — Timothy Morton*

PromptWeaver already generates prompts from a combinatorial space so vast it
will never be exhausted. Hyperobject Mode makes that space *visible* — a
real-time 3D ASCII renderer that transforms every generation into a living
geometric entity, shaped by template semantics, component properties, and
embedding-space relationships.

The existing TUI stays untouched. This is a parallel visual mode: press a key,
and the matrix rain dissolves into a rotating, warping, breathing
four-dimensional object that *is* the prompt.

---

## 1. Core Concept

A dedicated viewport widget renders 3D geometry in ASCII/Unicode at 15–20 fps
using pure-Python math (no OpenGL, no external renderers). The geometry is not
decorative — it is a *projection of the prompt's identity*:

- **Template** determines the *class* of geometric form (tesseract, attractor,
  terrain, corridor, etc.)
- **Components** determine *how* that form looks — its surface, lighting,
  animation, camera, post-processing
- **Embeddings** (when available) determine *dynamics* — energy, tension,
  gravity, void proximity

Every prompt produces a unique visual signature. Two prompts from the same
template with different components will look recognizably related but distinct.
Two prompts from different templates will be entirely different geometric species.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HyperobjectViewport (widget)                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SceneGraph                                               │  │
│  │  ├── geometry: Mesh (vertices, edges, faces)              │  │
│  │  ├── camera: Camera (position, target, fov)               │  │
│  │  ├── lights: list[Light]                                  │  │
│  │  ├── particles: ParticleSystem                            │  │
│  │  └── post_fx: list[PostEffect]                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  RenderPipeline                                           │  │
│  │  1. animate()      — update transforms per frame          │  │
│  │  2. transform()    — model → world → view → clip          │  │
│  │  3. project()      — perspective divide → screen coords   │  │
│  │  4. rasterize()    — scan-convert to char grid            │  │
│  │  5. shade()        — depth/normal → ASCII char selection   │  │
│  │  6. colorize()     — palette mapping → Rich styles        │  │
│  │  7. particles()    — render ambient particle layer        │  │
│  │  8. composite()    — merge layers                         │  │
│  │  9. post_process() — scanlines, bloom, aberration, etc.   │  │
│  │  10. emit()        — write Rich Text → Textual widget     │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  PromptInterpreter                                        │  │
│  │  Maps GeneratedPrompt → SceneGraph parameters             │  │
│  │  ├── template_id → GeometryFactory                        │  │
│  │  ├── components  → visual parameter vector                │  │
│  │  └── embeddings  → dynamics (energy, tension, gravity)    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Module layout

```
promptweaver/
├── hyperobject/
│   ├── __init__.py
│   ├── viewport.py        # HyperobjectViewport (Textual widget)
│   ├── scene.py           # SceneGraph, Camera, Light
│   ├── geometry.py        # Mesh primitives, vertex/edge/face containers
│   ├── primitives.py      # Geometry factories for all 14 templates
│   ├── transform.py       # Matrix math: rotation, projection, perspective
│   ├── rasterizer.py      # 3D → character grid conversion
│   ├── shaders.py         # ASCII surface shaders (depth → char)
│   ├── particles.py       # Ambient particle systems
│   ├── postfx.py          # Post-processing effect stack
│   ├── interpreter.py     # Prompt → scene parameter mapping
│   ├── state.py           # VisualState accumulator (component persistence)
│   ├── transitions.py     # Morph/dissolve between geometries
│   ├── dynamics.py        # Embedding-driven animation modifiers
│   ├── embedding_cache.py # Numpy-only .npz loader + precomputation
│   └── lut.py             # Pre-computed sin/cos/projection lookup tables
```

---

## 3. The Geometry Catalog

Each template maps to a signature geometric form. These are chosen for visual
distinctiveness and thematic resonance with the template's conceptual space.

### 3.1 `material_study` → **Rotating Icosahedron**

A faceted gem. 12 vertices, 30 edges, 20 triangular faces. Rotates slowly,
each face catches the light at a different angle — face-normal shading produces
a jewel-like quality where individual facets flip between bright and dark.

*Why:* Material study is about surfaces and substance. The icosahedron has
enough faces to show material variation but few enough that each face is legible
as a discrete surface.

```
       ╱╲ ╱╲
      ╱  ╳  ╲
     ╱  ╱ ╲  ╲
    ╱──╱   ╲──╲
    ╲  ╲   ╱  ╱
     ╲  ╲ ╱  ╱
      ╲  ╳  ╱
       ╲╱ ╲╱
```

**Animation:** Dual-axis rotation (Y + tilted X). Rotation speed derived from
`temporal_state` component.

### 3.2 `textural_macro` → **Undulating Noise Surface**

A 2D heightmap mesh viewed from a shallow angle, animated with scrolling Perlin
noise. Looks like extreme macro photography of a material surface — ridges,
valleys, and plateaus rendered in depth-shaded ASCII.

```
  ░░▒▒▓▓██▓▓▒▒░░    ░░▒▒▓▓▓▓
  ▒▒▓▓████████▓▓▒▒░░▒▒▓▓████
  ▓▓████████████████▓▓████████
  ████████▓▓████████████████▓▓
  ████▓▓▒▒░░▒▒▓▓████████▓▓▒▒
  ▓▓▒▒░░      ░░▒▒▓▓████▒▒░░
```

**Implementation:** Grid of NxM vertices, Y-displaced by `noise(x + t, z + t)`.
Render as shaded quads. The noise octaves, frequency, and amplitude are all
driven by components.

**Animation:** Noise offset scrolls diagonally. Occasional "ripple" events
(radial displacement wave from a random point).

### 3.3 `environmental` → **Fractal Terrain Flyover**

A larger terrain mesh (like textural_macro but with perspective depth) viewed
from above at an angle, with a slowly advancing camera creating the sensation
of flight over an alien landscape. Horizon line visible.

```
                        ·  · ·
                   · · · ·░░· · · ·
              · · ·░░▒▒▓▓██▓▓▒▒░░· · ·
         · · ░░▒▒▓▓████████████▓▓▒▒░░· ·
    ░░▒▒▓▓████████████████████████████▓▓▒▒
  ▓▓██████████████████████████████████████
```

**Implementation:** Large heightmap grid with multi-octave noise. Perspective
projection with a forward-offset camera. Far vertices fade to dim/dots.

**Animation:** Camera advances along Z-axis (wrapping), creating infinite
flyover. Terrain regenerates at the far edge. Speed from `temporal_state`.

### 3.4 `atmospheric_depth` → **Particle Nebula**

No solid geometry. Instead, a cloud of 200–500 particles forming an organic,
breathing volumetric shape — like a nebula or a smoke simulation frozen in time.
Particles have depth-based brightness and size (nearer = brighter, larger char).

```
              ·
          ·  ∙ •    ·
        ·  • ● ∙  ·
      ∙  ● ◉ ● •  ∙   ·
        • ● ◉ ● ∙   ·
      ·  ∙ • ● •  ·
          ·  ∙  ·
              ·
```

**Implementation:** Particle positions sampled from a 3D Gaussian distribution,
then perturbed by noise. Sorted back-to-front for correct occlusion. Depth →
char size (`·`, `∙`, `•`, `●`, `◉`).

**Animation:** Slow breathing (scale oscillation). Individual particles drift on
noise-driven velocity fields. Palette breathing synced to generation rhythm.

### 3.5 `process_state` → **Metaball Fusion**

Two or three organic blobs, partially merged — frozen mid-transformation. The
surface is an isosurface of the combined scalar field, rendered via raymarching
through a character grid.

```
      ▓▓████▓▓
    ▓▓████████▓▓
    ██████████████▓▓
    ████████████████▓▓
    ██████████████████▓▓
      ▓▓████████████████
        ▓▓██████████████
          ▓▓▓▓████████▓▓
              ▓▓▓▓▓▓▓▓
```

**Implementation:** 2–3 point charges, each with radius `r`. Field:
`f(p) = Σ(rᵢ² / |p - cᵢ|²)`. Surface at `f = threshold`. Render by sampling a
2D slice (or raymarching from camera). Character chosen by field gradient.

**Animation:** Blob centers orbit each other on elliptical paths, slowly.
Periodically one blob splits or two merge. Speed from `temporal_state`.

### 3.6 `material_collision` → **Intersecting Platonic Solids**

Two different Platonic solids (e.g. cube + octahedron) rotating and phasing
through each other. Where they intersect, the surface rendering changes —
denser characters, brighter color, visual interference patterns.

```
         ╱──────╲
        ╱  ╱▲╲   ╲
       │  ╱▲▲▲╲   │
       │ ╱▲▲▲▲▲╲  │
       │╱▲▲▲▲▲▲▲╲ │
        ╲▲▲▲▲▲▲╱  │
         ╲▲▲▲▲╱  ╱
          ╲▲▲╱  ╱
           ╲╱──╱
```

**Implementation:** Two separate meshes with independent rotation matrices. For
each screen cell, determine which mesh(es) have a visible face — if both, apply
an interference shader (XOR pattern, moire, or high-contrast alternation).

**Animation:** Each solid rotates at a different speed on different axes,
creating a constantly-shifting intersection pattern. Ratio of speeds irrational
(e.g. 1.0 and φ) so the pattern never exactly repeats.

### 3.7 `specimen` → **Wireframe Organism in Measurement Grid**

A complex organic wireframe (based on a subdivided icosphere or a
parameterized shell curve) displayed inside a scientific measurement overlay:
crosshairs, grid lines, a reticle border, and readout text.

```
  ┌──────────┬──────────┐
  │    ╱╲  ╱╲│          │
  │   ╱ ╳╱╳ ╲│· · · · · │
  │──╱╱╳╳╳╳╲╲──────────│
  │  ╲╲╳╳╳╳╱╱   + +    │
  │   ╲ ╳╲╳ ╱│          │
  │    ╲╱  ╲╱│          │
  └──────────┴──────────┘
  ├─ 0x3f ─── specimen ─┤
```

**Implementation:** Edges-only rendering (no face fill) of a subdivided
icosphere. Overlay: a grid of `·` characters, crosshair lines through center,
a thin box-drawing border. The overlay is composited behind the wireframe.

**Animation:** Slow rotation. Occasional "scan line" — a horizontal bar of
brighter rendering that sweeps top-to-bottom, as if the specimen is being
imaged.

### 3.8 `minimal_object` → **Lone Torus**

A single torus, perfectly centered, rotating in empty black space. The
quintessence of form. Large negative space. The torus surface is shaded with
maximum depth gradation.

```
             ·····
          ··░░▒▒▒░░··
        ·░▒▓██████▓▒░·
       ░▒▓██▓▒░░▒▓██▓▒░
      ░▒▓█▓▒░    ░▒▓█▓▒░
      ░▒▓█▓░      ░▓█▓▒░
      ░▒▓█▓▒░    ░▒▓█▓▒░
       ░▒▓██▓▒░░▒▓██▓▒░
        ·░▒▓██████▓▒░·
          ··░░▒▒▒░░··
             ·····
```

**Implementation:** Parametric torus: `(R + r·cos(v))·cos(u)`,
`(R + r·cos(v))·sin(u)`, `r·sin(v)`. R/r ratio from `scale_perspective`.
Rendered with full face shading (not wireframe). Depth → char density.

**Animation:** Slow dual-axis rotation. Very slow. Meditative. Breathing
amplitude on the minor radius `r` (subtle pulsation).

### 3.9 `abstract_field` → **Strange Attractor**

A Lorenz or Rössler attractor — thousands of points tracing chaotic orbits,
rendered as a luminous point cloud. Pure mathematics visualized. No solid
surface, no edges — just a trail of light through phase space.

```
          ·  ·∙·
       · ∙∙•∙·∙∙·
      ∙∙••∙  ···∙∙•
     ∙•∙·      ·∙•∙∙
    ∙•·           ·•∙·
     ∙∙·  ·∙∙∙··∙∙∙
      ·∙∙•∙∙∙∙•∙·
        · ·∙•∙· ·
```

**System:** `dx = σ(y−x)`, `dy = x(ρ−z) − y`, `dz = xy − βz` where σ=10,
ρ=28, β=8/3 (Lorenz). Alternative: Rössler, Thomas, Aizawa — selected by
hashing template components.

**Implementation:** Pre-integrate ~5000 steps. Store as point trail. Render
back-to-front with depth-based brightness. Trail head brightest, tail fades.
Rotate the entire point cloud slowly.

**Animation:** The integration continues each frame (5–10 new steps), trail
head advances, oldest points drop off. The attractor orbits eternally.
Component `phenomenon_pattern` selects which attractor system.

### 3.10 `temporal_diptych` → **Split Morph**

The viewport splits into two halves. Each half contains a different geometric
primitive. In the center gap between them, vertices from each form interpolate
toward the other — a continuous metamorphosis frozen at the midpoint.

```
  ╭────────╮  ╭────────╮
  │ ▓▓████ │  │ ╱╲  ╱╲ │
  │ ████▓▓ │→→│╱ ╳╲╱╳ ╲│
  │ ▓▓████ │  │╲ ╳╱╲╳ ╱│
  │ ████▓▓ │  │ ╲╱  ╲╱ │
  ╰────────╯  ╰────────╯
       ↕ interpolation ↕
```

**Implementation:** Two meshes (sphere → cube, torus → icosahedron, etc.),
vertex-matched via closest-point pairing. Each vertex position is
`lerp(posA, posB, t)` where `t` varies by screen-X position (0 at left edge,
1 at right edge). The transition zone in the center shows the metamorphosis.

**Animation:** The `t` mapping oscillates slowly (a wave that slides left and
right), so the morph zone drifts. Each side also rotates independently.

### 3.11 `liminal` → **Infinite Corridor**

Perspective-projected tunnel or corridor, built from repeating rectangular
frames receding toward a vanishing point. The corridor bends slightly, shifts,
and the camera drifts forward through it endlessly.

```
  ╔══════════════════════════╗
  ║ ╔══════════════════════╗ ║
  ║ ║ ╔══════════════════╗ ║ ║
  ║ ║ ║ ╔══════════════╗ ║ ║ ║
  ║ ║ ║ ║ ╔══════════╗ ║ ║ ║ ║
  ║ ║ ║ ║ ║   ·····  ║ ║ ║ ║ ║
  ║ ║ ║ ║ ╚══════════╝ ║ ║ ║ ║
  ║ ║ ║ ╚══════════════╝ ║ ║ ║
  ║ ║ ╚══════════════════╝ ║ ║
  ║ ╚══════════════════════╝ ║
  ╚══════════════════════════╝
```

**Implementation:** N rectangular frames at increasing Z-depth, perspective-
projected. Frame edges are box-drawing characters. Deeper frames → dimmer
palette color. Floor/ceiling lines connect corresponding corners.

**Animation:** Camera advances along Z. As the nearest frame passes behind
camera, a new frame spawns at the far end. Optional lateral drift (camera
sways left-right on a sine wave) and rotation (slight roll oscillation).
`spatial_logic` component determines corridor geometry (straight, curved,
branching).

### 3.12 `ruin_state` → **Fragmenting Solid**

A solid form (cube or other Platonic) in the process of shattering. Faces
separate from the core, drifting outward, rotating independently. Cracks are
visible as gaps between fragments. Entropy made visible.

```
     ╱──╲
    ╱ ██ ╲        ╱╲
   │ ████ │      ╱▓▓╲
   │ ████ │
    ╲ ██ ╱   ╱──╲
     ╲──╱   │▒▒▒▒│
             ╲──╱      ╱╲
                      ╱░░╲
```

**Implementation:** Start with a solid mesh. At initialization, partition faces
into N fragments (Voronoi-like groups). Each fragment has an `offset` vector
and independent rotation. Fragments slowly drift outward from the centroid.

**Animation:** Drift velocity increases over time (accelerating entropy). After
reaching a threshold distance, fragments snap back and the cycle restarts —
ruin and restoration looping. `temporal_state` controls the cycle speed.

### 3.13 `essence` → **Möbius Strip**

A single Möbius strip — one surface, one boundary, mathematically irreducible.
The geometric haiku. Rendered with smooth face shading, slowly rotating, the
twist visible as a continuous gradient that wraps back on itself.

```
          ·░▒▓██▓▒░·
       ·░▒▓██▓▒░  ░▒▓█
     ·░▒▓█▓░·       ░▒▓█
    ░▒▓█▓░·            ▒▓█
   ░▓█▓░·             ·░▓█
    ▓█▒░·            ·░▓█▓
     █▓▒░·       ·░▒▓█▓░·
       █▓▒░·  ·░▒▓██▓░·
          █▓▒▒▓██▓▒░·
```

*Why:* `essence` is the most minimal template — only 3 slots. The Möbius strip
is the most minimal interesting surface — one side, one edge, yet
topologically non-trivial. Three components, one surface. Each component's
weight is visually maximal because there's so little else.

**Implementation:** Parametric Möbius:
`x = (1 + (v/2)·cos(u/2))·cos(u)`,
`y = (1 + (v/2)·cos(u/2))·sin(u)`,
`z = (v/2)·sin(u/2)`, where u ∈ [0, 2π], v ∈ [-0.4, 0.4].
Rendered as a mesh strip with face shading. The twist creates a natural
brightness gradient that cycles through the shader ramp.

**Animation:** Slow single-axis rotation. The strip breathes — `v` range
oscillates subtly (width pulsation). Because `essence` only changes 3 visual
parameters, the other 9 carry through from prior prompts, making each
`essence` generation feel like a *subtle adjustment* to a rich existing scene.

### 3.14 `site_decay` → **Eroding Voxel Grid**

A 3D grid of character blocks — an architectural skeleton dissolving. No
central subject (matching the template's omission of `subject_form` and
`material_substance`). The site *is* the geometry. Blocks fade, fall, and
vanish according to `phenomenon_pattern`, at `temporal_state` speed.

```
   ██  ██  ██     ██
   ██  ██  ██  ██ ██
   ██  ██     ██  ██
      ██  ██  ██
   ██     ██     ██
         ██  ░░
   ░░        ░░  ░░
         ░░
```

*Why:* `site_decay` is about *place and process* with no artistic rendering
style. A voxel grid is inherently architectural (place) and erosion is
inherently process-driven. The absence of a central subject mirrors the
template's lack of `subject_form`. The aesthetic emerges from the void left
behind — what's missing *is* the subject.

**Implementation:** 3D grid (e.g. 8×6×8) of boolean occupancy. Initially full
or pattern-filled. Each frame, cells are removed according to a decay function
(cellular automaton rules, random with gravity, wave-front, etc. — selected by
`phenomenon_pattern`). Remaining cells rendered as filled blocks with depth
shading. Camera orbits slowly.

**Animation:** Decay cycle: full grid → erosion over ~10 seconds →
near-empty → pause → rebuild (cells return in reverse order) → repeat.
`temporal_state` controls cycle speed. `texture_density` controls initial
grid density (sparse scaffold vs. dense monolith). When the grid is mostly
gone, the remaining blocks feel like ruins — isolated fragments of a place
that used to be coherent.

---

## 4. Component → Visual Parameter Mapping

Each of the 12 component categories maps to a *specific rendering parameter*.
This means every generated prompt produces a unique visual configuration.

The mapping uses a deterministic hash of the component word(s) to select from
preset values within each parameter space, ensuring consistency — the same
component always produces the same visual effect.

### 4.1 Mapping Table

| Category | Rendering Parameter | Effect Space |
|---|---|---|
| `subject_form` | **Mesh complexity** | Vertex count / subdivision level (3 tiers: sparse, standard, dense) |
| `material_substance` | **Surface shader** | Which ASCII character set renders the surface (6 presets) |
| `texture_density` | **Detail frequency** | Noise octaves, mesh granularity, particle count (continuous) |
| `light_behavior` | **Lighting rig** | Light direction, count, type, and intensity (8 presets) |
| `color_logic` | **Palette application** | How palette colors map to depth/normals (6 strategies) |
| `atmosphere_field` | **Ambient particles** | Particle type, density, behavior around the geometry |
| `phenomenon_pattern` | **Animation easing** | The mathematical function driving motion (sin, noise, fractal, etc.) |
| `spatial_logic` | **Camera & composition** | Camera angle, FOV, and any symmetry mirroring |
| `scale_perspective` | **Zoom / distance** | How close the camera is to the object |
| `temporal_state` | **Animation speed** | Rotation speed, morph rate, particle velocity |
| `setting_location` | **Environment** | Background treatment (void, grid floor, particle field, etc.) |
| `medium_render` | **Post-processing** | Which effects are active and at what intensity |

### 4.2 Surface Shaders (from `material_substance`)

The surface shader determines which characters represent the geometry's surface
at different depth/brightness levels. Mapped by hashing the chosen
`material_substance` word:

```python
SHADERS: dict[str, str] = {
    # name:         dark → light (10 levels)
    "block":       " ░░▒▒▓▓███",
    "ascii":       " .·:-=+*#%@",
    "circuit":     " ·─│┌┐└┘┼═║",
    "organic":     " .·°oO@8&%#",
    "minimal":     "  · ·∙∙••●●",
    "glass":       "  · ·.:░▒▓█",
    "bone":        " .·:;+=≡≣#█",
    "ferrofluid":  " ~∼≈≋∽∿⌇⌇█",
    "silk":        "  · ··.·.░▒",
    "ceramic":     " ·.·:○◌◍◉●█",
}
```

Words that don't have a direct match fall through to a consistent hash-based
selection from the list.

### 4.3 Lighting Rigs (from `light_behavior`)

```python
LIGHT_RIGS: dict[str, LightConfig] = {
    "volumetric_god_rays": LightConfig(
        direction=(0.3, -1.0, 0.5), intensity=1.4, style="directional",
        god_rays=True,  # scanline brightening from light source
    ),
    "caustics": LightConfig(
        direction=(0.0, -0.8, 0.6), intensity=1.0, style="directional",
        caustic_noise=True,  # rippling brightness overlay
    ),
    "subsurface_scattering": LightConfig(
        direction=(0.5, -0.5, 0.7), intensity=0.8, style="wrap",
        wrap_factor=0.5,  # light wraps around edges
    ),
    "rim_light": LightConfig(
        direction=(-0.5, 0.0, -1.0), intensity=1.2, style="rim",
        # bright edges, dark center
    ),
    "bokeh": LightConfig(
        direction=(0.0, -1.0, 0.3), intensity=0.6, style="point",
        point_count=5, falloff="quadratic",
    ),
    # ... etc, one per light_behavior word, with fallback hashing
}
```

### 4.4 Color Strategies (from `color_logic`)

How palette colors are distributed across the geometry:

| Strategy | Description |
|---|---|
| **depth_gradient** | Near = bright, far = dim (default) |
| **normal_bands** | Color bands based on face normal direction |
| **binary** | Two-tone: faces above/below median depth |
| **triadic_zones** | Three palette colors in angular zones (120° each) |
| **iridescent** | Color shifts based on view angle (simulated thin-film) |
| **monochrome** | Single palette.primary, only brightness varies |

### 4.5 Animation Easings (from `phenomenon_pattern`)

The function that drives the primary animation parameter (rotation angle,
morph factor, particle drift, etc.):

| Easing | Function | Visual Character |
|---|---|---|
| `crystallization` | Step function with smooth transitions | Snaps between discrete states |
| `erosion` | Logarithmic decay | Starts fast, slows asymptotically |
| `growth` | Exponential ease-in | Starts slow, accelerates |
| `glitch` | Random discontinuous jumps | Teleports unpredictably |
| `fractals` | Recursive sine: `sin(t) + sin(2t)/2 + sin(4t)/4` | Complex harmonic motion |
| `wave` | Simple sine | Smooth oscillation |
| `turbulence` | Perlin noise over time | Organic, unpredictable drift |
| `pulse` | `abs(sin(t))` | Rhythmic heartbeat |

### 4.6 Camera Presets (from `spatial_logic`)

| Preset | Camera Behavior |
|---|---|
| `symmetrical` | Front-on, centered, no drift |
| `radial` | Orbits the object in a circle |
| `diagonal` | Fixed 45° angle, slow zoom breathe |
| `recursive` | Fractal zoom: camera oscillates between close and far |
| `flowing` | Smooth Lissajous path around object |
| `isometric` | Fixed isometric angle (30° tilt) |

### 4.7 Environments (from `setting_location`)

What exists in the space around the geometry:

| Environment | Description |
|---|---|
| `void` | Pure black. Nothing. Maximum contrast. |
| `grid_floor` | Perspective grid plane beneath the object (Tron-style) |
| `particle_dust` | Sparse floating particles (like dust motes in a beam) |
| `constellation` | Distant dim points, like stars |
| `fog_gradient` | Bottom rows fade into dim characters |
| `scan_field` | Horizontal scan lines sweeping through the space |

### 4.8 Post-Processing (from `medium_render`)

| Effect | Description |
|---|---|
| `oil_impasto` | Thicken bright characters (neighbor expansion) |
| `charcoal` | High contrast, noise grain overlay, edge emphasis |
| `risograph` | Halftone dithering pattern |
| `daguerreotype` | Vignette (dim edges), slight noise |
| `3d_render` | Clean, no post-processing (pass-through) |
| `glitch_art` | Random scanline displacement, color channel offset |
| `crt` | Barrel distortion, scanlines, slight phosphor glow |
| `blueprint` | Wireframe-only rendering, grid overlay, all one color |

---

## 5. Component Persistence (Visual State Accumulator)

Not every template uses every component category. `essence` uses only 3 of 12;
`site_decay` uses 5; even most full templates omit 6 categories. Without
persistence, the viewport would lose most of its visual parameters every time
the template changes.

### 5.1 The Problem

The 12 component categories each drive a rendering parameter (shader, lighting,
camera, particles, post-fx, etc.). If the incoming prompt doesn't include a
category, what drives that parameter?

### 5.2 The Solution: Carry-Forward State

The viewport maintains a **persistent visual state** — a dict of 12 slots, one
per category. When a new prompt arrives, *only the slots for categories present
in that prompt are updated*. Everything else carries forward from the last
prompt that touched it.

```python
@dataclass
class VisualState:
    """Accumulated visual parameters — one slot per component category.

    Each slot holds the most recent component word(s) for that category.
    Updated incrementally: only categories present in the new prompt change.
    """
    slots: dict[str, list[str]]  # category → most recent word(s)
    # Initialized with sensible defaults on first launch

    def apply_prompt(self, prompt: GeneratedPrompt) -> set[str]:
        """Update slots from prompt. Returns set of categories that changed."""
        changed: set[str] = set()
        for category, words in prompt.components.items():
            if self.slots.get(category) != words:
                self.slots[category] = words
                changed.add(category)
        return changed

# Defaults for first launch (before any prompt has been generated):
INITIAL_STATE: dict[str, list[str]] = {
    "subject_form":       ["sphere"],
    "material_substance": ["glass"],
    "texture_density":    ["smooth"],
    "light_behavior":     ["soft ambient light"],
    "color_logic":        ["monochromatic"],
    "atmosphere_field":   ["dust motes"],
    "phenomenon_pattern": ["crystallization"],
    "spatial_logic":      ["symmetrical"],
    "scale_perspective":  ["eye level"],
    "temporal_state":     ["suspended"],
    "setting_location":   ["void"],
    "medium_render":      ["3d render"],
}
```

### 5.3 What This Creates

The visual state becomes a **palimpsest** — layers of different prompts'
contributions accumulating over time:

| Template | Updates | Carries Forward |
|---|---|---|
| `material_study` (6 slots) | subject, material, texture, light, color, medium | atmosphere, phenomenon, spatial, scale, temporal, setting |
| `essence` (3 slots) | texture, subject, spatial | *9 categories unchanged* — feels like a surgical adjustment |
| `site_decay` (5 slots) | scale, setting, temporal, phenomenon, texture | *7 categories unchanged* — keeps lighting, shader, camera, post-fx |
| `abstract_field` (5 slots) | phenomenon, spatial, color, atmosphere, medium | *7 categories unchanged* — keeps the "object" from prior prompt |

This means:
- **`essence` prompts feel minimal** — they change the form, its texture, and
  its arrangement, but the lighting rig, surface shader, camera distance,
  particle type, and post-fx all persist. Three things shift; nine stay.
- **Template switches feel organic** — not everything resets at once. The new
  template inherits the visual residue of whatever came before.
- **Over many generations**, the viewport accumulates a unique visual identity
  shaped by the full history of prompts, not just the current one.
- **The visual has memory.** It remembers the fog from three prompts ago, the
  lighting from five prompts ago, the camera angle from the first generation.

### 5.4 Transition Behavior

When `apply_prompt()` returns the set of changed categories, the transition
system uses this to decide *what* to animate:

- **Geometry change** (template switched) → full dissolve → tesseract → new form
- **Multiple parameters changed** → shader cross-fade, camera lerp, particle
  blend over ~1 second
- **Few parameters changed** (`essence`) → smooth morph of only the affected
  parameters; geometry barely shifts, just adjusts

This creates a natural rhythm: heavy templates produce dramatic transitions,
minimal templates produce subtle modulations.

### 5.5 Embedding Awareness in Persistence

When embeddings are available, the carry-forward state also affects dynamic
calculations. The "prompt energy" metric uses the *full visual state* (all 12
slots), not just the components from the current prompt. This means `essence`
prompts still produce rich embedding dynamics — they're computed from the
accumulated state, which includes contributions from many prior prompts.

---

## 6. Embedding-Driven Dynamics

Embedding integration requires only **numpy** — no torch, no transformers.
Pre-computed embeddings are permanently cached in `.npz` files that ship with
the project. All dynamic computation is pure linear algebra on L2-normalized
vectors (cosine similarity = dot product).

### 6.1 Embedding Source: Cached `.npz` Files

Two pre-computed embedding caches exist in `promptweaver/data/`:

| File | Models | Dimensions | Words | Size |
|---|---|---|---|---|
| `curated_embeddings.npz` | CLIP (ViT-B/32) + T5-large | 512 + 1024 | 748 | 4.5 MB |
| `embedding_cache_openclip_t5xxl.npz` | OpenCLIP (ViT-bigG-14) + T5-XXL | 1280 + 4096 | 1,339 | 27 MB |

All vectors are **L2-normalized**, so cosine similarity = `np.dot(a, b)`.

The smaller `curated_embeddings.npz` is the default — it covers the active
component pool, loads in ~50ms, and uses ~3MB of RAM. The larger file can be
used for richer dynamics if available.

### 6.2 Startup Precomputation

On first launch (or when the `.npz` is detected), precompute and cache:

```python
class EmbeddingCache:
    """Numpy-only embedding lookup. No torch required."""

    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(str(data["_metadata_json"]))

        # Build word → vector lookup (CLIP space, 512d)
        self.clip: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.endswith("/clip"):
                word = key.rsplit("/", 2)[-2]  # {cat}/{word}/clip
                self.clip[word] = data[key]

        # Precompute category centroids
        self.centroids: dict[str, np.ndarray] = {}
        for cat, words in self._group_by_category(data).items():
            vecs = np.stack([self.clip[w] for w in words])
            centroid = vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            self.centroids[cat] = centroid

    def prompt_vectors(self, components: dict[str, list[str]]) -> np.ndarray:
        """Look up CLIP vectors for all components in a prompt.
        Returns (N, 512) array. ~microseconds."""
        vecs = []
        for words in components.values():
            for w in words:
                if w in self.clip:
                    vecs.append(self.clip[w])
        return np.stack(vecs) if vecs else np.zeros((1, 512))
```

**Startup cost:** ~50ms to load `.npz` + ~2ms to precompute 12 centroids.
**Per-prompt cost:** Dictionary lookups + a single `np.dot()` — **< 0.1ms**.

### 6.3 Prompt Energy

**Input:** All component words from the full visual state (§5, including
carry-forward slots).

**Computation:**
```python
vecs = cache.prompt_vectors(visual_state.slots)  # (N, 512)
sim_matrix = vecs @ vecs.T                        # (N, N) pairwise cosine sim
# Mean off-diagonal distance
mask = ~np.eye(len(vecs), dtype=bool)
energy = 1.0 - sim_matrix[mask].mean()            # 0 = identical, 1 = orthogonal
```

**Cost:** For ~12 components at 512d: a 12×512 @ 512×12 matmul = **< 0.05ms**.

**Effect:** Scales global animation speed and intensity.

| Energy Level | Visual |
|---|---|
| Low (< 0.3) | Slow rotation, gentle breathing, minimal particles |
| Medium (0.3–0.6) | Normal speed, moderate particle activity |
| High (0.6–0.8) | Fast rotation, energetic particles, sharper lighting |
| Extreme (> 0.8) | Chaotic motion, geometry distortion, glitch artifacts |

### 6.4 Semantic Gravity (Vertex Displacement)

**Input:** Per-component CLIP vectors.

**Computation:** Project each component's 512d vector to a 3D direction. This
can be done simply — use the first 3 principal components of the component
vectors (a 3-component PCA), or even just take coordinates [0:3] of each
normalized vector. Each component becomes a "pole" in 3D space.

**Per-frame:** For each vertex, compute weighted attraction toward each pole:
```python
displacement = sum(
    pole_direction * (1.0 / (1.0 + distance_to_pole))
    for pole_direction in poles
) * gravity_strength
```

**Cost:** With 12 poles and ~100 vertices, this is ~1200 multiply-adds per
frame. Negligible.

**Effect:** The geometry subtly deforms — bulging toward semantically dense
regions and thinning toward sparse ones. Components with similar embeddings
create convergent pull; dissimilar ones create tension.

### 6.5 Cross-Category Tension

**Input:** Cosine similarity between each pair of chosen components across
different categories.

**Computation:**
```python
vecs = cache.prompt_vectors(visual_state.slots)
sim_matrix = vecs @ vecs.T
# Find cross-category pairs with unusually high similarity
tension_pairs = [(i, j, sim_matrix[i, j])
                 for i, j in cross_category_indices
                 if sim_matrix[i, j] > 0.7]
tension_score = max(s for _, _, s in tension_pairs) if tension_pairs else 0.0
```

**Effect:** Visual interference patterns — moiré waves, flickering brightness,
edge rippling. The geometry vibrates along the axis between semantically
redundant components. Makes redundancy *visible*.

### 6.6 Void Proximity

**Input:** Mean embedding vector of all components in the visual state.

**Computation:**
```python
mean_vec = vecs.mean(axis=0)
mean_vec /= np.linalg.norm(mean_vec)
# Distance to each category centroid
centroid_sims = np.array([
    np.dot(mean_vec, c) for c in cache.centroids.values()
])
# Low mean similarity to centroids = in a void region
void_proximity = 1.0 - centroid_sims.mean()
```

**Effect:** Prompts near void regions render with lower opacity — geometry
becomes more wireframe-like, particles become sparse, depth shading becomes
flatter. Prompts in dense embedding space render as solid, vivid, fully
realized forms.

### 6.7 Graceful Degradation

If numpy is not installed, or if `.npz` files are missing:
- **Prompt Energy** defaults to `0.45` (medium)
- **Semantic Gravity** disabled (no vertex displacement)
- **Cross-Category Tension** disabled (no interference)
- **Void Proximity** defaults to `0.5` (half opacity)

All component → parameter mappings still function via deterministic hashing.
The rendering works fully without embeddings — they're a bonus layer.

### 6.8 Performance Summary

| Operation | When | Cost |
|---|---|---|
| Load `.npz` | Startup (once) | ~50ms |
| Precompute centroids | Startup (once) | ~2ms |
| Look up component vectors | Per prompt | ~0.01ms |
| Pairwise similarity matrix | Per prompt | ~0.05ms |
| Semantic gravity poles | Per prompt | ~0.1ms |
| Gravity displacement | Per frame | ~0.01ms |

**Total per-prompt overhead: < 0.2ms.** Completely invisible.

---

## 7. Rendering Pipeline — Technical Detail

### 7.1 Coordinate System

Right-handed. X-right, Y-up, Z-toward-camera (screen normal).

### 7.2 Mesh Representation

```python
@dataclass
class Mesh:
    vertices: list[Vec3]          # model-space positions
    edges: list[tuple[int, int]]  # index pairs
    faces: list[tuple[int, ...]]  # index tuples (triangles or quads)
    normals: list[Vec3]           # per-face normals (computed)
```

### 7.3 Transform Chain

```python
def transform_vertex(v: Vec3, model: Mat4, view: Mat4, proj: Mat4) -> Vec2:
    world = model @ Vec4(v, 1.0)
    eye   = view @ world
    clip  = proj @ eye
    # perspective divide
    ndc_x = clip.x / clip.w
    ndc_y = clip.y / clip.w
    # map to screen characters
    screen_x = int((ndc_x + 1) / 2 * width)
    screen_y = int((1 - ndc_y) / 2 * height)  # Y-flip
    return Vec2(screen_x, screen_y), clip.z / clip.w  # + depth
```

All matrix math is pure Python with pre-computed LUTs for trig functions.
The `Mat4` and `Vec3`/`Vec4` types are minimal dataclasses with `__matmul__`.

### 7.4 ASCII Rasterization

For **filled** geometry (faces):

1. Project all vertices to screen space.
2. For each face, compute screen-space bounding box.
3. For each character cell in the bbox, test if inside the projected triangle
   (barycentric coordinates or edge function).
4. If inside, interpolate depth. If nearer than z-buffer value, write to
   z-buffer and color buffer.
5. Map depth to ASCII character via the active shader.

For **wireframe** geometry (edges only):

1. Project both endpoints.
2. Bresenham's line algorithm in character space.
3. Each character on the line gets the edge character and interpolated depth.

For **point clouds** (attractors, particles):

1. Project each point.
2. Write single character at screen position if nearer than z-buffer.
3. Character selected by depth.

### 7.5 Face-Normal Shading

After projection, compute the dot product of each face's world-space normal
with the light direction. This produces a brightness value in [0, 1] that
indexes into the shader's character ramp.

```python
brightness = max(0, dot(face_normal, light_dir))
char_idx = int(brightness * (len(shader_chars) - 1))
char = shader_chars[char_idx]
```

Wrap lighting (for subsurface scattering):
```python
brightness = (dot(face_normal, light_dir) + wrap) / (1 + wrap)
```

### 7.6 Depth-Based Palette Coloring

The z-buffer value (after projection) maps to palette color bands:

```python
if depth < 0.25:
    style = palette.bright
elif depth < 0.50:
    style = palette.primary
elif depth < 0.75:
    style = palette.rain_mid
else:
    style = palette.rain_dim
```

Alternative strategies (from `color_logic`) override this default.

### 7.7 Aspect Ratio Correction

Terminal characters are taller than wide (~2:1). All Y coordinates are scaled
by 0.5 during projection to produce visually correct proportions.

---

## 8. The Tesseract

The tesseract (4D hypercube) is the *signature visual element* of Hyperobject
Mode. It appears during transitions, idle states, and startup. It represents the
higher-dimensional space from which all prompts are drawn.

### 8.1 Geometry

**16 vertices:** All combinations of (±1, ±1, ±1, ±1)
**32 edges:** Vertex pairs differing in exactly one coordinate
**24 square faces** (8 cubic cells in 4D)

### 8.2 4D Rotation

Two independent rotation planes (XW and YZ) create the characteristic
"inside-out folding" motion that makes the hypercube appear to pass through
itself:

```python
def rotate_4d(v: Vec4, angle_xw: float, angle_yz: float) -> Vec4:
    # Rotation in XW plane
    x1 = v.x * cos(angle_xw) - v.w * sin(angle_xw)
    w1 = v.x * sin(angle_xw) + v.w * cos(angle_xw)
    # Rotation in YZ plane
    y1 = v.y * cos(angle_yz) - v.z * sin(angle_yz)
    z1 = v.y * sin(angle_yz) + v.z * cos(angle_yz)
    return Vec4(x1, y1, z1, w1)
```

### 8.3 4D → 3D Projection

Perspective projection from 4D to 3D (then standard 3D → 2D):

```python
def project_4d_to_3d(v: Vec4, distance: float = 2.5) -> Vec3:
    w_factor = 1.0 / (distance - v.w)
    return Vec3(v.x * w_factor, v.y * w_factor, v.z * w_factor)
```

### 8.4 Rendering

The tesseract is rendered as wireframe with depth-faded edges. Edges closer to
the viewer are brighter (palette.bright), further edges dimmer (palette.rain_dim).
Vertices are rendered as bright dots (`●`).

The inner and outer cubes of the hypercube projection are clearly visible, with
connecting edges between them. As the 4D rotation progresses, the inner and
outer cubes swap — the signature "folding" effect.

### 8.5 When It Appears

| Context | Behavior |
|---|---|
| **Startup** | Tesseract fades in, rotates for 2–3 seconds before first generation |
| **Generation transition** | Current geometry dissolves → tesseract briefly appears → new geometry forms |
| **Idle** (>15s no generation) | Current geometry slowly morphs into the tesseract |
| **Artifact event** (0.2% chance) | Tesseract glitches: vertices scatter, edges fragment, then snap back |

---

## 9. Transitions & Morphing

### 9.1 Generation Transition Sequence

When a new prompt is generated, the viewport plays a 1.5-second transition:

```
Frame 0–10:    Current geometry begins fragmenting
                 → faces separate, drift outward
                 → edges flicker, brightness drops
Frame 10–20:   Fragments collapse to a point
                 → tesseract emerges from the singularity
                 → tesseract rotates through one full 4D fold
Frame 20–30:   Tesseract vertices begin migrating
                 → each vertex lerps toward its target on the new geometry
                 → new surface shader fades in
Frame 30–40:   New geometry solidifies
                 → component-driven parameters activate
                 → full brightness restored
```

Total: ~40 frames at 20fps = 2 seconds. Runs concurrently with the GlitchPrompt
text decode animation (14 frames at 24fps ≈ 0.6s), so the geometry transition
outlasts the text transition, keeping visual interest alive.

### 9.2 Vertex Morphing

To morph between two meshes with different vertex counts:

1. **Match:** Use the mesh with more vertices as the target. For the source,
   duplicate vertices (nearest-neighbor assignment) to match count.
2. **Interpolate:** `lerp(src_vertex, dst_vertex, t)` where `t` eases from 0→1
   using the active `phenomenon_pattern` easing function.
3. **Surface transition:** Cross-fade between the two surface shaders by
   blending character selections.

### 9.3 Dissolve Effect

For the fragmentation phase, each face is assigned to a fragment group.
Each group gets a random outward velocity vector. Rendering continues normally,
but each fragment's vertices are offset by `velocity * t²` (accelerating drift).

---

## 10. Particle Systems

Particles provide atmosphere around the main geometry. They exist in a separate
render layer, composited behind the geometry (or in front, for dust/snow).

### 10.1 Particle Types (from `atmosphere_field`)

| Type | Character | Behavior |
|---|---|---|
| `fog` | `░`, `▒` | Slow upward drift, high density, concentrated at bottom |
| `smoke` | `·`, `∙`, `°` | Turbulent drift (noise velocity), medium density |
| `spores` | `·`, `∘`, `○` | Random walk, very slow, sparse |
| `rain` | `│`, `┃`, `:` | Fast downward, high density, streaks |
| `dust_motes` | `·`, `∙` | Brownian motion, catch light (flash occasionally) |
| `snow` | `·`, `*`, `✦` | Slow diagonal drift, gentle oscillation |
| `embers` | `·`, `∙`, `•` | Fast upward with deceleration, bright head fading tail |
| `data_stream` | `0`, `1`, hex chars | Vertical streams (matrix rain but sparser, 3D-positioned) |

### 10.2 Particle Lifecycle

```
spawn → drift (velocity + noise) → fade (brightness decreases) → die
```

Spawn rate from `texture_density` component. Lifetime from `temporal_state`.
New particles spawn at random positions in a shell around the geometry.

---

## 11. Post-Processing Stack

Applied to the completed character grid before final output. Each effect is a
function `(grid: CharGrid) -> CharGrid`. Multiple effects compose.

### 11.1 Scanlines

Every Nth row is dimmed one step. N from 2 (dense) to 4 (subtle).

```python
for y in range(height):
    if y % scanline_period == 0:
        for x in range(width):
            grid[y][x].dim_one_step()
```

### 11.2 Chromatic Aberration

Three color channels (R, G, B from the palette) are rendered offset by 1
character in different directions. Only applies to bright characters.

### 11.3 Bloom

Bright characters (`█`, `#`, `@`, etc.) propagate their brightness to
adjacent cells. A 1-cell radius convolution with a small kernel.

### 11.4 Dithering (Floyd-Steinberg)

When the depth → character mapping produces smooth gradients, optionally apply
error diffusion to create a stippled, print-like texture.

### 11.5 Vignette

Characters near the viewport edges are dimmed progressively. Creates focus
toward the center.

### 11.6 CRT Barrel Distortion

Characters are displaced radially outward from the center, simulating a CRT
screen's curvature. Implemented by remapping source coordinates before output.

### 11.7 Edge Detection (Sobel)

Compute character-space gradients. Only render characters where the gradient
exceeds a threshold. Produces a silhouette/outline rendering.

---

## 12. TUI Integration

### 12.1 Two Display Modes

The HyperobjectViewport replaces the MatrixRain widget in the layout. Both
widgets occupy the same slot — only one is visible at a time.

| Key | Mode | What's Visible |
|---|---|---|
| `v` | **Hyperobject** | 3D viewport (replacing rain) |
| (default) | **Matrix Rain** | Classic digital rain |
| `h` | **Hacker Log** | System trace (existing) |

The three modes cycle: Rain → Hyperobject → Hacker Log → Rain.
Or: `v` toggles between Rain ↔ Hyperobject. `h` toggles between
current ↔ Hacker Log. Implementation TBD based on feel.

### 12.2 Fullscreen Mode

`V` (shift-v) toggles fullscreen: the viewport expands to fill the entire
main column (hiding GlitchPrompt, negative, components, entropy meter). Only
the viewport, the banner, the sidebar, and footer remain.

In fullscreen, the geometry renders at much higher resolution (more characters =
more detail). This is the "sit back and watch" mode.

### 12.3 Widget API

```python
class HyperobjectViewport(Static):
    """3D ASCII renderer driven by prompt semantics."""

    FPS: float = 1 / 18  # ~18 fps target

    def set_palette(self, palette: Palette) -> None: ...
    def set_prompt(self, prompt: GeneratedPrompt) -> None: ...
    def set_embeddings(self, embeddings: Optional[DualEmbeddings]) -> None: ...
    def set_fullscreen(self, fullscreen: bool) -> None: ...
```

`set_prompt()` triggers the transition sequence (dissolve → tesseract → new
geometry). The widget manages its own timer for animation frames.

### 12.4 App Integration

In `app.py`, minimal changes:

```python
from .hyperobject import HyperobjectViewport

# In compose():
yield HyperobjectViewport(id="hyperobject-viewport")

# In _render():
viewport = self.query_one("#hyperobject-viewport", HyperobjectViewport)
viewport.set_prompt(self.current)

# In _apply_palette():
viewport.set_palette(palette)

# New binding:
Binding("v", "toggle_hyperobject", "HYPER")
```

### 12.5 CLI Flag

```bash
promptweaver --hyper        # Launch directly in Hyperobject Mode
promptweaver --hyper --fs   # Launch in fullscreen Hyperobject Mode
```

---

## 13. Performance Strategy

### 13.1 Target

18 fps at 80×24 terminal. 12+ fps at 160×48. Graceful degradation beyond that.

### 13.2 Optimizations

| Technique | Details |
|---|---|
| **Trig LUTs** | Pre-compute 4096-entry sin/cos tables. All rotation uses table lookup. |
| **Frame skipping** | If a frame takes >55ms, skip the next frame's render (keep animating state). |
| **Adaptive detail** | Viewport width < 60 chars → reduce mesh subdivision by 1 level. |
| **Dirty rectangles** | Only re-render regions that changed (hard for 3D, but particles can be layered). |
| **Edge-only mode** | If FPS drops below 10, auto-switch to wireframe rendering (faster). |
| **Z-buffer array** | Use a flat `list[float]` for the depth buffer, not nested structures. |
| **String building** | Build output as a single `Text()` with bulk appends, not per-character. |
| **No numpy required** | Pure Python math. Optional numpy acceleration if available (matrix multiply). |

### 13.3 Profiling Hooks

A `--hyper-debug` flag overlays:
- Current FPS counter
- Vertex count
- Face count
- Active post-effects
- Embedding energy (if available)

---

## 14. Implementation Phases

### Phase 1: Foundation (Core Renderer)

- `lut.py` — Trig lookup tables, `Vec3`/`Vec4`/`Mat4` types
- `transform.py` — Rotation, projection, perspective matrices
- `geometry.py` — `Mesh` dataclass, vertex/edge/face containers
- `rasterizer.py` — Z-buffered ASCII rasterization (wireframe + filled)
- `shaders.py` — Character depth maps, face-normal shading
- `scene.py` — `SceneGraph`, `Camera`, `Light` types
- `viewport.py` — `HyperobjectViewport` Textual widget (timer-driven rendering)

**Deliverable:** A rotating wireframe cube in the viewport slot. Palette-colored.

### Phase 2: Geometry Catalog

- `primitives.py` — All 14 template geometries:
  - Icosahedron, noise surface, terrain, particle cloud
  - Metaballs, intersecting solids, wireframe organism
  - Torus, strange attractor, split morph
  - Infinite corridor, fragmenting solid
  - Möbius strip, eroding voxel grid
- `interpreter.py` — Template → geometry factory mapping

**Deliverable:** Each template renders its signature form. Geometry changes on
each generation.

### Phase 3: Component Mapping & Persistence

- `state.py` — `VisualState` accumulator with carry-forward logic
- Extend `interpreter.py` with the full component → parameter mapping
- Surface shader selection from `material_substance`
- Lighting rig from `light_behavior`
- Color strategy from `color_logic`
- Camera preset from `spatial_logic`
- Animation easing from `phenomenon_pattern`
- Zoom from `scale_perspective`
- Speed from `temporal_state`
- Environment from `setting_location`
- Mesh complexity from `subject_form`
- Detail frequency from `texture_density`
- Carry-forward: unchanged categories persist from prior prompts
- Transition intensity scales with number of changed categories

**Deliverable:** Every prompt produces a visually unique configuration. Same
template + different components = recognizably different renderings. `essence`
prompts feel like surgical adjustments; `material_study` prompts feel like
broad reconfigurations. The visual has memory.

### Phase 4: Tesseract & Transitions

- 4D hypercube geometry and rotation
- 4D → 3D → 2D projection pipeline
- Transition sequence: dissolve → tesseract → morph → new form
- Vertex matching and interpolation between different meshes
- Idle detection → tesseract drift

**Deliverable:** The generation flow feels cinematic. Each prompt arrives through
a dimensional gateway.

### Phase 5: Particles & Post-FX

- `particles.py` — Particle system with 8 atmosphere types
- `postfx.py` — Scanlines, bloom, chromatic aberration, vignette, CRT, dithering,
  edge detection
- `medium_render` → post-FX stack mapping

**Deliverable:** Full visual richness. The rendering feels like a living,
breathing terminal art installation.

### Phase 6: Embedding Integration

- `dynamics.py` — Prompt energy, semantic gravity, cross-category tension,
  void proximity
- `embedding_cache.py` — Numpy-only loader for `curated_embeddings.npz`
- Startup precomputation: category centroids, word→vector lookup
- Per-prompt: pairwise similarity, energy, tension, void proximity (< 0.2ms)
- Per-frame: gravity displacement on vertices (< 0.01ms)
- Graceful fallback when numpy or `.npz` not available

**Deliverable:** The geometry responds to the *meaning* of the prompt, not just
its surface parameters. No torch, no transformers — just cached vectors and
dot products.

### Phase 7: Polish

- Fullscreen mode (`V`)
- `--hyper` and `--hyper-debug` CLI flags
- Performance profiling and optimization pass
- Adaptive detail based on terminal size
- Frame rate stabilization

---

## 15. Dependencies

**No new required dependencies.** The entire renderer is pure Python math.

**Optional (already in project deps):**
- `numpy` — Used for embedding dynamics (load `.npz`, dot products, PCA).
  Also accelerates batch vertex transforms if available. Falls back to pure
  Python lists/math if numpy is not installed.

**NOT required:**
- `torch` — Not needed. Embeddings are pre-computed and cached in `.npz` files.
- `transformers` — Not needed. No model inference at runtime.

The renderer detects numpy at import time. Without numpy, the 3D rendering
works perfectly (all geometry, shading, particles, post-fx). Only the
embedding-driven dynamics (§6) are disabled — those degrade gracefully to
fixed defaults.

---

## 16. Visual Identity

The aesthetic goal is **"terminal sublime"** — the feeling of staring into
something vast through the tiny window of a terminal emulator. Every element
should feel like it exists in a space larger than what's visible.

- Geometry extends beyond frame edges (vertices that project off-screen are
  clipped, but their edges are drawn right to the border)
- Particles drift in from outside the viewport and exit the other side
- The tesseract's 4D rotation implies a dimensionality that can't be fully
  displayed
- Depth shading creates the illusion that the terminal has physical depth
- Post-processing (scanlines, vignette) frames the viewport as a *window into*
  something, not a rendering *on* a surface

The terminal is not a screen. It is a viewport into the combinatorial hyperspace
from which every prompt is drawn.
