# apeiron

Combinatorial prompt generator for image models. Produces unique, never-repeating prompts by combining components from 12 visual axes — materials, textures, lighting, color palettes, spatial arrangements, rendering techniques, and more.

741 components across 14 templates yield a combinatorial space of ~10.6 trillion unique prompts. Every generation is hash-tracked and deduplicated, so you'll never see the same one twice.

```
$ ap --random
scar-tissue knitting cathedral interior, dust motes, fault gouge drag fold
reclaiming surfaces, analogous cerulean to azure, holographic interferometry

$ ap --random --template essence
riveted chainmail links abacus, vanishing point focus

$ ap --random --template temporal_diptych
brine-leaching becoming antler-casting, fornix in interlocking gears,
isinglass, dusty olive with vermilion pinprick, cross-processed film
```

## How it works

Each prompt is assembled from a **template** (a sentence structure with slots) and **component pools** (curated word lists for each visual category).

Templates define different modes of composition:

| Template | Structure | What it does |
|---|---|---|
| `essence` | `{texture} {form}, {spatial}` | Visual haiku — 3 components, maximum weight per word |
| `material_collision` | `{material} and {material} merging, {phenomenon} at boundary...` | Forces two substances together and renders the seam |
| `temporal_diptych` | `{state} becoming {state}, {form} in {spatial}...` | Captures transformation between two moments |
| `ruin_state` | `{state} {setting}, {atmosphere}, {phenomenon} reclaiming...` | Abandoned places consumed by natural processes |
| `specimen` | `{scale} of {material} {form}, {medium}...` | Clinical imaging — microscopy, cross-sections, scans |
| `liminal` | `{setting}, {light} from beyond, {atmosphere}...` | Threshold spaces — light always coming from somewhere else |

The 12 component categories are orthogonal visual axes:

- **subject_form** — archetypal shapes (torus, foliation, paraboloid, scallop)
- **material_substance** — what it's made of (abalone shell, isinglass, shagreen leather, perlite)
- **texture_density** — surface quality (terry cloth loops, stucco spatter, riveted chainmail links)
- **light_behavior** — how light interacts (cross-polarized glare, butterfly lighting, noctilucent cloud glow)
- **color_logic** — palette relationships, not single colors (molten gold on obsidian, absinthe green and bruised purple)
- **atmosphere_field** — environmental media (sulfurous fumes, bio-luminescent fog, dust motes)
- **phenomenon_pattern** — visual processes (wax bloom exudation, dandelion seed dispersal, voronoi cells)
- **spatial_logic** — compositional arrangement (bilateral symmetry, zigzag ascent, nested concentric)
- **scale_perspective** — viewing position (from a blimp gondola, atomic force microscope, microfluidic channel)
- **temporal_state** — moment in process (antler-casting, brine-leaching, scar-tissue knitting)
- **setting_location** — environmental context (flooded basement, bioluminescent mangrove swamp, cathedral interior)
- **medium_render** — artistic technique (glass plate negative, oil pastel drawing, mri scan, carbon print process)

Components were selected from ~3,700 LLM-generated candidates using OpenCLIP ViT-bigG-14 + T5-v1_1-XXL embeddings with gated filtering, then hand-curated. Each component in the negatable categories (color, light, atmosphere, texture, temporal, medium) has a semantic opposite that gets composed into the negative prompt automatically.

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

### TUI

```bash
ap           # launch the terminal UI
ap --hyper   # launch with 3D hyperobject viewport active
```

Keybindings:

| Key | Action |
|---|---|
| `Space` / `Enter` | Generate next prompt |
| `t` | Cycle template filter |
| `f` | Toggle favorite |
| `a` | Auto-generate (one every 2s) |
| `v` | Toggle hyperobject viewport |
| `h` | Toggle hacker trace log |
| `c` | Copy positive prompt |
| `n` | Copy negative prompt |
| `q` | Quit |

### CLI

```bash
ap --random                          # single prompt to stdout
ap --random --template essence       # single prompt from specific template
ap --batch 50                        # generate 50 unique prompts
ap --batch 10 --format json          # structured output with components
ap --stats                           # generation statistics
ap --export prompts.json             # export all saved prompts
ap --export favs.json --favorites-only
```

### Database

All generated prompts are persisted to `~/.local/share/apeiron/prompts.db` (SQLite). Use `--db PATH` to override.

## Templates

There are 14 templates, each emphasizing different subsets of the 12 visual axes. Some are dense (`material_collision` uses 7 slots), some are sparse (`essence` uses 3). Templates that omit categories like `medium_render` or `color_logic` leave those decisions to the image model — a deliberate choice that produces different aesthetic ranges.

Full template list: `material_study`, `environmental`, `process_state`, `specimen`, `abstract_field`, `liminal`, `textural_macro`, `ruin_state`, `minimal_object`, `material_collision`, `atmospheric_depth`, `temporal_diptych`, `essence`, `site_decay`.

## Hyperobject mode

The `--hyper` flag activates a real-time 3D ASCII viewport that renders prompt-reactive geometry in the TUI. Every generated prompt drives the scene — each template maps to a different geometry type, and components configure lighting, camera, shaders, particle systems, and post-processing effects.

Each template produces a distinct visual form:

| Template | Geometry | Rendering |
|---|---|---|
| `material_study` | Subdivided icosahedron | Gouraud-shaded filled faces |
| `minimal_object` | Torus (donut.c-style) | Direct parametric surface sampling |
| `essence` | Mobius strip | Direct parametric surface sampling |
| `specimen` | Wireframe organism | Edge-only rendering with vertex dots |
| `liminal` | Corridor | Wireframe with perspective recession |
| `atmospheric_depth` | Particle nebula | Point cloud with per-point brightness |
| `abstract_field` | Lorenz attractor trail | Growing point cloud with persistence |
| `site_decay` | Voxel grid | Block erosion/rebuild animation |
| `ruin_state` | Fragmenting solid | Fragment groups with drift |
| `temporal_diptych` | Morphing mesh pair | Blend between two forms over time |
| `material_collision` | Intersecting solids | Dual-mesh overlay |
| `environmental` | Procedural terrain | Heightmap mesh |
| `textural_macro` | Noise surface | Heightmap mesh |
| `process_state` | Metaballs | Filled mesh |

When the template changes between prompts, the viewport plays a transition sequence: the current geometry dissolves, a rotating 4D tesseract (hypercube) wireframe appears, then the new geometry materializes from it. The tesseract is projected from 4D to 3D with continuous rotation across the XW and YZ hyperplanes.

Component words map to scene parameters through the interpreter:

- **light_behavior** selects lighting presets (directional, dramatic side, rim, soft wrap)
- **spatial_logic** positions the camera (symmetrical views, orbital angles, extreme perspectives)
- **scale_perspective** adjusts zoom
- **temporal_state** controls animation speed
- **material_substance** selects the shader character ramp (luminance gradients like ` .,-~:;=!*#$@`)
- **medium_render** activates post-processing effects (glitch, scanlines, blur)
- **atmosphere_field** spawns particle systems

Visual state persists across prompts — only the categories present in the new prompt update; the rest carry forward. Over many generations the viewport accumulates a unique visual identity shaped by its full history.

### CLI snapshots

You can render a single-frame hyperobject snapshot from the command line:

```
$ ap --random --snapshot
╭──────────────────── // minimal_object ────────────────────╮
│                                                           │
│                         ;::~~::=                          │
│                     ,,,,,,,,,,,,,,,-*                      │
│                   ,,,,,--,,,,,,,,,,,,,:                    │
│                  ,,,,,-~:;-,,,,,,,,,,,,~                   │
│                  ,,,,,,~:!#,,,,,,,,,,,,,-                  │
│                  ,,,,,,,-:=!@·  ,=;~~-,,,-                 │
│                  ,,,,,,,,,,-~:;;;;~--,,,,,                 │
│                   ,,,,,,,,,,,,,,,,,,,,,,,,                 │
│                    ,,,,,,,,,,,,,,,,,,,,,                   │
│                      ,,,,,,,,,,,,,,,,                     │
│                                                           │
╰──────────────────── 0xa380aeb746fb373f ───────────────────╯
╭────────────────────── minimal_object ─────────────────────╮
│ solitary silk arch, radial symmetry, chromatic lens        │
│ aberration, cool arctic blue warm amber, fresco buon      │
╰───────────────────────────────────────────────────────────╯
```

Use `--tesseract` to render the 4D hypercube wireframe instead of the template geometry:

```
$ ap --random --snapshot --tesseract
```

Use `--snapshot-size WxH` to control viewport dimensions (defaults to terminal size).

See [HYPEROBJECT_SPEC.md](HYPEROBJECT_SPEC.md) for the full rendering architecture.
