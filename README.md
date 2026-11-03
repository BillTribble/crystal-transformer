# Crystal Transformer

**Ray traced crystal experiment with procedural growth and generative ambient soundtrack.**

An interactive real-time WebGL2 crystal ray-tracing studio (**Crystal Transformer**) built on **[Jack Bush](https://github.com/jackbush)**'s [`geode` sketch in `jackbush/glimmers`](https://github.com/jackbush/glimmers/tree/main/src/sketches/geode).

## Credits & Attribution

- **Original Geode Ray-Tracer & Convex Polyhedron Shader Architecture**: **[Jack Bush (@jackbush)](https://github.com/jackbush)** — [`https://github.com/jackbush/glimmers`](https://github.com/jackbush/glimmers) (`src/sketches/geode`).
- **Interactive Studio Extensions**: 3D mouse/touch orbit with smooth post-drag momentum blending back into continuous slow rotation, viewport-aware continuous crystal growth (growing sideways on wider viewports or upwards on taller viewports until filling the screen, then slowing down), 8 mineralogical crystal habits, settings randomization, and custom preset saving.

## Features

- **Interactive 3D Orbit, Momentum & Zoom**: Click and drag anywhere on the canvas (or 1-finger touch) to orbit horizontally and vertically; releasing the drag coasts with smooth angular momentum and seamlessly settles back into the continuous slow rotation. Scroll or pinch to zoom, and double-click to reset the camera.
- **Viewport-Aware Auto-Growth & Plant Grow Button (`G`)**:
  - **Auto Grow (ON by default in Settings)**: Crystals automatically nucleate and grow outward until they fill the screen, then smoothly slow down.
  - **Grow Plant Button (`G`)**: Click the sprout/plant button in the top-right HUD anytime to surge crystal growth even further — sprouting additional crystal branches and extending spire lengths with every click.
  - **Aspect-Aware Growth Direction**: Measures the live viewport (`width` vs `height`) and camera frustum so crystals grow **sideways** when there is more horizontal width or **upwards** when there is more vertical height.
  - **8 Crystal Habits**: *Quartz Cluster*, *Cathedral Spire*, *Twin Branching*, *Scepter Overgrowth*, *Octahedral Fluorite*, *Tabular Gem*, *Druzy Geode Bowl*, and *Dendritic Starburst*.
  - **Rich Morphology & Optics**: Prism fold symmetry (`3..6`), shaft taper, crown bevels, stepped bismuth/hopper terraces, multi-generation concentric growth phantoms, bi-colour core-to-rim absorption zoning, golden rutile needles, iridescent thin-film cleavage fractures, and wavelength-split chromatic dispersion (`fire`).
- **Settings Randomization (`R`)**: One-click mineralogical randomizer generating physically plausible crystal habits, jewel-tone OKHSL absorption colours, inclusions, and studio softbox lighting.
- **Curated Presets & Custom Settings Saving**:
  - **10 Curated Gem Presets**: Fixed-width preset bar (`‹` / `›`) with *Amethyst Cathedral*, *Classic Quartz Prism*, *Phantom Emerald Twin*, *Stepped Bismuth Hopper*, *High-Dispersion Diamond*, *Ruby Druzy Spire*, *Watermelon Tourmaline*, *Glacial Ice Dendrite*, *Rutilated Smoky Scepter*, and *Tabular Aquamarine*.
  - **Custom Preset Saving**: Save, load, and delete custom named presets in `localStorage`, copy shareable URLs with encoded settings (`?preset=...&s=...`), and export/import JSON state files.

## Run Locally

Zero dependencies and no build step required:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000>.
