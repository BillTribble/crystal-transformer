#!/usr/bin/env python3
"""Bundles all Crystals ES modules into a single zero-waterfall inline script inside index.html."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent

MODULES = [
    ("random", "kit/random.js", [], ["rng", "hash1", "hash2"], None),
    ("okhsl", "kit/okhsl.js", [], ["oklchToRgb", "okhslRow", "okhslToHex", "hexToOkhsl"], None),
    ("standard", "kit/standard.js", [], ["STANDARD_KEYS", "playback"], None),
    ("film", "kit/film.js", [], ["FILM_GLSL", "film"], None),
    (
        "gl",
        "kit/gl.js",
        [],
        [
            "FULLSCREEN_VERT",
            "program",
            "shaderParams",
            "paramDecls",
            "uploadParams",
            "frameUniforms",
            "shaderHeader",
            "uploadFrame",
            "probeFloatTargets",
        ],
        None,
    ),
    ("chrome", "kit/chrome.js", [], ["setupChrome"], None),
    ("loop", "kit/loop.js", [], ["createLoop"], None),
    ("kit_params", "kit/params.js", ["const { hexToOkhsl, okhslToHex } = __mod_okhsl;"], ["createModel"], None),
    (
        "colour_picker",
        "kit/colour-picker.js",
        ["const { hexToOkhsl, okhslRow, okhslToHex } = __mod_okhsl;"],
        ["createColourPicker"],
        None,
    ),
    (
        "panel",
        "kit/panel.js",
        ["const { createColourPicker } = __mod_colour_picker;"],
        ["STATE_TAG", "buildPanel"],
        None,
    ),
    (
        "cluster",
        "cluster.js",
        [],
        ["MAX_CRYSTALS", "CRYSTAL_PLANES", "PLANES_AT", "ROW_W", "ROWS", "cluster", "resolveHabitSystem", "buildCluster", "resetEvolution", "stepEvolution"],
        None,
    ),
    (
        "sound",
        "sound.js",
        [],
        ["TRACK_ROTATE_SECONDS", "ENSO_REVERB_MODES", "ENSO_DRONE_MODES", "ENSO_WASH_INSTRUMENTS", "SAMPLE_ZONES", "ENSO_TRACKS", "ENSO_TRACK_NAMES", "createEnsoSound"],
        None,
    ),
    ("manifest", "manifest.js", [], [], "__default_manifest"),
    (
        "params",
        "params.js",
        [
            "const { playback } = __mod_standard;",
            "const { film } = __mod_film;",
            "const { okhslToHex } = __mod_okhsl;",
            "const { ENSO_TRACK_NAMES, ENSO_REVERB_MODES, ENSO_DRONE_MODES, ENSO_WASH_INSTRUMENTS } = __mod_sound;",
        ],
        ["HABITS", "MIRROR_MODES", "presets", "randomize"],
        "__default_params",
    ),
    (
        "run",
        "kit/run.js",
        [
            "const { createModel } = __mod_kit_params;",
            "const { buildPanel } = __mod_panel;",
            "const { setupChrome } = __mod_chrome;",
            "const { createLoop } = __mod_loop;",
            "const { hexToOkhsl, okhslToHex } = __mod_okhsl;",
            "const { buildCluster, cluster, resetEvolution, resolveHabitSystem, stepEvolution } = __mod_cluster;",
            "const { createEnsoSound } = __mod_sound;",
        ],
        ["run"],
        None,
    ),
    (
        "shader",
        "shader.js",
        [
            "const { FILM_GLSL } = __mod_film;",
            "const { cluster, CRYSTAL_PLANES, MAX_CRYSTALS, PLANES_AT, ROW_W, ROWS } = __mod_cluster;",
            "const { MIRROR_MODES } = __mod_params;",
        ],
        [],
        "__default_shader",
    ),
    (
        "sketch",
        "sketch.js",
        [
            "const SHADER = __mod_shader.default;",
            "const { buildCluster, cluster } = __mod_cluster;",
            "const { FULLSCREEN_VERT, frameUniforms, probeFloatTargets, program, shaderHeader, shaderParams, uploadFrame, uploadParams } = __mod_gl;",
        ],
        [],
        "__default_sketch",
    ),
]


def transform_module(name, rel_path, preamble_lines, named_exports, default_var):
  src = (ROOT / rel_path).read_text(encoding="utf-8")
  # Strip top-level import lines
  src = re.sub(r"^\s*import\s+[^\n]+;\s*$", "", src, flags=re.MULTILINE)
  # Replace export default
  if default_var:
    src = re.sub(
        r"^\s*export\s+default\s+function\s+([A-Za-z0-9_$]+)",
        f"function {default_var}",
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r"^\s*export\s+default\s+",
        f"const {default_var} = ",
        src,
        flags=re.MULTILINE,
    )
  # Replace named exports
  src = re.sub(
      r"^\s*export\s+(const|let|var|function)\s+",
      r"\1 ",
      src,
      flags=re.MULTILINE,
  )

  ret_items = list(named_exports)
  if default_var:
    ret_items.append(f"default: {default_var}")
  ret_expr = "{ " + ", ".join(ret_items) + " }"
  preamble = "\n  ".join(preamble_lines)
  if preamble:
    preamble = "  " + preamble + "\n"

  return (
      f"  // === Module: {rel_path} ===\n"
      f"  const __mod_{name} = (() => {{\n"
      f"{preamble}"
      f"{src}\n"
      f"    return {ret_expr};\n"
      f"  }})();\n"
  )


def main():
  chunks = [
      "  // Self-contained zero-waterfall bundle (inlines kit/*.js, cluster.js, shader.js, params.js, sketch.js, manifest.js)"
  ]
  for mod in MODULES:
    chunks.append(transform_module(*mod))

  chunks.append(
      "  const manifest = __mod_manifest.default;\n"
      "  const params = __mod_params.default;\n"
      "  const { presets, randomize } = __mod_params;\n"
      "  const sketch = __mod_sketch.default;\n"
      "  const { run } = __mod_run;\n"
      "  run('crystals', { manifest, params, presets, randomize, sketch });"
  )
  bundle_js = "\n".join(chunks)

  index_path = ROOT / "index.html"
  html = index_path.read_text(encoding="utf-8")
  html = re.sub(r"<title>.*?</title>", "<title>Crystal Transformer</title>", html, count=1)
  html = re.sub(
      r'(<span id="preset-name">).*?(</span>)(?=</button>|<span class="hud-preset-chevron")',
      r'\1<span class="preset-name-outgoing">Classic Quartz Prism</span><span class="preset-name-incoming" aria-hidden="true"></span>\2',
      html,
      count=1,
  )
  og_block = (
      '  <meta property="og:type" content="website" />\n'
      '  <meta property="og:title" content="Crystal Transformer" />\n'
      '  <meta property="og:description" content="Ray traced crystal experiment with procedural growth and generative ambient soundtrack." />\n'
      '  <meta property="og:url" content="https://billtribble.github.io/crystal-transformer/" />\n'
      '  <meta property="og:image" content="https://billtribble.github.io/crystal-transformer/preview.png" />\n'
      '  <meta property="og:image:type" content="image/png" />\n'
      '  <meta property="og:image:width" content="1080" />\n'
      '  <meta property="og:image:height" content="1080" />\n'
      '  <meta name="twitter:card" content="summary_large_image" />\n'
      '  <meta name="twitter:title" content="Crystal Transformer" />\n'
      '  <meta name="twitter:description" content="Ray traced crystal experiment with procedural growth and generative ambient soundtrack." />\n'
      '  <meta name="twitter:image" content="https://billtribble.github.io/crystal-transformer/preview.png" />'
  )
  if 'property="og:image"' not in html:
    html = html.replace(
        '<meta name="theme-color" content="#030304" />',
        '<meta name="theme-color" content="#030304" />\n' + og_block,
        1,
    )
  pattern = re.compile(r"<script type=\"module\">.*?</script>", re.DOTALL)
  replacement = f'<script type="module">\n{bundle_js}\n  </script>'
  new_html, count = pattern.subn(lambda _: replacement, html, count=1)
  if count != 1:
    raise RuntimeError("Failed to locate <script type=\"module\"> block in index.html")
  index_path.write_text(new_html, encoding="utf-8")
  print(f"BUNDLED_OK bytes={len(new_html)}")


if __name__ == "__main__":
  main()
