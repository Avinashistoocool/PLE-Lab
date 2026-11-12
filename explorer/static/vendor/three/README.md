# Vendored three.js

Used by `explorer/static/map3d.js` (the 3D token map). Vendored so the explorer works fully offline
and without a build step.

| file(s) | source | version |
|---|---|---|
| `three.module.js` | `three/build/three.module.js` | r170 (`three@0.170.0`) |
| `addons/controls/OrbitControls.js` | `three/examples/jsm/controls/` | r170 |
| `addons/postprocessing/{EffectComposer,RenderPass,UnrealBloomPass,OutputPass,ShaderPass,MaskPass,Pass}.js` | `three/examples/jsm/postprocessing/` | r170 |
| `addons/shaders/{CopyShader,LuminosityHighPassShader,OutputShader}.js` | `three/examples/jsm/shaders/` | r170 |
| `addons/renderers/CSS2DRenderer.js` | `three/examples/jsm/renderers/` | r170 |

**Local modification:** the addons originally import from the bare specifier `'three'`. They were
rewritten to relative paths (`'../../three.module.js'`) so they load in the browser without an import
map. No other changes.

## Licence

three.js is released under the MIT License, Copyright © 2010-2024 three.js authors. The full text is
in [`LICENSE`](LICENSE) next to this file.
