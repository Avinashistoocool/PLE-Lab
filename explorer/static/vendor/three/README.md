# Vendored three.js

The 3D token map (`explorer/static/map3d.js`) uses this copy of three.js. The copy is in the repository, thus the explorer works fully offline and without a build step.

| file(s) | source | version |
|---|---|---|
| `three.module.js` | `three/build/three.module.js` | r170 (`three@0.170.0`) |
| `addons/controls/OrbitControls.js` | `three/examples/jsm/controls/` | r170 |
| `addons/postprocessing/{EffectComposer,RenderPass,UnrealBloomPass,OutputPass,ShaderPass,MaskPass,Pass}.js` | `three/examples/jsm/postprocessing/` | r170 |
| `addons/shaders/{CopyShader,LuminosityHighPassShader,OutputShader}.js` | `three/examples/jsm/shaders/` | r170 |
| `addons/renderers/CSS2DRenderer.js` | `three/examples/jsm/renderers/` | r170 |

**Local modification:** the original addons import from the bare specifier `'three'`. We changed these imports to relative paths (`'../../three.module.js'`). Thus, the browser can load them without an import map. We made no other changes.

## Licence

three.js is released under the MIT License, Copyright © 2010-2024 three.js authors. The full text is
in [`LICENSE`](LICENSE) next to this file.
