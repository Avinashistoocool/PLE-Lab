// ES-module shim for the vendored d3 v7.9.0 UMD bundle (ISC licence, see LICENSE).
// The UMD file attaches itself to globalThis.d3 when evaluated as a module.
import './d3.min.js';

const d3 = globalThis.d3;
export default d3;
export { d3 };
