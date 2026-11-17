// Test helper: node resize_cli.mjs <in.rgb> <w> <h> <outW> <outH> <out.rgb>
// Runs the browser preprocessing code (web/static/js/preprocess.js) on raw RGB bytes.
import { readFileSync, writeFileSync } from "node:fs";
import { resizeRGB } from "../../web/static/js/preprocess.js";

const [inPath, w, h, outW, outH, outPath] = process.argv.slice(2);
const out = resizeRGB(new Uint8Array(readFileSync(inPath)), +w, +h, +outW, +outH);
writeFileSync(outPath, out);
