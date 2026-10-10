// Runs the exported fly (web/model) on inputs from test_js_parity.py and prints its outputs.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { FlyModel } from '../web/js/engine/fly.js';

const [modelDir, inputPath] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(`${modelDir}/meta.json`, 'utf8'));
const raw = gunzipSync(readFileSync(`${modelDir}/fly.bin.gz`));
const fly = new FlyModel(meta, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const inp = JSON.parse(readFileSync(inputPath, 'utf8'));
const t0 = performance.now();
const { probs, h } = fly.forward(Float32Array.from(inp.sp), Float32Array.from(inp.vec), Uint8Array.from(inp.mask), Float32Array.from(inp.h));
const ms = performance.now() - t0;
const win = fly.winChance(Float32Array.from(inp.sp), Float32Array.from(inp.vec));
console.log(JSON.stringify({ probs: Array.from(probs), h_sample: inp.h_sample_idx.map((i) => h[i]), win, ms }));
