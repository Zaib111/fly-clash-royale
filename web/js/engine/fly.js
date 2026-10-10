// The fly brain + value network in plain JavaScript (port of fly/brain.py and fly/critic.py,
// inference only). Loads the files written by export_web.py.
import { C, G, GW, GH, V, NUM_ACTIONS } from './game.js';

const TYPED = { int32: Int32Array, uint32: Uint32Array, uint16: Uint16Array, uint8: Uint8Array, float32: Float32Array, float16: Uint16Array };

// float16 bits -> float32
const F16 = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  const s = i & 0x8000 ? -1 : 1, e = (i >> 10) & 0x1f, f = i & 0x3ff;
  F16[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
}

export async function gunzip(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x1f || u8[1] !== 0x8b) return buf;      // server already decompressed it
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

export class FlyModel {
  constructor(meta, buffer) {
    this.meta = meta;
    const A = {};
    for (const [name, m] of Object.entries(meta.arrays)) A[name] = new TYPED[m.dtype](buffer, m.offset, m.length);
    this.n = meta.n;
    this.steps = meta.steps;
    this.leak = meta.leak;
    this.crow = A.crow;
    // undo the per-row delta encoding of column indices
    const col = new Uint32Array(A.col_delta.length);
    for (let r = 0; r < this.n; r++) {
      let c = 0;
      for (let e = this.crow[r]; e < this.crow[r + 1]; e++) { c = e === this.crow[r] ? A.col_delta[e] : c + A.col_delta[e]; col[e] = c; }
    }
    this.col = col;
    this.w = new Float32Array(A.w.length);
    for (let i = 0; i < A.w.length; i++) this.w[i] = F16[A.w[i]];
    Object.assign(this, {
      bias: A.bias, eyeIdx: A.eye_idx, eyeCell: A.eye_cell, eyeW: A.eye_w, eyeB: A.eye_b,
      senseIdx: A.sense_idx, senseW: A.sense_w, senseB: A.sense_b, readIdx: A.read_idx,
      d1w: A.dec1_w, d1b: A.dec1_b, d2w: A.dec2_w, d2b: A.dec2_b,
      c0w: A['critic.conv.0.weight'], c0b: A['critic.conv.0.bias'], c2w: A['critic.conv.2.weight'], c2b: A['critic.conv.2.bias'],
      h0w: A['critic.head.0.weight'], h0b: A['critic.head.0.bias'], h2w: A['critic.head.2.weight'], h2b: A['critic.head.2.bias'],
      vizIdx: A.viz_idx, vizPos: A.viz_pos, vizGroup: A.viz_group,
    });
    this.hidden = this.d1b.length;
    this.u = new Float32Array(this.n);
    this.tmp = new Float32Array(this.n);
  }

  static async load(base = 'model') {
    const meta = await (await fetch(`${base}/meta.json`)).json();
    const buf = await gunzip(await (await fetch(`${base}/fly.bin.gz`)).arrayBuffer());
    return new FlyModel(meta, buf);
  }

  initialState() { return new Float32Array(this.n); }

  // One decision. sp: Float32Array(C*G), vec: Float32Array(V), mask: Uint8Array(NUM_ACTIONS),
  // h: Float32Array(n) (not modified). Returns { probs, h }.
  forward(sp, vec, mask, h) {
    const n = this.n, u = this.u;
    u.set(this.bias);                                           // u = input current + bias
    const ne = this.eyeIdx.length;
    for (let k = 0; k < ne; k++) {
      const cell = this.eyeCell[k];
      let s = this.eyeB[k];
      for (let c = 0; c < C; c++) s += this.eyeW[k * C + c] * sp[c * G + cell];
      u[this.eyeIdx[k]] += s;
    }
    const ns = this.senseIdx.length;
    for (let j = 0; j < ns; j++) {
      let s = this.senseB[j];
      for (let i = 0; i < V; i++) s += this.senseW[j * V + i] * vec[i];
      u[this.senseIdx[j]] += s;
    }
    let cur = Float32Array.from(h), next = this.tmp;
    const a = this.leak, crow = this.crow, col = this.col, w = this.w;
    for (let step = 0; step < this.steps; step++) {
      for (let r = 0; r < n; r++) {
        let acc = u[r];
        for (let e = crow[r], end = crow[r + 1]; e < end; e++) acc += w[e] * cur[col[e]];
        next[r] = (1 - a) * cur[r] + a * (acc > 0 ? Math.tanh(acc) : 0);
      }
      [cur, next] = [next, cur];
    }
    this.tmp = next;
    // decoder: descending + motor neurons -> action logits
    const nr = this.readIdx.length, H = this.hidden;
    const z = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      let s = this.d1b[j];
      for (let i = 0; i < nr; i++) s += this.d1w[j * nr + i] * cur[this.readIdx[i]];
      z[j] = s > 0 ? s : 0;
    }
    const logits = new Float64Array(NUM_ACTIONS);
    let mx = -Infinity;
    for (let k = 0; k < NUM_ACTIONS; k++) {
      if (!mask[k]) { logits[k] = -Infinity; continue; }
      let s = this.d2b[k];
      for (let j = 0; j < H; j++) s += this.d2w[k * H + j] * z[j];
      logits[k] = s;
      if (s > mx) mx = s;
    }
    const probs = new Float32Array(NUM_ACTIONS);
    let tot = 0;
    for (let k = 0; k < NUM_ACTIONS; k++) { const e = mask[k] ? Math.exp(logits[k] - mx) : 0; probs[k] = e; tot += e; }
    for (let k = 0; k < NUM_ACTIONS; k++) probs[k] /= tot;
    return { probs, h: cur };
  }

  // Value network: estimated chance (0-1) that the observing player wins.
  winChance(sp, vec) {
    const conv = (input, cin, H, W, weight, b, cout, stride) => {
      const Ho = Math.floor((H + 2 - 3) / stride) + 1, Wo = Math.floor((W + 2 - 3) / stride) + 1;
      const out = new Float32Array(cout * Ho * Wo);
      for (let o = 0; o < cout; o++) {
        for (let y = 0; y < Ho; y++) {
          for (let x = 0; x < Wo; x++) {
            let s = b[o];
            for (let c = 0; c < cin; c++) {
              for (let ky = 0; ky < 3; ky++) {
                const iy = y * stride + ky - 1;
                if (iy < 0 || iy >= H) continue;
                for (let kx = 0; kx < 3; kx++) {
                  const ix = x * stride + kx - 1;
                  if (ix < 0 || ix >= W) continue;
                  s += weight[((o * cin + c) * 3 + ky) * 3 + kx] * input[(c * H + iy) * W + ix];
                }
              }
            }
            out[(o * Ho + y) * Wo + x] = s > 0 ? s : 0;
          }
        }
      }
      return [out, Ho, Wo];
    };
    const [a1] = conv(sp, C, GH, GW, this.c0w, this.c0b, 32, 1);
    const [a2] = conv(a1, 32, GH, GW, this.c2w, this.c2b, 32, 2);
    const x = new Float32Array(a2.length + V);
    x.set(a2);
    x.set(vec, a2.length);
    const H = this.h0b.length;
    let v = this.h2b[0];
    for (let j = 0; j < H; j++) {
      let s = this.h0b[j];
      for (let i = 0; i < x.length; i++) s += this.h0w[j * x.length + i] * x[i];
      if (s > 0) v += this.h2w[j] * s;
    }
    return Math.min(1, Math.max(0, (v + 1) / 2));
  }
}
