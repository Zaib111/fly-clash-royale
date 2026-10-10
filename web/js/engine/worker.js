// Runs a whole match in the browser (no Python): the game, the bots, and the fly brain.
// Speaks the same snapshot format as play3d.py so the viewer can't tell the difference.
import {
  CARDS, DT, DECISION_TICKS, G, GW, NT, BRIDGES, HEIGHT, OWN_SIDE_END, POCKET_END, RIVER_HI, RIVER_LO, WIDTH,
  Match, RandomBot, ScriptedBot, actionMask, applyAction, decodeAction, observe, rng,
} from './game.js';
import { FlyModel, gunzip } from './fly.js';

const GROUPS = ['central brain', 'eyes & optic lobes', 'other senses', 'descending & motor'];
const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const r3 = (v) => Math.round(v * 1000) / 1000;

async function loadModel(base) {
  const meta = await (await fetch(`${base}/meta.json`)).json();
  const res = await fetch(`${base}/fly.bin.gz`);
  if (!res.ok) throw new Error(`could not download the fly (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    post({ type: 'progress', text: `Downloading the fly's brain… ${(got / 1e6).toFixed(1)}${total ? ` / ${(total / 1e6).toFixed(1)}` : ''} MB` });
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  post({ type: 'progress', text: 'Wiring up 134,181 neurons…' });
  return new FlyModel(meta, await gunzip(buf.buffer));
}

class FlyAgent {
  constructor(model, seed) {
    this.model = model;
    this.r = rng(seed);
    this.name = 'Fly';
    this.kind = 'fly';
    this.reset();
  }
  reset() { this.h = this.model.initialState(); this.top = []; this.probs = null; this.sp = null; this.last = 0; }
  act(m, p) {
    const [sp, vec] = observe(m, p);
    const mask = actionMask(m, p);
    const { probs, h } = this.model.forward(sp, vec, mask, this.h);
    this.h = h;
    let x = this.r(), a = 0;
    for (let k = 0; k < probs.length; k++) { x -= probs[k]; if (x <= 0) { a = k; break; } }
    if (!mask[a]) a = 0;
    const options = new Map();
    for (let k = 0; k < probs.length; k++) {
      if (probs[k] > 1e-4) { const d = describe(m, p, k); options.set(d, (options.get(d) || 0) + probs[k]); }
    }
    this.top = [...options.entries()].sort((u, v) => v[1] - u[1]).slice(0, 3);
    Object.assign(this, { probs, sp, last: a });
    return a;
  }
}

function describe(m, p, a) {
  const d = decodeAction(a);
  if (!d) return 'Wait';
  const [slot, cx, cy] = d;
  const name = CARDS[m.hand(p)[slot]].name;
  const lane = cx <= 2 ? 'left' : cx <= 5 ? 'middle' : 'right';
  const depth = cy <= 2 ? 'back' : cy <= 5 ? 'mid' : cy === 6 ? 'bridge' : cy <= 8 ? 'river' : 'enemy side';
  return `${name} - ${lane}, ${depth}`;
}

function toB64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

class Game {
  constructor(specs, model, speed) {
    this.model = model;
    this.players = specs.map((s, i) => (s === 'human' ? { name: 'You', kind: 'human', act: () => 0, reset() {} }
      : s === 'random' ? new RandomBot(7 + i, 0.15) : s === 'scripted' ? new ScriptedBot() : new FlyAgent(model, 11 + i)));
    const flies = this.players.filter((p) => p.kind === 'fly').length;
    if (flies === 2) this.players.forEach((p, i) => { if (p.kind === 'fly') p.name = i === 0 ? 'Fly (blue)' : 'Fly (red)'; });
    this.human = specs.indexOf('human') >= 0 ? specs.indexOf('human') : null;
    this.view = this.human === 1 ? 1 : 0;
    this.speed = speed;
    this.paused = false;
    this.matchId = 0;
    this.newMatch();
  }

  newMatch() {
    this.matchId += 1;
    this.m = new Match((Date.now() + this.matchId * 7919) >>> 0);
    this.m.events = [];
    this.players.forEach((p) => p.reset());
    this.decisions = 0;
    this.thinking = {};
    this.winChance = null;
  }

  info() {
    const brains = {};
    const mm = this.model && this.model.meta;
    this.players.forEach((p, i) => {
      if (p.kind === 'fly') brains[i] = { points: this.model.vizIdx.length, neurons: mm.n, connections: mm.edges, synapses: mm.synapses, groups: GROUPS };
    });
    return {
      arena: { w: WIDTH, h: HEIGHT, river: [RIVER_LO, RIVER_HI], bridges: BRIDGES, troop_limit: OWN_SIDE_END, pocket_end: POCKET_END, dt: DT },
      cards: CARDS.map((c) => ({ name: c.name, cost: c.cost, kind: c.kind, radius: c.radius, is_air: c.is_air, count: c.count, spell_radius: c.spell_radius, range: c.range })),
      players: this.players.map((p) => ({ name: p.name, kind: p.kind })),
      view: this.view, human: this.human, has_value: !!this.model, brains, static: true,
    };
  }

  brainFrame(agent, seat) {
    const model = this.model, m = this.m, h = agent.h;
    const idx = model.vizIdx, nShow = idx.length;
    const vis = new Float32Array(nShow);
    for (let i = 0; i < nShow; i++) vis[i] = h[idx[i]];
    const sorted = Float32Array.from(vis).sort();
    const scale = Math.max(sorted[Math.floor(0.995 * (nShow - 1))], 0.05);
    const act = new Uint8Array(nShow);
    for (let i = 0; i < nShow; i++) act[i] = Math.round(Math.min(1, vis[i] / scale) ** 0.8 * 255);
    let active = 0, activeOut = 0;
    for (let i = 0; i < h.length; i++) if (h[i] > 0.05) active++;
    for (const i of model.readIdx) if (h[i] > 0.05) activeOut++;
    const sp = agent.sp;
    const planes = new Uint8Array(5 * G);
    for (let cell = 0; cell < G; cell++) {
      let own = 0, enemy = 0;
      for (let k = 0; k < NT; k++) { own += sp[k * G + cell]; enemy += sp[(NT + k) * G + cell]; }
      const vals = [own / 1.5, enemy / 1.5, sp[2 * NT * G + cell], sp[(2 * NT + 1) * G + cell], sp[(2 * NT + 2) * G + cell] + sp[(2 * NT + 3) * G + cell]];
      vals.forEach((v, j) => { planes[j * G + cell] = Math.round(Math.min(1, Math.max(0, v)) * 255); });
    }
    const probs = agent.probs;
    const intent = new Array(G).fill(0), slots = [0, 0, 0, 0];
    for (let s = 0; s < 4; s++) for (let c = 0; c < G; c++) { const v = probs[1 + s * G + c]; intent[c] += v; slots[s] += v; }
    const d = decodeAction(agent.last);
    return {
      act: toB64(act), active, active_out: activeOut, vision: toB64(planes),
      intent: intent.map((v) => Math.round(v * 1e4) / 1e4), slots: slots.map((v) => Math.round(v * 1e4) / 1e4),
      wait: Math.round(probs[0] * 1e4) / 1e4, chosen: d ? [d[0], d[2] * GW + d[1]] : null,
      hand: m.hand(seat), elixir: r3(m.elixir[seat]),
    };
  }

  snapshot(events, brains) {
    const m = this.m;
    return {
      match: this.matchId, t: r3(m.time), tick: m.ticks, speed: this.speed, paused: this.paused,
      units: m.units.map((u) => {
        const t = u.target && u.target.alive ? u.target : null;
        return [u.uid, u.card, u.owner, r3(u.x), r3(u.y), Math.round(u.hp), Math.round(u.max_hp), r3(Math.max(0, u.deploy)), t ? r3(t.x) : null, t ? r3(t.y) : null];
      }),
      towers: m.towers.flat().map((t) => [t.owner, t.is_king ? 1 : 0, t.x, t.y, Math.round(t.hp), Math.round(t.max_hp), t.active ? 1 : 0, t.alive ? 1 : 0]),
      elixir: m.elixir.map(r3), hand: [m.hand(0), m.hand(1)], next: [m.next_card(0), m.next_card(1)],
      crowns: [m.crowns(0), m.crowns(1)], time_left: r3(m.time_left()), double: m.double_elixir(), done: m.done,
      winner: m.winner, events: events.map((e) => e.map((v) => (typeof v === 'number' ? r3(v) : v))), brains,
      thinking: this.thinking, decisions: this.decisions, pockets: [m.pockets(0), m.pockets(1)], win_chance: this.winChance,
    };
  }

  tick() {
    const m = this.m;
    const brains = {};
    if (m.ticks % DECISION_TICKS === 0) {
      this.decisions += 1;
      if (this.model) {
        const [sp, vec] = observe(m, this.view);
        this.winChance = r3(this.model.winChance(sp, vec));
      }
      const actions = this.players.map((pl, p) => {
        const a = pl.act(m, p);
        if (pl.kind === 'fly') {
          this.thinking[p] = pl.top.map(([d, pr]) => [d, r3(pr)]);
          brains[p] = this.brainFrame(pl, p);
        }
        return a;
      });
      actions.forEach((a, p) => applyAction(m, p, a));
    }
    m.step();
    const events = m.events;
    m.events = [];
    return this.snapshot(events, brains);
  }

  run() {
    let next = performance.now();
    let lastIdle = 0;
    const loop = () => {
      const now = performance.now();
      if (this.paused || this.m.done) {
        if (now - lastIdle > 250) { post({ type: 'snap', data: this.snapshot([], {}) }); lastIdle = now; }
        next = now;
        setTimeout(loop, 30);
        return;
      }
      post({ type: 'snap', data: this.tick() });
      next += (DT * 1000) / this.speed;
      const delay = next - performance.now();
      if (delay < -500) next = performance.now();
      setTimeout(loop, Math.max(0, delay));
    };
    loop();
  }
}

let game = null;
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.cmd === 'init') {
      const model = msg.players.includes('fly') ? await loadModel(msg.model || '../../model') : null;
      game = new Game(msg.players, model, msg.speed || 1);
      const info = game.info();
      post({ type: 'init', info });
      if (model) {
        const n = model.vizIdx.length;
        const buf = new ArrayBuffer(n * 13);
        new Float32Array(buf, 0, n * 3).set(model.vizPos);
        new Uint8Array(buf, n * 12, n).set(model.vizGroup);
        post({ type: 'brainviz', buffer: buf }, [buf]);
      }
      game.run();
    } else if (msg.cmd === 'play') {
      const ok = game && game.human !== null && game.m.play(game.human, msg.slot, msg.x, msg.y);
      post({ type: 'played', id: msg.id, ok: !!ok });
    } else if (msg.cmd === 'control') {
      if (msg.what === 'pause') game.paused = !game.paused;
      else if (msg.what === 'speed') game.speed = Math.min(16, Math.max(0.25, msg.value));
      else if (msg.what === 'restart') { game.newMatch(); game.paused = false; }
    }
  } catch (err) {
    post({ type: 'error', text: String(err && err.message || err) });
  }
};
