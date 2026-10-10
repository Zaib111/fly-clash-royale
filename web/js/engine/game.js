// JavaScript port of game/cards.py, game/sim.py, game/env.py and game/bots.py, so the
// whole game can run in the browser. Keep it in lockstep with the Python version:
// tests/test_js_parity.py checks both produce the same matches and observations.

// ------------------------------------------------------------------ cards
const card = (o) => ({
  kind: 'troop', count: 1, hp: 0, damage: 0, hit_speed: 1, range: 1, speed: 1, radius: 0.5,
  is_air: false, hits_air: false, buildings_only: false, sight: 5.5,
  spell_radius: 0, tower_damage: 0, spell_delay: 0, ...o,
});
export const CARDS = [
  card({ name: 'Knight', cost: 3, hp: 1452, damage: 167, hit_speed: 1.2, range: 0.7, speed: 1.0, radius: 0.5 }),
  card({ name: 'Archers', cost: 3, count: 2, hp: 252, damage: 89, hit_speed: 0.9, range: 5.0, speed: 1.0, radius: 0.4, hits_air: true }),
  card({ name: 'Giant', cost: 5, hp: 3275, damage: 211, hit_speed: 1.5, range: 0.7, speed: 0.75, radius: 0.75, buildings_only: true, sight: 7.5 }),
  card({ name: 'Goblins', cost: 2, count: 3, hp: 167, damage: 99, hit_speed: 1.1, range: 0.5, speed: 2.0, radius: 0.35 }),
  card({ name: 'Musketeer', cost: 4, hp: 598, damage: 181, hit_speed: 1.0, range: 6.0, speed: 1.0, radius: 0.5, hits_air: true, sight: 6.0 }),
  card({ name: 'MiniPekka', cost: 4, hp: 1129, damage: 598, hit_speed: 1.6, range: 0.7, speed: 1.5, radius: 0.5 }),
  card({ name: 'Fireball', cost: 4, kind: 'spell', damage: 572, tower_damage: 172, spell_radius: 2.5, spell_delay: 0.6 }),
  card({ name: 'Minions', cost: 3, count: 3, hp: 190, damage: 84, hit_speed: 1.0, range: 1.6, speed: 1.5, radius: 0.4, is_air: true, hits_air: true }),
];
export const CARD_INDEX = Object.fromEntries(CARDS.map((c, i) => [c.name, i]));
export const NUM_CARDS = CARDS.length;
export const TROOP_IDS = CARDS.map((c, i) => (c.kind === 'troop' ? i : -1)).filter((i) => i >= 0);
const PRINCESS = { hp: 3052, damage: 109, hit_speed: 0.8, range: 7.5, radius: 1.5 };
const KING = { hp: 4824, damage: 109, hit_speed: 1.0, range: 7.0, radius: 2.0 };
const DEPLOY_TIME = 1.0, START_ELIXIR = 5.0, MAX_ELIXIR = 10.0, ELIXIR_PER_SEC = 1 / 2.8;

// ------------------------------------------------------------------ arena
export const WIDTH = 18.0, HEIGHT = 32.0, RIVER_LO = 15.0, RIVER_HI = 17.0;
export const BRIDGES = [3.5, 14.5];
export const TICKS_PER_SEC = 10, DT = 1.0 / TICKS_PER_SEC, MATCH_SECONDS = 180.0;
export const OWN_SIDE_END = 14.0, POCKET_END = 25.0;
const DOUBLE_ELIXIR_AT = 120.0;

// mulberry32: small seeded RNG for deck shuffles and bots
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Tower {
  constructor(owner, isKing, x, y) {
    const s = isKing ? KING : PRINCESS;
    Object.assign(this, { owner, is_king: isKing, x, y, hp: s.hp, max_hp: s.hp, damage: s.damage,
      hit_speed: s.hit_speed, range: s.range, radius: s.radius, cooldown: 0, target: null,
      active: !isKing, alive: true, is_air: false, is_building: true });
  }
}

class Unit {
  constructor(owner, cardId, x, y) {
    const c = CARDS[cardId];
    Object.assign(this, { owner, card: cardId, x, y, hp: c.hp, max_hp: c.hp, deploy: DEPLOY_TIME,
      cooldown: 0, target: null, alive: true, uid: 0, is_building: false, is_air: c.is_air, radius: c.radius });
  }
  get stats() { return CARDS[this.card]; }
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export class Match {
  constructor(seed = 1, cycles = null) {
    this.time = 0;
    this.ticks = 0;
    this.units = [];
    this.spells = [];
    this.towers = [0, 1].map((p) => {
      const ys = (y) => (p === 0 ? y : HEIGHT - y);
      return [new Tower(p, true, 9.0, ys(3.0)), new Tower(p, false, 3.5, ys(6.5)), new Tower(p, false, 14.5, ys(6.5))];
    });
    this.elixir = [START_ELIXIR, START_ELIXIR];
    if (cycles) this.cycle = cycles.map((c) => c.slice());
    else {
      const r = rng(seed);
      this.cycle = [0, 1].map(() => {
        const deck = [...Array(NUM_CARDS).keys()];
        for (let i = deck.length - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [deck[i], deck[j]] = [deck[j], deck[i]];
        }
        return deck;
      });
    }
    this.winner = null;
    this.done = false;
    this.tower_damage_dealt = [0, 0];
    this.events = null;
    this._next_uid = 1;
  }

  _emit(...ev) { if (this.events) this.events.push([this.time, ...ev]); }

  hand(p) { return this.cycle[p].slice(0, 4); }
  next_card(p) { return this.cycle[p][4]; }
  crowns(p) {
    const enemy = this.towers[1 - p];
    if (!enemy[0].alive) return 3;
    return enemy.slice(1).filter((t) => !t.alive).length;
  }
  double_elixir() { return this.time >= DOUBLE_ELIXIR_AT; }
  time_left() { return Math.max(0, MATCH_SECONDS - this.time); }
  pockets(p) { const e = this.towers[1 - p]; return [!e[1].alive, !e[2].alive]; }

  can_deploy_troop(p, x, y) {
    if (!(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)) return false;
    const own = p === 0 ? y : HEIGHT - y;
    if (own < OWN_SIDE_END) return true;
    if (own >= RIVER_HI && own < POCKET_END) {
      const [left, right] = this.pockets(p);
      return x < WIDTH / 2 ? left : right;
    }
    return false;
  }

  can_play(p, slot, x, y) {
    if (this.done || !(slot >= 0 && slot < 4)) return false;
    const c = CARDS[this.cycle[p][slot]];
    if (this.elixir[p] < c.cost) return false;
    if (c.kind === 'spell') return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;
    return this.can_deploy_troop(p, x, y);
  }

  play(p, slot, x, y) {
    if (!this.can_play(p, slot, x, y)) return false;
    const cyc = this.cycle[p];
    const cardId = cyc[slot];
    cyc[slot] = cyc.splice(4, 1)[0];
    cyc.push(cardId);
    const c = CARDS[cardId];
    this.elixir[p] -= c.cost;
    this._emit('play', cardId, p, x, y);
    if (c.kind === 'spell') {
      this.spells.push({ owner: p, card: cardId, x, y, delay: c.spell_delay });
      this._emit('cast', cardId, p, x, y);
    } else {
      for (let i = 0; i < c.count; i++) {
        let ox = 0, oy = 0;
        if (c.count !== 1) {
          const ang = (2 * Math.PI * i) / c.count;
          ox = 0.6 * Math.cos(ang);
          oy = 0.6 * Math.sin(ang);
        }
        const ux = Math.min(Math.max(x + ox, 0.3), WIDTH - 0.3);
        const u = new Unit(p, cardId, ux, y + oy);
        u.uid = this._next_uid++;
        this.units.push(u);
      }
    }
    return true;
  }

  step(ticks = 1) {
    for (let i = 0; i < ticks; i++) {
      if (this.done) return;
      this._tick();
    }
  }

  _tick() {
    const rate = ELIXIR_PER_SEC * (this.double_elixir() ? 2 : 1);
    for (const p of [0, 1]) this.elixir[p] = Math.min(MAX_ELIXIR, this.elixir[p] + rate * DT);
    this._resolve_spells();
    for (const u of this.units) if (u.alive) this._unit_tick(u);
    for (const p of [0, 1]) for (const t of this.towers[p]) if (t.alive && t.active) this._tower_tick(t);
    this.units = this.units.filter((u) => u.alive);
    this.ticks += 1;
    this.time = this.ticks * DT;
    this._check_end();
  }

  _damage(attacker, target, amount) {
    if (!target.alive) return;
    target.hp -= amount;
    if (target.is_building) {
      this.tower_damage_dealt[attacker] += Math.min(amount, target.hp + amount);
      if (target.is_king) target.active = true;
    }
    if (target.hp <= 0) {
      target.alive = false;
      target.hp = 0;
      this._emit('death', target.is_building, target.owner, target.x, target.y, target.is_building ? null : target.card);
      if (target.is_building && !target.is_king) this.towers[target.owner][0].active = true;
    }
  }

  _resolve_spells() {
    const keep = [];
    for (const s of this.spells) {
      s.delay -= DT;
      if (s.delay > 0) { keep.push(s); continue; }
      const c = CARDS[s.card];
      const enemy = 1 - s.owner;
      this._emit('boom', s.card, s.x, s.y);
      for (const u of this.units) {
        if (u.owner === enemy && u.alive && u.deploy <= 0 && Math.hypot(u.x - s.x, u.y - s.y) <= c.spell_radius + u.radius) {
          this._damage(s.owner, u, c.damage);
        }
      }
      for (const t of this.towers[enemy]) {
        if (t.alive && Math.hypot(t.x - s.x, t.y - s.y) <= c.spell_radius + t.radius) this._damage(s.owner, t, c.tower_damage);
      }
    }
    this.spells = keep;
  }

  _valid_target(u, o) {
    const c = u.stats;
    if (!o.alive || o.owner === u.owner) return false;
    if (o.is_building) return true;
    if (c.buildings_only || o.deploy > 0) return false;
    return c.hits_air || !o.is_air;
  }

  _find_target(u) {
    const c = u.stats;
    const enemy = 1 - u.owner;
    let best = null, bestD = c.sight;
    if (!c.buildings_only) {
      for (const o of this.units) {
        if (o.owner === enemy && this._valid_target(u, o)) {
          const d = dist(u, o) - o.radius;
          if (d < bestD) { best = o; bestD = d; }
        }
      }
    }
    if (best === null) {
      bestD = Infinity;
      for (const t of this.towers[enemy]) {
        if (t.alive) {
          const d = dist(u, t);
          if (d < bestD) { best = t; bestD = d; }
        }
      }
    }
    return best;
  }

  _waypoint(u, tx, ty) {
    if (u.is_air) return [tx, ty];
    const myBottom = u.y < 16.0, tBottom = ty < 16.0;
    const inRiver = u.y >= RIVER_LO && u.y <= RIVER_HI;
    if (myBottom === tBottom && !inRiver) return [tx, ty];
    if (inRiver) return [u.x, tBottom ? RIVER_LO - 0.6 : RIVER_HI + 0.6];
    let bx = BRIDGES[0];
    for (const b of BRIDGES) if (Math.abs(u.x - b) + Math.abs(b - tx) < Math.abs(u.x - bx) + Math.abs(bx - tx)) bx = b;
    if (Math.abs(u.x - bx) > 0.2 || Math.abs(u.y - 16.0) > 1.6) return [bx, myBottom ? RIVER_LO - 0.5 : RIVER_HI + 0.5];
    return [bx, myBottom ? RIVER_HI + 0.6 : RIVER_LO - 0.6];
  }

  _in_range(u, t) { return dist(u, t) - t.radius - u.radius <= u.stats.range; }

  _unit_tick(u) {
    if (u.deploy > 0) { u.deploy -= DT; return; }
    const c = u.stats;
    let t = u.target;
    if (t === null || !t.alive || !this._in_range(u, t)) {
      t = this._find_target(u);
      if (t !== u.target) u.cooldown = Math.max(u.cooldown, c.hit_speed * 0.5);
      u.target = t;
    }
    if (t === null) return;
    if (this._in_range(u, t)) {
      u.cooldown -= DT;
      if (u.cooldown <= 0) {
        this._emit('attack', u.card, u.owner, u.x, u.y, t.x, t.y, u.uid);
        this._damage(u.owner, t, c.damage);
        u.cooldown = c.hit_speed;
      }
      return;
    }
    u.cooldown = Math.max(0, u.cooldown - DT);
    const [wx, wy] = this._waypoint(u, t.x, t.y);
    const dx = wx - u.x, dy = wy - u.y;
    const d = Math.hypot(dx, dy);
    if (d > 1e-6) {
      const step = Math.min(d, c.speed * DT);
      u.x += (dx / d) * step;
      u.y += (dy / d) * step;
    }
  }

  _tower_tick(t) {
    let tgt = t.target;
    if (tgt === null || !tgt.alive || dist(t, tgt) - tgt.radius > t.range) {
      tgt = null;
      let best = t.range;
      for (const o of this.units) {
        if (o.owner !== t.owner && o.alive && o.deploy <= 0) {
          const d = dist(t, o) - o.radius;
          if (d <= best) { tgt = o; best = d; }
        }
      }
      t.target = tgt;
    }
    if (tgt === null) { t.cooldown = Math.max(0, t.cooldown - DT); return; }
    t.cooldown -= DT;
    if (t.cooldown <= 0) {
      this._emit('tower_attack', t.is_king, t.owner, t.x, t.y, tgt.x, tgt.y);
      this._damage(t.owner, tgt, t.damage);
      t.cooldown = t.hit_speed;
    }
  }

  _check_end() {
    for (const p of [0, 1]) {
      if (!this.towers[1 - p][0].alive) { this.winner = p; this.done = true; return; }
    }
    if (this.time >= MATCH_SECONDS) {
      const c0 = this.crowns(0), c1 = this.crowns(1);
      if (c0 !== c1) this.winner = c0 > c1 ? 0 : 1;
      else {
        const [d0, d1] = this.tower_damage_dealt;
        this.winner = Math.abs(d0 - d1) < 1e-6 ? -1 : d0 > d1 ? 0 : 1;
      }
      this.done = true;
    }
  }
}

// ------------------------------------------------------------------ observations & actions (game/env.py)
export const GW = 9, GH = 16, CELL = 2.0, G = GW * GH, NUM_ACTIONS = 1 + 4 * G, DECISION_TICKS = TICKS_PER_SEC / 2;
const TROOP_PLANE = Object.fromEntries(TROOP_IDS.map((cid, i) => [cid, i]));
export const NT = TROOP_IDS.length, C = 2 * NT + 4, V = 1 + 4 * NUM_CARDS + NUM_CARDS + 4 + 1 + 1 + 2 + 2 + 6;

function cells(pred) {
  const out = new Uint8Array(G);
  for (let cy = 0; cy < GH; cy++) for (let cx = 0; cx < GW; cx++) out[cy * GW + cx] = pred(cx * CELL + CELL / 2, cy * CELL + CELL / 2) ? 1 : 0;
  return out;
}
const OWN_CELLS = cells((x, y) => y < OWN_SIDE_END);
const POCKET_CELLS = [cells((x, y) => y >= RIVER_HI && y < POCKET_END && x < WIDTH / 2),
  cells((x, y) => y >= RIVER_HI && y < POCKET_END && x >= WIDTH / 2)];

export function cellToWorld(p, cx, cy) {
  const x = cx * CELL + CELL / 2, y = cy * CELL + CELL / 2;
  return [x, p === 0 ? y : HEIGHT - y];
}
export function worldToCell(p, x, y) {
  if (p === 1) y = HEIGHT - y;
  return [Math.min(GW - 1, Math.max(0, Math.floor(x / CELL))), Math.min(GH - 1, Math.max(0, Math.floor(y / CELL)))];
}
export function decodeAction(a) {
  if (a === 0) return null;
  a -= 1;
  const slot = Math.floor(a / G), rem = a % G;
  return [slot, rem % GW, Math.floor(rem / GW)];   // slot, cx, cy
}
export const encodeAction = (slot, cx, cy) => 1 + slot * G + cy * GW + cx;
export function applyAction(m, p, a) {
  const d = decodeAction(a);
  if (!d) return false;
  const [x, y] = cellToWorld(p, d[1], d[2]);
  return m.play(p, d[0], x, y);
}

export function actionMask(m, p) {
  const mask = new Uint8Array(NUM_ACTIONS);
  mask[0] = 1;
  if (m.done) return mask;
  const [left, right] = m.pockets(p);
  const troop = new Uint8Array(G);
  for (let i = 0; i < G; i++) troop[i] = OWN_CELLS[i] | (left ? POCKET_CELLS[0][i] : 0) | (right ? POCKET_CELLS[1][i] : 0);
  m.hand(p).forEach((cid, s) => {
    const c = CARDS[cid];
    if (m.elixir[p] < c.cost) return;
    const base = 1 + s * G;
    for (let i = 0; i < G; i++) mask[base + i] = c.kind === 'spell' ? 1 : troop[i];
  });
  return mask;
}

export function observe(m, p) {
  const sp = new Float32Array(C * G);   // [plane][cy][cx]
  const vec = new Float32Array(V);
  const at = (plane, cx, cy) => plane * G + cy * GW + cx;
  for (const u of m.units) {
    const [cx, cy] = worldToCell(p, u.x, u.y);
    sp[at(TROOP_PLANE[u.card] + (u.owner === p ? 0 : NT), cx, cy)] += u.hp / 1000.0;
  }
  for (const owner of [0, 1]) {
    const plane = 2 * NT + (owner === p ? 0 : 1);
    for (const t of m.towers[owner]) {
      const [cx, cy] = worldToCell(p, t.x, t.y);
      sp[at(plane, cx, cy)] = t.alive ? t.hp / t.max_hp : -1.0;
    }
  }
  for (const s of m.spells) {
    const [cx, cy] = worldToCell(p, s.x, s.y);
    sp[at(2 * NT + 2 + (s.owner === p ? 0 : 1), cx, cy)] = 1.0;
  }
  const q = 1 - p;
  let i = 0;
  vec[i++] = m.elixir[p] / 10.0;
  const hand = m.hand(p);
  hand.forEach((cid, s) => { vec[i + s * NUM_CARDS + cid] = 1.0; });
  i += 4 * NUM_CARDS;
  vec[i + m.next_card(p)] = 1.0;
  i += NUM_CARDS;
  hand.forEach((cid, s) => { vec[i + s] = m.elixir[p] >= CARDS[cid].cost ? 1 : 0; });
  i += 4;
  vec[i++] = m.time_left() / MATCH_SECONDS;
  vec[i++] = m.double_elixir() ? 1 : 0;
  vec[i] = m.crowns(p) / 3.0; vec[i + 1] = m.crowns(q) / 3.0; i += 2;
  vec[i] = m.towers[p][0].active ? 1 : 0; vec[i + 1] = m.towers[q][0].active ? 1 : 0; i += 2;
  [...m.towers[p], ...m.towers[q]].forEach((t, k) => { vec[i + k] = t.hp / t.max_hp; });
  i += 6;
  if (i !== V) throw new Error('observation size mismatch');
  return [sp, vec];
}

// ------------------------------------------------------------------ bots (game/bots.py)
export class RandomBot {
  constructor(seed = 1, playProb = 0.15) { this.r = rng(seed); this.playProb = playProb; this.name = 'Random Bot'; this.kind = 'bot'; }
  reset() {}
  act(m, p) {
    const mask = actionMask(m, p);
    mask[0] = 0;
    const legal = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
    if (!legal.length || this.r() > this.playProb) return 0;
    return legal[Math.floor(this.r() * legal.length)];
  }
}

const ANTI_AIR = ['Musketeer', 'Archers', 'Minions'];
const ANTI_TANK = ['MiniPekka', 'Knight', 'Minions'];
const GENERAL = ['Knight', 'Musketeer', 'MiniPekka', 'Archers', 'Goblins', 'Minions'];
const ownY = (p, y) => (p === 0 ? y : HEIGHT - y);

export class ScriptedBot {
  constructor() { this.name = 'Scripted Bot'; this.kind = 'bot'; }
  reset() {}
  _action(p, slot, x, oy) { const [cx, cy] = worldToCell(p, x, ownY(p, oy)); return encodeAction(slot, cx, cy); }
  _slotFor(m, p, names) {
    const hand = m.hand(p);
    for (const n of names) {
      const cid = CARD_INDEX[n];
      if (hand.includes(cid) && m.elixir[p] >= CARDS[cid].cost) return hand.indexOf(cid);
    }
    return null;
  }
  act(m, p) {
    const enemy = 1 - p;
    const threats = m.units.filter((u) => u.owner === enemy && u.deploy <= 0 && ownY(p, u.y) < 17.5);
    if (threats.length) {
      const fb = this._slotFor(m, p, ['Fireball']);
      if (fb !== null) {
        for (const u of threats) {
          const near = threats.filter((o) => Math.hypot(o.x - u.x, o.y - u.y) < 2.5);
          if (near.length >= 3) {
            const x = near.reduce((s, o) => s + o.x, 0) / near.length;
            const y = near.reduce((s, o) => s + ownY(p, o.y), 0) / near.length;
            return this._action(p, fb, x, y);
          }
        }
      }
      let t = threats[0];
      for (const u of threats) if (ownY(p, u.y) < ownY(p, t.y)) t = u;
      const defenders = m.units.filter((u) => u.owner === p && Math.hypot(u.x - t.x, u.y - t.y) < 5 && !CARDS[u.card].buildings_only);
      if (defenders.length < 2) {
        const tc = CARDS[t.card];
        const prefs = tc.is_air ? ANTI_AIR : tc.hp > 1000 ? ANTI_TANK : GENERAL;
        const slot = this._slotFor(m, p, prefs);
        if (slot !== null) return this._action(p, slot, t.x, Math.max(1, Math.min(13, ownY(p, t.y) - 3)));
      }
      return 0;
    }
    if (m.elixir[p] >= 9.5) {
      const et = m.towers[enemy];
      const left = et[1].alive ? et[1].hp : 0, right = et[2].alive ? et[2].hp : 0;
      const bx = left <= right ? BRIDGES[0] : BRIDGES[1];
      let slot = this._slotFor(m, p, ['Giant']);
      if (slot !== null) return this._action(p, slot, bx, 12.5);
      const support = m.units.filter((u) => u.owner === p && CARDS[u.card].buildings_only);
      slot = this._slotFor(m, p, GENERAL);
      if (slot !== null) return this._action(p, slot, support.length ? support[0].x : bx, support.length ? 9.0 : 12.5);
    }
    return 0;
  }
}
