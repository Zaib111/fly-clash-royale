import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { buildArena, toLocal } from './arena.js';
import { makeFly, makePortrait, makeRubble, makeTower, makeUnit, TEAM } from './models.js';
import { Effects } from './fx.js';
import { BrainView, GROUP_COLORS } from './brain.js';
import { connect } from './transport.js';

const $ = (id) => document.getElementById(id);
const transport = await connect();     // Python server if running, otherwise the in-browser engine
const info = transport.info;
const A = info.arena;
const CARDS = info.cards;
const VIEW = info.view;            // seat shown at the bottom (always blue)
const HUMAN = info.human;
const DT = A.dt;
const BUFFER = 2.2 * DT;           // play back this far behind the newest tick, in game seconds
const teamOf = (seat) => (seat === VIEW ? 0 : 1);
const flySeats = Object.keys(info.brains).map(Number);

// ------------------------------------------------------------------ renderer & scene
const canvas = $('arena');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa5d4ee);
scene.fog = new THREE.Fog(0xa5d4ee, 75, 160);
const camera = new THREE.PerspectiveCamera(34, 1, 0.5, 400);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.maxPolarAngle = 1.32;
controls.minDistance = 12;
controls.maxDistance = 95;

scene.add(new THREE.HemisphereLight(0xe6f2ff, 0x56743a, 1.15));
const sun = new THREE.DirectionalLight(0xfff1dc, 2.7);
sun.position.set(-14, 34, 18);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -26, right: 26, top: 26, bottom: -26, near: 1, far: 100 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
scene.add(sun);

const world = new THREE.Group();
if (VIEW === 1) world.rotation.y = Math.PI;
scene.add(world);
const arena = buildArena(world, A, renderer);
const fx = new Effects(world, A);

function fitCamera() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const yTop = 1 - (2 * 62) / h, yBot = -1 + (2 * 132) / h;
  const pitch = 1.0;
  const far = [new THREE.Vector3(-9.4, 4.0, -16.4), new THREE.Vector3(9.4, 4.0, -16.4)];
  const near = [new THREE.Vector3(-9.4, 0, 16.4), new THREE.Vector3(9.4, 0, 16.4)];
  let d = 50, tz = 0;
  for (let i = 0; i < 60; i++) {
    camera.position.set(0, d * Math.sin(pitch), tz + d * Math.cos(pitch));
    camera.lookAt(0, 0, tz);
    camera.updateMatrixWorld();
    const pf = far.map((p) => p.clone().project(camera));
    const pn = near.map((p) => p.clone().project(camera));
    const top = Math.max(...pf.map((p) => p.y)), bot = Math.min(...pn.map((p) => p.y));
    const xmax = Math.max(...pn.map((p) => Math.abs(p.x)), ...pf.map((p) => Math.abs(p.x)));
    const scale = Math.max((top - bot) / (yTop - yBot), xmax / 0.97);
    d *= Math.pow(scale, 0.6);
    tz -= ((top + bot) / 2 - (yTop + yBot) / 2) * d * 0.15;
  }
  controls.target.set(0, 0, tz);
  controls.update();
}
new ResizeObserver(fitCamera).observe(canvas);
fitCamera();
canvas.addEventListener('dblclick', fitCamera);

// ------------------------------------------------------------------ static UI
const portraits = Object.fromEntries(CARDS.map((c) => [c.name, makePortrait(c.name)]));
const CARD_BG = {
  Knight: ['#7ea2e0', '#36579a'], Archers: ['#8fcf7a', '#3f7a35'], Giant: ['#e3a96a', '#8a5426'],
  Goblins: ['#8fd86a', '#3e7d2c'], Musketeer: ['#b38be0', '#5a3290'], MiniPekka: ['#7f97d6', '#283c78'],
  Fireball: ['#ffb36b', '#b0451b'], Minions: ['#8aa0f2', '#38429c'],
};
function cardHTML(c) {
  const [a, b] = CARD_BG[c.name] || ['#888', '#444'];
  return `<div class="inner" style="background: radial-gradient(circle at 50% 35%, ${a}, ${b})">
    <img src="${portraits[c.name]}" alt=""><div class="name">${c.name === 'MiniPekka' ? 'Mini P.E.K.K.A' : c.name}</div>
    <div class="cover"></div></div><div class="cost"><b>${c.cost}</b></div>`;
}
const handEls = [0, 1, 2, 3].map((i) => {
  const el = document.createElement('div');
  el.className = 'card';
  el.dataset.slot = i;
  el.addEventListener('click', () => selectSlot(i));
  $('hand').appendChild(el);
  return el;
});
for (let i = 0; i < 10; i++) {
  const s = document.createElement('div');
  s.className = 'seg';
  s.appendChild(document.createElement('i'));
  $('elixir-bar').appendChild(s);
}
const segs = [...$('elixir-bar').children].map((s) => s.firstChild);
$('opp-name').textContent = info.players[1 - VIEW].name;
$('me-name').textContent = info.players[VIEW].name;
const vsFly = flySeats.length > 0;
if (vsFly) {
  const b = info.brains[flySeats[0]];
  $('subtitle').textContent = `${b.neurons.toLocaleString()} neurons · ${(b.connections / 1e6).toFixed(1)}M connections · ` +
    `${(b.synapses / 1e6).toFixed(1)}M synapses · adult Drosophila brain (FlyWire v783)`;
} else {
  $('subtitle').textContent = `${info.players[0].name} vs ${info.players[1].name} — no fly in this match`;
}
setTimeout(() => { $('hint').style.opacity = 0; }, 7000);

// ------------------------------------------------------------------ brain panel + fly avatar
let brainView = null;
const brainSeat = flySeats.length ? flySeats[0] : null;
if (brainSeat !== null) {
  $('brain-section').classList.remove('hidden');
  document.querySelector('[data-tab="decision"]').classList.remove('hidden');
  const buf = await transport.brainBuffer(brainSeat);
  brainView = new BrainView($('brain'), buf, info.brains[brainSeat].points);
  const toggleBig = () => { document.body.classList.toggle('brain-big'); fitCamera(); };
  $('brain-expand').addEventListener('click', toggleBig);
  window.addEventListener('keydown', (e) => { if (e.key === 'b' || e.key === 'B') toggleBig(); });
  $('legend').innerHTML = info.brains[brainSeat].groups.map((g, i) => {
    const c = GROUP_COLORS[i].map((v) => Math.round(v * 255)).join(',');
    return `<div><span style="background: rgb(${c})"></span>${g}</div>`;
  }).join('');
}
fitCamera();

const avatars = new Map();
for (const seat of flySeats) {
  const fly = makeFly();
  fly.group.scale.setScalar(2.3);
  // hover behind its own king tower; if it's the bottom player, off to the side so it doesn't block the view
  const kingY = seat === 0 ? 3 : A.h - 3;
  // hover just in front of its own king tower, facing the battle
  const front = seat === 0 ? 3.6 : -3.6;
  const home = seat === VIEW ? toLocal(A, A.w + 3.6, kingY + front, 5.5)
    : toLocal(A, A.w / 2, kingY + front, 4.6);
  fly.group.position.copy(home);
  world.add(fly.group);
  avatars.set(seat, { ...fly, home, swoop: null, yaw: seat === 0 ? Math.PI : 0 });
}

// ------------------------------------------------------------------ towers & units
let towers = [];
function buildTowers(snap) {
  for (const t of towers) { world.remove(t.model.group); if (t.rubble) world.remove(t.rubble); }
  towers = snap.towers.map(([owner, isKing, x, y]) => {
    const model = makeTower(!!isKing, teamOf(owner));
    model.group.position.copy(toLocal(A, x, y));
    model.group.rotation.y = owner === 0 ? 0 : Math.PI;
    world.add(model.group);
    return { model, owner, isKing: !!isKing, x, y, rubble: null, hp: null, flashT: -9, aim: null };
  });
}

const units = new Map();
function unitScale(c) { return Math.max(1.5, (c.radius / 0.5) * 2.0); }
function getUnit(uid, card, owner) {
  let u = units.get(uid);
  if (!u) {
    const c = CARDS[card];
    const m = makeUnit(c.name, teamOf(owner));
    m.group.scale.setScalar(unitScale(c));
    world.add(m.group);
    u = { m, card, owner, c, yaw: owner === 0 ? Math.PI : 0, lastAttack: -9, flashT: -9, hp: null, last: null };
    units.set(uid, u);
  }
  return u;
}
const WHITE = new THREE.Color(1, 1, 1);
// Briefly whiten an object when it takes damage. `obj.flashing` tracks whether it needs resetting.
function flash(obj, mats, k) {
  if (k <= 0 && !obj.flashing) return;
  for (const m of mats) {
    if (!m.emissive) continue;
    if (!m.userData.e0) m.userData.e0 = m.emissive.clone();
    m.emissive.copy(m.userData.e0).lerp(WHITE, k);
  }
  obj.flashing = k > 0;
}
const lerpAngle = (a, b, k) => a + (((b - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI) * k;

// ------------------------------------------------------------------ snapshots & playback clock
let snaps = [];
let pending = [];
let matchId = null;
let renderT = 0;
let latest = null;
let lastThinking = null;

transport.onSnapshot((s) => {
  if (s.match !== matchId) resetMatch(s);
  latest = s;
  if (!snaps.length || s.tick > snaps[snaps.length - 1].tick) snaps.push(s);
  if (snaps.length > 80) snaps.splice(0, snaps.length - 80);
  for (const ev of s.events) pending.push(ev);
  for (const [seat, fr] of Object.entries(s.brains || {})) {
    if (brainView && Number(seat) === brainSeat) {
      brainView.setActivity(fr.act);
      updateStats(fr, s);
      if (fr.vision) renderVision(fr);
      if (fr.slots) renderFlyHand(fr);
      flashState('deciding', 'warn');
    }
  }
  if (s.thinking && brainSeat !== null && s.thinking[brainSeat] && s.thinking[brainSeat] !== lastThinking) {
    lastThinking = s.thinking[brainSeat];
    renderThinking(lastThinking);
  }
}, () => { $('status').textContent = 'disconnected'; $('status').className = 'pill'; });

function resetMatch(s) {
  matchId = s.match;
  snaps = [];
  pending = [];
  for (const u of units.values()) world.remove(u.m.group);
  units.clear();
  fx.clear();
  buildTowers(s);
  renderT = s.t;
  $('result').classList.add('hidden');
  for (const av of avatars.values()) av.swoop = null;
  clearMoves();
  updateWinbar(0.5);
}

function sample(t) {
  let i = snaps.length - 1;
  while (i > 0 && snaps[i].t > t) i--;
  const a = snaps[i], b = snaps[Math.min(i + 1, snaps.length - 1)];
  const span = b.t - a.t;
  const k = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 1;
  return { a, b, k };
}

// ------------------------------------------------------------------ events
function towerAt(x, y) {
  return towers.find((t) => Math.abs(t.x - x) < 0.01 && Math.abs(t.y - y) < 0.01);
}
function handleEvent(ev) {
  const [t, kind] = ev;
  if (kind === 'attack') {
    const [, , card, owner, x, y, tx, ty, uid] = ev;
    const c = CARDS[card];
    const u = units.get(uid);
    if (u) u.lastAttack = t;
    const hitH = towerAt(tx, ty) ? 2.0 : 0.7;
    if (c.range > 1.5) {
      const style = { Archers: 'arrow', Musketeer: 'bullet', Minions: 'orb' }[c.name] || 'orb';
      fx.projectile(t, style, x, y, c.is_air ? 1.9 : 0.8, tx, ty, hitH);
    } else {
      fx.slash(t + 0.12, tx, ty, c.name === 'MiniPekka' || c.name === 'Giant');
    }
  } else if (kind === 'tower_attack') {
    const [, , isKing, owner, x, y, tx, ty] = ev;
    const tw = towerAt(x, y);
    if (tw) tw.aim = { x: tx, y: ty };
    fx.projectile(t, isKing ? 'cannon' : 'arrow', x, y, isKing ? 3.6 : 3.2, tx, ty, 0.7);
  } else if (kind === 'cast') {
    const [, , card, p, x, y] = ev;
    const king = towers.find((tw) => tw.owner === p && tw.isKing);
    fx.fireball(t, 0.6, king.x, king.y, x, y);
  } else if (kind === 'boom') {
    const [, , card, x, y] = ev;
    fx.explosion(t, x, y, CARDS[card].spell_radius);
  } else if (kind === 'death') {
    const [, , isBuilding, owner, x, y] = ev;
    if (isBuilding) fx.explosion(t, x, y, 2.4, true);
    else fx.puff(t, x, y);
  } else if (kind === 'play') {
    const [, , card, p, x, y] = ev;
    const c = CARDS[card];
    if (c.kind === 'troop') fx.deployRing(t, x, y, 1.0, TEAM[teamOf(p)]);
    if (p !== HUMAN) popCard(c, p, x, y);
    logMove(t, card, p, x, y);
    const av = avatars.get(p);
    if (av) {
      av.swoop = { t0: performance.now() / 1000, to: toLocal(A, x, y, 3.4), card: cardSprite(c.name) };
      world.add(av.swoop.card);
      if (p === brainSeat) flashState(`played ${c.name}`, 'live', 1500);
    }
  }
}

// ------------------------------------------------------------------ labels (HP bars, Zz, card pops)
const labelPool = new Map();
const used = new Set();
const v3 = new THREE.Vector3();
function project(local) {
  v3.copy(local);
  world.localToWorld(v3);
  v3.project(camera);
  return [(v3.x + 1) / 2 * canvas.clientWidth, (1 - v3.y) / 2 * canvas.clientHeight, v3.z < 1];
}
function label(key, cls, html) {
  let el = labelPool.get(key);
  if (!el) {
    el = document.createElement('div');
    $('labels').appendChild(el);
    labelPool.set(key, el);
  }
  if (el.className !== cls) el.className = cls;
  if (html !== undefined && el._html !== html) { el.innerHTML = html; el._html = html; }
  el.style.display = '';
  used.add(key);
  return el;
}
function hpBar(key, local, frac, team, width, number) {
  const [sx, sy, ok] = project(local);
  if (!ok) return;
  const el = label(key, `hp${team ? ' red' : ''}${number !== undefined ? ' big' : ''}`,
    `<i></i>${number !== undefined ? '<b></b>' : ''}`);
  el.style.left = `${sx}px`;
  el.style.top = `${sy}px`;
  el.style.width = `${width}px`;
  el.firstChild.style.width = `${Math.max(0, frac) * 100}%`;
  if (number !== undefined) el.lastChild.textContent = number;
}
function popCard(c, p, x, y) {
  const [sx, sy] = project(toLocal(A, x, y, 1.5));
  const el = document.createElement('div');
  el.className = `popcard${teamOf(p) ? ' red' : ''}`;
  el.innerHTML = `<img src="${portraits[c.name]}" alt="">`;
  el.style.left = `${sx}px`;
  el.style.top = `${sy}px`;
  $('labels').appendChild(el);
  setTimeout(() => el.remove(), 1400);
}

// ------------------------------------------------------------------ tabs, moves log, who's-winning bar
function showTab(name) {
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
}
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
showTab(brainSeat !== null ? 'decision' : 'moves');

function whereDesc(p, x, y) {
  const own = p === 0 ? y : A.h - y;                       // distance from that player's back wall
  const lane = x < A.w / 3 ? 'left' : x < (2 * A.w) / 3 ? 'middle' : 'right';
  const depth = own < 6 ? 'back' : own < 11 ? 'mid' : own < A.troop_limit ? 'bridge' : own < A.river[1] ? 'river'
    : own < A.pocket_end ? 'enemy side' : 'enemy king';
  return `${lane}, ${depth}`;
}
function logMove(t, card, p, x, y) {
  const list = $('moves');
  const empty = list.querySelector('.empty');
  if (empty) empty.remove();
  const c = CARDS[card];
  const li = document.createElement('li');
  li.className = 'new';
  const mm = Math.floor(t / 60), ss = String(Math.floor(t % 60)).padStart(2, '0');
  const time = document.createElement('span');
  time.className = 't';
  time.textContent = `${mm}:${ss}`;
  const dot = document.createElement('i');
  dot.className = 'dot';
  dot.style.background = teamOf(p) ? 'var(--red)' : 'var(--blue)';
  const what = document.createElement('span');
  const img = document.createElement('img');
  img.src = portraits[c.name];
  img.alt = '';
  what.append(img, ` ${c.name === 'MiniPekka' ? 'Mini P.E.K.K.A' : c.name}`);
  what.style.display = 'flex';
  what.style.alignItems = 'center';
  what.style.gap = '6px';
  const who = document.createElement('span');
  who.className = 'who';
  who.textContent = `${info.players[p].name} · ${whereDesc(p, x, y)}`;
  li.append(time, dot, what, who);
  list.prepend(li);
  while (list.children.length > 120) list.lastChild.remove();
}
function clearMoves() { $('moves').innerHTML = '<li class="empty">No cards played yet.</li>'; }

if (info.has_value) $('winbar').classList.remove('hidden');
function updateWinbar(p) {
  if (p === null || p === undefined) return;
  $('winbar-fill').style.height = `${Math.round(p * 100)}%`;
  $('winbar-bot-n').textContent = `${Math.round(p * 100)}%`;
  $('winbar-top-n').textContent = `${Math.round((1 - p) * 100)}%`;
}

// ------------------------------------------------------------------ the fly's view and card pick
const textures = {};
function cardSprite(name) {
  if (!textures[name]) {
    textures[name] = new THREE.TextureLoader().load(portraits[name]);
    textures[name].colorSpace = THREE.SRGBColorSpace;
  }
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: textures[name], transparent: true, depthWrite: false }));
  s.scale.set(1.9, 2.3, 1);
  return s;
}

const visionCtx = $('vision').getContext('2d');
function renderVision(fr) {
  const g = visionCtx, W = 180, H = 320, cw = W / 9, ch = H / 16;
  g.setTransform(2, 0, 0, 2, 0, 0);   // canvas is 2x for sharpness
  const bin = atob(fr.vision);
  const at = (plane, cy, cx) => bin.charCodeAt(plane * 144 + cy * 9 + cx) / 255;
  g.fillStyle = '#07070c';
  g.fillRect(0, 0, W, H);
  g.fillStyle = 'rgba(60, 140, 220, 0.18)';          // the river, rows 7-8
  g.fillRect(0, H - 9 * ch, W, 2 * ch);
  const maxIntent = Math.max(...fr.intent);
  for (let cy = 0; cy < 16; cy++) {
    for (let cx = 0; cx < 9; cx++) {
      const x = cx * cw + cw / 2, y = H - (cy * ch + ch / 2);   // its own side at the bottom
      const own = at(0, cy, cx), enemy = at(1, cy, cx), myTower = at(2, cy, cx), theirTower = at(3, cy, cx), spell = at(4, cy, cx);
      const intent = maxIntent > 0.002 ? fr.intent[cy * 9 + cx] / maxIntent : 0;
      if (intent > 0.05) {
        const grd = g.createRadialGradient(x, y, 0, x, y, cw * 0.9);
        grd.addColorStop(0, `rgba(255, 190, 90, ${0.75 * intent})`);
        grd.addColorStop(1, 'rgba(255, 190, 90, 0)');
        g.fillStyle = grd;
        g.fillRect(x - cw, y - ch, cw * 2, ch * 2);
      }
      g.beginPath();                                   // one "ommatidium" per cell
      g.arc(x, y, cw * 0.36, 0, Math.PI * 2);
      g.fillStyle = '#1b1e2b';
      g.fill();
      if (myTower > 0 || theirTower > 0) {
        g.fillStyle = myTower > 0 ? `rgba(71, 198, 255, ${0.25 + 0.5 * myTower})` : `rgba(255, 122, 69, ${0.25 + 0.5 * theirTower})`;
        g.fillRect(x - cw * 0.42, y - ch * 0.42, cw * 0.84, ch * 0.84);
      }
      for (const [v, col] of [[own, '71, 198, 255'], [enemy, '255, 122, 69']]) {
        if (v <= 0) continue;
        g.beginPath();
        g.arc(x, y, cw * (0.2 + 0.2 * Math.min(1, v * 2)), 0, Math.PI * 2);
        g.fillStyle = `rgba(${col}, ${0.55 + 0.45 * Math.min(1, v * 2)})`;
        g.fill();
      }
      if (spell > 0) {
        g.beginPath();
        g.arc(x, y, cw * 0.45, 0, Math.PI * 2);
        g.strokeStyle = '#ffe066';
        g.lineWidth = 2;
        g.stroke();
      }
    }
  }
  if (fr.chosen) {
    const cell = fr.chosen[1], cx = cell % 9, cy = Math.floor(cell / 9);
    g.beginPath();
    g.arc(cx * cw + cw / 2, H - (cy * ch + ch / 2), cw * 0.55, 0, Math.PI * 2);
    g.strokeStyle = '#ffd27a';
    g.lineWidth = 3;
    g.stroke();
  }
}

const flyHand = [0, 1, 2, 3].map(() => {
  const el = document.createElement('div');
  el.className = 'pick';
  el.innerHTML = '<div class="bar"><i></i></div><img alt=""><span></span><em></em>';
  $('fly-hand').appendChild(el);
  return el;
});
function renderFlyHand(fr) {
  fr.hand.forEach((cid, i) => {
    const c = CARDS[cid], el = flyHand[i];
    const img = el.querySelector('img');
    if (img.dataset.card !== c.name) { img.src = portraits[c.name]; img.dataset.card = c.name; }
    el.querySelector('span').textContent = `${c.name === 'MiniPekka' ? 'Mini P.E.K.K.A' : c.name} · ${c.cost}`;
    el.querySelector('em').textContent = `${(fr.slots[i] * 100).toFixed(0)}%`;
    el.querySelector('.bar i').style.width = `${fr.slots[i] * 100}%`;
    el.classList.toggle('chosen', !!fr.chosen && fr.chosen[0] === i);
    el.classList.toggle('broke', fr.elixir < c.cost);
  });
  const w = $('fly-wait');
  w.querySelector('em').textContent = `${(fr.wait * 100).toFixed(0)}%`;
  w.querySelector('.bar i').style.width = `${fr.wait * 100}%`;
  w.classList.toggle('chosen', !fr.chosen);
  $('fly-elixir').innerHTML = `elixir <b>${fr.elixir.toFixed(1)}</b> / 10`;
}

// ------------------------------------------------------------------ fly panel text
function updateStats(fr, s) {
  const b = info.brains[brainSeat];
  const cells = [
    [b.neurons.toLocaleString(), 'neurons'],
    [`${(b.connections / 1e6).toFixed(1)}M`, 'connections'],
    [s.decisions.toLocaleString(), 'decisions'],
    [fr.active.toLocaleString(), 'active now'],
    [fr.active_out.toLocaleString(), 'motor active'],
    [`${Math.floor(s.t / 60)}:${String(Math.floor(s.t % 60)).padStart(2, '0')}`, 'sim time'],
  ];
  $('stats').innerHTML = cells.map(([v, k]) => `<div><b>${v}</b><small>${k}</small></div>`).join('');
}
function renderThinking(list) {
  $('think-list').innerHTML = list.map(([desc, p]) =>
    `<li><i style="width:${Math.round(p * 100)}%"></i><span>${desc}<em>${(p * 100).toFixed(0)}%</em></span></li>`).join('');
}
let stateTimer = null;
function flashState(text, cls, ms = 220) {
  const el = $('fly-state');
  el.textContent = text;
  el.className = `pill small ${cls}`;
  clearTimeout(stateTimer);
  stateTimer = setTimeout(() => { el.textContent = 'watching'; el.className = 'pill small'; }, ms);
}

// ------------------------------------------------------------------ human input
let selected = null;
const canPlay = HUMAN !== null && HUMAN === VIEW;
const ghosts = {};
// red "can't deploy" tint: enemy side minus any pocket the human has opened
const zone = new THREE.Group();
const zoneMat = new THREE.MeshBasicMaterial({ color: 0xff3030, transparent: true, opacity: 0.16, depthWrite: false });
const ownY = (y) => (HUMAN === 1 ? A.h - y : y);   // own-perspective y -> world y
function zonePiece(x0, x1, y0, y1) {               // own-perspective box
  const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), zoneMat);
  m.rotation.x = -Math.PI / 2;
  m.position.copy(toLocal(A, (x0 + x1) / 2, ownY((y0 + y1) / 2), 0.07));
  zone.add(m);
  return m;
}
zonePiece(0, A.w, A.troop_limit, A.river[1]);
zonePiece(0, A.w, A.pocket_end, A.h);
const pocketPieces = [zonePiece(0, A.w / 2, A.river[1], A.pocket_end), zonePiece(A.w / 2, A.w, A.river[1], A.pocket_end)];
zone.visible = false;
world.add(zone);
const spellRing = new THREE.Mesh(new THREE.RingGeometry(0.93, 1, 48),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, depthWrite: false }));
spellRing.rotation.x = -Math.PI / 2;
spellRing.visible = false;
world.add(spellRing);
let hover = null;

function selectSlot(i) {
  if (!canPlay) return;
  selected = selected === i ? null : i;
}
function legal(c, x, y) {
  if (x < 0 || x >= A.w || y < 0 || y >= A.h) return false;
  if (c.kind === 'spell') return true;
  const own = HUMAN === 0 ? y : A.h - y;
  if (own < A.troop_limit) return true;
  const open = latest && latest.pockets ? latest.pockets[HUMAN] : [false, false];
  return own >= A.river[1] && own < A.pocket_end && (x < A.w / 2 ? open[0] : open[1]);
}
const ray = new THREE.Raycaster();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
function pick(ev) {
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const p = new THREE.Vector3();
  if (!ray.ray.intersectPlane(groundPlane, p)) return null;
  world.worldToLocal(p);
  return { x: p.x + A.w / 2, y: A.h / 2 - p.z };
}
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointermove', (e) => { hover = pick(e); });
canvas.addEventListener('pointerleave', () => { hover = null; });
canvas.addEventListener('pointerup', async (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
  if (!canPlay || selected === null || !latest) return;
  const p = pick(e);
  if (!p) return;
  const slot = selected;
  const res = await transport.play(slot, p.x, p.y);
  if (res.ok) selected = null;
  else {
    handEls[slot].classList.remove('denied');
    void handEls[slot].offsetWidth;
    handEls[slot].classList.add('denied');
  }
});
function control(cmd) {
  if (!latest) return;
  if (cmd === 'faster') transport.control('speed', latest.speed * 2);
  else if (cmd === 'slower') transport.control('speed', latest.speed / 2);
  else transport.control(cmd);
}
document.querySelectorAll('#controls button').forEach((b) => b.addEventListener('click', () => control(b.dataset.cmd)));
$('rematch').addEventListener('click', () => control('restart'));
window.addEventListener('keydown', (e) => {
  if (e.key >= '1' && e.key <= '4') selectSlot(Number(e.key) - 1);
  else if (e.key === 'Escape') selected = null;
  else if (e.key === ' ') { e.preventDefault(); control('pause'); }
  else if (e.key === '+' || e.key === '=') control('faster');
  else if (e.key === '-') control('slower');
  else if (e.key === 'r' || e.key === 'R') control('restart');
});

function updateGhost(hand, elixir) {
  for (const g of Object.values(ghosts)) g.group.visible = false;
  spellRing.visible = false;
  zone.visible = false;
  if (!canPlay || selected === null) return;
  const c = CARDS[hand[selected]];
  zone.visible = c.kind === 'troop';
  const open = latest && latest.pockets ? latest.pockets[HUMAN] : [false, false];
  pocketPieces.forEach((p, i) => { p.visible = !open[i]; });
  if (!hover) return;
  const ok = legal(c, hover.x, hover.y) && elixir >= c.cost;
  if (c.kind === 'spell') {
    spellRing.visible = true;
    spellRing.position.copy(toLocal(A, hover.x, hover.y, 0.08));
    spellRing.scale.setScalar(c.spell_radius);
    spellRing.material.color.set(ok ? 0xffffff : 0xff5050);
    return;
  }
  if (!ghosts[c.name]) {
    ghosts[c.name] = makeUnit(c.name, 0, { ghost: true });
    ghosts[c.name].group.scale.setScalar(unitScale(c));
    world.add(ghosts[c.name].group);
  }
  const g = ghosts[c.name];
  g.group.visible = true;
  g.group.position.copy(toLocal(A, hover.x, hover.y, c.is_air ? 1.8 : 0));
  g.group.rotation.y = HUMAN === 0 ? Math.PI : 0;
  for (const m of g.mats) m.opacity = ok ? 0.55 : 0.2;
}

// ------------------------------------------------------------------ per-frame updates
let lastHand = [];
let lastNext = null;
function updateHUD(a, b, k) {
  const me = VIEW;
  const el = a.elixir[me] + (b.elixir[me] - a.elixir[me]) * k;
  const hand = b.hand[me];
  hand.forEach((cid, i) => {
    const c = CARDS[cid];
    const e = handEls[i];
    if (lastHand[i] !== cid) e.innerHTML = cardHTML(c);
    e.querySelector('.cover').style.height = `${(1 - Math.min(1, el / c.cost)) * 100}%`;
    e.classList.toggle('selected', canPlay && selected === i);
  });
  lastHand = hand.slice();
  if (lastNext !== b.next[me]) { $('next-card').innerHTML = cardHTML(CARDS[b.next[me]]); lastNext = b.next[me]; }
  segs.forEach((s, i) => { s.style.width = `${Math.max(0, Math.min(1, el - i)) * 100}%`; });
  $('elixir-n').textContent = Math.floor(el);
  const left = Math.max(0, b.time_left - (renderT - b.t));
  const secs = Math.ceil(left);
  $('time').textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`;
  $('timer').classList.toggle('double', b.double);
  $('x2').classList.toggle('hidden', !b.double);
  const ct = b.crowns[1 - me], cb = b.crowns[me];
  $('crowns-top').textContent = ct;
  $('crowns-bottom').textContent = cb;
  $('crowns-top').parentElement.classList.toggle('has', ct > 0);
  $('crowns-bottom').parentElement.classList.toggle('has', cb > 0);
  updateGhost(hand, el);
}

function updateTowers(b, now) {
  b.towers.forEach(([owner, isKing, x, y, hp, maxHp, active, alive], i) => {
    const tw = towers[i];
    if (!tw) return;
    if (tw.hp !== null && hp < tw.hp) tw.flashT = now;
    tw.hp = hp;
    const g = tw.model.group;
    if (!alive) {
      if (!tw.rubble) {
        tw.rubble = makeRubble(tw.isKing, teamOf(owner));
        tw.rubble.position.copy(g.position);
        world.add(tw.rubble);
      }
      g.visible = false;
      return;
    }
    g.visible = true;
    const fl = Math.max(0, 1 - (now - tw.flashT) / 0.12);
    flash(tw, tw.model.mats, fl * 0.3);
    const P = tw.model.parts;
    if (tw.isKing) {
      P.eyesOpen.visible = !!active;
      P.eyesClosed.visible = !active;
    }
    if (tw.aim && (active || !tw.isKing)) {
      const target = toLocal(A, tw.aim.x, tw.aim.y);
      g.worldToLocal(world.localToWorld(target));
      const want = Math.atan2(target.x, target.z) + Math.PI;
      P.aim.rotation.y = lerpAngle(P.aim.rotation.y, want, 0.2);
    }
    const team = teamOf(owner);
    const half = tw.isKing ? 2.0 : 1.5;
    hpBar(`tw${i}`, toLocal(A, x, y, tw.model.height + 0.6), hp / maxHp, team, tw.isKing ? 88 : 70, hp);
    if (tw.isKing && !active) {
      const [sx, sy] = project(toLocal(A, x + half * 0.6, y, tw.model.height + 1.6));
      const el = label(`zz${i}`, 'zz', 'z<small>z</small>');
      el.style.left = `${sx}px`;
      el.style.top = `${sy}px`;
    }
  });
}

function updateUnits(a, b, k, now, dt) {
  const ra = new Map(a.units.map((r) => [r[0], r]));
  const rb = new Map(b.units.map((r) => [r[0], r]));
  const alive = new Set([...ra.keys(), ...rb.keys()]);
  for (const [uid, u] of units) {
    if (!alive.has(uid)) { world.remove(u.m.group); units.delete(uid); }
  }
  for (const uid of alive) {
    const r0 = ra.get(uid), r1 = rb.get(uid);
    const r = r1 || r0;
    const [, card, owner, , , hp, maxHp, deploy] = r;
    const u = getUnit(uid, card, owner);
    let x, y;
    if (r0 && r1) { x = r0[3] + (r1[3] - r0[3]) * k; y = r0[4] + (r1[4] - r0[4]) * k; }
    else { x = r[3]; y = r[4]; }
    if (!r1) { u.m.group.visible = k < 0.5; }   // died during this tick
    const c = u.c;
    const g = u.m.group;
    const P = u.m.parts;
    const air = c.is_air;
    const pos = toLocal(A, x, y, 0);
    const moved = u.last ? Math.hypot(pos.x - u.last.x, pos.z - u.last.z) / Math.max(dt, 1e-3) : 0;
    u.last = pos.clone();
    // facing: target, else direction of travel, else toward the enemy
    let want = u.yaw;
    if (r[8] !== null && r[8] !== undefined) {
      const tp = toLocal(A, r[8], r[9]);
      want = Math.atan2(tp.x - pos.x, tp.z - pos.z);
    } else if (moved > 0.2 && u.prevPos) {
      want = Math.atan2(pos.x - u.prevPos.x, pos.z - u.prevPos.z);
    }
    u.prevPos = pos.clone();
    u.yaw = lerpAngle(u.yaw, want, Math.min(1, dt * 10));
    g.rotation.y = u.yaw;
    const wt = now + (uid % 7) * 0.37;
    const walking = moved > 0.15 && deploy <= 0;
    let h = air ? 1.8 + Math.sin(wt * 3) * 0.12 : 0;
    if (walking && !air) h += Math.abs(Math.sin(wt * 10)) * 0.06;
    // deploy: drop in and pop to size
    const dp = Math.max(0, deploy);
    const born = 1 - Math.min(1, dp);
    if (dp > 0) h += Math.max(0, 1 - born * 3) * 2.5;
    g.scale.setScalar(unitScale(c) * (dp > 0 ? 0.75 + 0.25 * born : 1));
    pos.y = h;
    g.position.copy(pos);
    if (P.legs) P.legs.forEach((l, i) => { l.rotation.x = walking ? Math.sin(wt * 10 + i * Math.PI) * 0.6 : 0; });
    if (P.wings) P.wings.forEach((w, i) => { w.rotation.z = (i ? -1 : 1) * (0.2 + Math.sin(wt * 28) * 0.55); });
    if (P.arms && walking) P.arms.forEach((ar, i) => { ar.rotation.x = Math.sin(wt * 10 + i * Math.PI) * 0.4; });
    // attack animation
    const ap = (renderT - u.lastAttack) / 0.3;
    if (ap >= 0 && ap < 1) {
      const s = Math.sin(ap * Math.PI);
      if (P.arm) P.arm.rotation.x = -s * 1.7;
      else if (P.weapon) P.weapon.rotation.x = -s * 0.25;
      u.m.body.position.z = c.range > 1.5 ? -s * 0.08 : s * 0.12;
    } else {
      if (P.arm && !walking) P.arm.rotation.x *= 0.8;
      u.m.body.position.z = 0;
    }
    // hit flash
    if (u.hp !== null && hp < u.hp) u.flashT = now;
    u.hp = hp;
    const fl = Math.max(0, 1 - (now - u.flashT) / 0.12);
    flash(u, u.m.mats, fl * 0.45);
    if (hp < maxHp && r1) {
      const top = (air ? 1.8 : 0) + unitScale(c) * (c.name === 'Giant' ? 1.55 : 1.2);
      hpBar(`u${uid}`, toLocal(A, x, y, top + 0.25), hp / maxHp, teamOf(owner), Math.max(26, c.radius * 60));
    }
  }
}

function updateAvatars(now, b) {
  for (const [seat, av] of avatars) {
    const P = av.parts;
    P.wings.forEach((w, i) => {
      const s = i ? -1 : 1;
      w.rotation.z = s * (0.25 + Math.sin(now * 95 + i) * 0.55);
      w.rotation.x = Math.sin(now * 95) * 0.15;
    });
    P.legs.forEach((l, i) => { l.rotation.x = Math.sin(now * 2 + i) * 0.12; });
    let target = av.home.clone();
    target.y += Math.sin(now * 2.1) * 0.3;
    if (av.swoop) {
      const p = now - av.swoop.t0;
      const card = av.swoop.card;
      if (p > 1.4) { world.remove(card); card.material.dispose(); av.swoop = null; }
      else {
        const k = p < 0.4 ? 1 - Math.pow(1 - p / 0.4, 3) : p < 0.6 ? 1 : 1 - (p - 0.6) / 0.8;
        target = av.home.clone().lerp(av.swoop.to, Math.max(0, k));
        // the fly carries its card in, lets go over the drop spot, and the card fades as it lands
        if (p < 0.45) card.position.copy(av.group.position).add(new THREE.Vector3(0, -1.3, 0));
        else card.position.y = Math.max(0.9, card.position.y - 0.12);
        card.material.opacity = p < 0.45 ? 1 : Math.max(0, 1 - (p - 0.45) / 0.6);
      }
    }
    av.group.position.lerp(target, av.swoop ? 0.35 : 0.08);
    // look at the action: the nearest enemy troop on its side, else the swoop spot / arena center
    let look = av.swoop ? av.swoop.to : toLocal(A, A.w / 2, A.h / 2);
    if (!av.swoop && b) {
      let best = 1e9;
      for (const r of b.units) {
        if (r[2] === seat) continue;
        const own = seat === 0 ? r[4] : A.h - r[4];
        if (own < best) { best = own; look = toLocal(A, r[3], r[4]); }
      }
    }
    const want = Math.atan2(look.x - av.group.position.x, look.z - av.group.position.z);
    av.yaw = lerpAngle(av.yaw, want, 0.06);
    av.group.rotation.y = av.yaw;
    av.body.rotation.x = av.swoop ? 0.35 : 0.08 + Math.sin(now * 1.3) * 0.03;
  }
}

function updateStatus(b) {
  const st = $('status');
  if (b.paused) { st.textContent = 'paused'; st.className = 'pill'; }
  else if (b.done) { st.textContent = 'match over'; st.className = 'pill'; }
  else if (vsFly && HUMAN !== null) { st.textContent = 'the fly is playing you'; st.className = 'pill live'; }
  else { st.textContent = b.double ? 'live · double elixir' : 'live'; st.className = 'pill live'; }
  $('speed').textContent = `${b.speed}x`;
  if (b.done && renderT >= b.t - 0.05) {
    const r = $('result');
    if (r.classList.contains('hidden')) {
      r.classList.remove('hidden');
      const t = $('result-title');
      if (b.winner === -1) { t.textContent = 'Draw'; t.style.background = '#444'; }
      else {
        t.textContent = `${info.players[b.winner].name} wins!`;
        t.style.background = teamOf(b.winner) ? 'linear-gradient(#f0525b,#c9232d)' : 'linear-gradient(#4b8cff,#2559c9)';
      }
      const n = b.winner === -1 ? 0 : b.crowns[b.winner];
      $('result-crowns').innerHTML = [0, 1, 2].map((i) => `<div class="crown${i < n ? ' on' : ''}"></div>`).join('');
    }
  }
}

let lastFrame = performance.now();
function frame(tms) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (tms - lastFrame) / 1000);
  lastFrame = tms;
  const now = tms / 1000;
  used.clear();
  if (snaps.length) {
    const newest = snaps[snaps.length - 1];
    const running = !(newest.paused || newest.done);
    renderT += dt * (running ? newest.speed : 1);
    const target = running ? newest.t - BUFFER * Math.max(1, newest.speed) : newest.t;
    const diff = target - renderT;
    if (Math.abs(diff) > 1.5 * Math.max(1, newest.speed)) renderT = target;
    else renderT += diff * Math.min(1, dt * 2.5);
    if (!running) renderT = Math.min(renderT, newest.t);
    pending.sort((p, q) => p[0] - q[0]);
    while (pending.length && pending[0][0] <= renderT) handleEvent(pending.shift());
    const { a, b, k } = sample(renderT);
    updateUnits(a, b, k, now, dt);
    updateTowers(b, now);
    updateHUD(a, b, k);
    updateAvatars(now, b);
    updateStatus(newest);
    updateWinbar(b.win_chance);
  }
  fx.update(renderT);
  arena.update(now);
  controls.update();
  renderer.render(scene, camera);
  for (const [key, el] of labelPool) if (!used.has(key)) el.style.display = 'none';
  if (brainView) brainView.update(dt);
}
requestAnimationFrame(frame);

// handy for poking at the scene from the browser console
window.flyDebug = { units, world, camera, get latest() { return latest; }, get renderT() { return renderT; } };
