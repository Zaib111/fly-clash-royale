// Low-poly models built from primitives (no external assets).
// Units face +Z in their local space; one unit of model space ~ one arena tile.
import * as THREE from 'three';

export const TEAM = [0x3478f6, 0xe8343e];
export const TEAM_DARK = [0x1d3f8f, 0x8a1c24];
const SKIN = 0xf3c7a0;
const STONE = 0x9ea0ad;
const STONE_DARK = 0x6d6e7a;
const STONE_LIGHT = 0xc4c6d0;
const GOLD = 0xffcc33;

const G = {
  sphere: new THREE.SphereGeometry(1, 24, 16),
  lowSphere: new THREE.SphereGeometry(1, 12, 8),
  box: new THREE.BoxGeometry(1, 1, 1),
  cyl: new THREE.CylinderGeometry(1, 1, 1, 20),
  cone: new THREE.ConeGeometry(1, 1, 18),
  ico: new THREE.IcosahedronGeometry(1, 2),
  dodeca: new THREE.DodecahedronGeometry(1, 0),
  halfTorus: new THREE.TorusGeometry(1, 0.08, 8, 24, Math.PI),
  ring: new THREE.RingGeometry(0.78, 1, 40),
  disc: new THREE.CircleGeometry(1, 32),
};

export function mat(color, opts = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0, ...opts });
}

// Adds a mesh to `parent`; returns it. pos/scale/rot are arrays.
function part(parent, geo, material, pos = [0, 0, 0], scale = [1, 1, 1], rot = [0, 0, 0]) {
  const m = new THREE.Mesh(G[geo], material);
  m.position.set(...pos);
  m.scale.set(...scale);
  m.rotation.set(...rot);
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}

function pivot(parent, pos) {
  const g = new THREE.Group();
  g.position.set(...pos);
  parent.add(g);
  return g;
}

// ------------------------------------------------------------------ troops
function legs(root, color, spread = 0.12, len = 0.32, width = 0.13) {
  const m = mat(color);
  return [-1, 1].map((s) => {
    const p = pivot(root, [s * spread, len, 0]);
    part(p, 'box', m, [0, -len / 2, 0], [width, len, width * 1.15]);
    return p;
  });
}

function eyes(root, y, z, spread, r = 0.035) {
  const m = mat(0x111111, { roughness: 0.3 });
  for (const s of [-1, 1]) part(root, 'sphere', m, [s * spread, y, z], [r, r, r]);
}

const BUILDERS = {
  Knight(root, team, P) {
    P.legs = legs(root, 0x5a5f6a);
    const steel = mat(0xb9bec9, { metalness: 0.45, roughness: 0.35 });
    part(root, 'box', steel, [0, 0.52, 0], [0.48, 0.42, 0.32]);
    part(root, 'box', mat(TEAM[team]), [0, 0.5, 0.01], [0.38, 0.34, 0.345]);
    part(root, 'box', mat(0x6b3f1f), [0, 0.33, 0], [0.5, 0.07, 0.34]);
    part(root, 'box', mat(TEAM_DARK[team]), [0, 0.56, -0.19], [0.42, 0.5, 0.04], [0.12, 0, 0]);
    part(root, 'sphere', mat(SKIN), [0, 0.88, 0.02], [0.2, 0.2, 0.2]);
    part(root, 'cyl', steel, [0, 0.98, 0], [0.22, 0.2, 0.22]);
    part(root, 'sphere', steel, [0, 1.07, 0], [0.22, 0.12, 0.22]);
    part(root, 'box', mat(0x1a1a1a), [0, 0.95, 0.2], [0.28, 0.04, 0.05]);
    part(root, 'box', mat(0x8a5a2b), [0, 0.8, 0.19], [0.22, 0.05, 0.05]);
    // shield on the left arm
    part(root, 'cyl', mat(TEAM[team]), [-0.3, 0.55, 0.08], [0.17, 0.05, 0.2], [0, 0, Math.PI / 2]);
    // sword arm
    P.arm = pivot(root, [0.3, 0.66, 0]);
    part(P.arm, 'box', steel, [0, -0.1, 0.05], [0.12, 0.22, 0.12]);
    P.weapon = pivot(P.arm, [0, -0.2, 0.12]);
    part(P.weapon, 'box', mat(0xe8edf5, { metalness: 0.7, roughness: 0.25 }), [0, 0.32, 0], [0.06, 0.6, 0.025]);
    part(P.weapon, 'box', mat(GOLD, { metalness: 0.6 }), [0, 0.02, 0], [0.2, 0.04, 0.05]);
    P.weapon.rotation.x = 0.9;
  },
  Archers(root, team, P) {
    P.legs = legs(root, 0x3c6b35, 0.09, 0.28, 0.11);
    const green = mat(0x4f8f45);
    part(root, 'cyl', green, [0, 0.5, 0], [0.18, 0.36, 0.16]);
    part(root, 'cone', mat(TEAM[team]), [0, 0.38, 0], [0.24, 0.22, 0.22], [Math.PI, 0, 0]);
    part(root, 'sphere', mat(SKIN), [0, 0.84, 0.02], [0.18, 0.18, 0.18]);
    const hair = mat(0xec5fa8);
    part(root, 'sphere', hair, [0, 0.9, -0.04], [0.2, 0.18, 0.2]);
    part(root, 'sphere', hair, [0, 0.82, -0.22], [0.09, 0.16, 0.09]);
    eyes(root, 0.86, 0.17, 0.065);
    P.weapon = pivot(root, [-0.22, 0.58, 0.16]);
    part(P.weapon, 'halfTorus', mat(0x7a4a22), [0, 0, 0], [0.3, 0.3, 0.6], [0, Math.PI / 2, Math.PI / 2]);
  },
  Giant(root, team, P) {
    P.legs = legs(root, 0x5b3a1e, 0.16, 0.38, 0.2);
    part(root, 'box', mat(0x8a5a2e), [0, 0.68, 0], [0.7, 0.6, 0.46]);
    part(root, 'box', mat(TEAM[team]), [0, 0.42, 0], [0.74, 0.1, 0.5]);
    part(root, 'box', mat(GOLD, { metalness: 0.5 }), [0, 0.42, 0.25], [0.12, 0.1, 0.02]);
    const skin = mat(0xe2ad7f);
    part(root, 'sphere', skin, [0, 1.15, 0.04], [0.27, 0.28, 0.27]);
    part(root, 'cone', mat(0xe0782a), [0, 1.42, -0.06], [0.09, 0.14, 0.09]);
    eyes(root, 1.18, 0.25, 0.09, 0.04);
    part(root, 'box', mat(0x5b3a1e), [0, 1.25, 0.24], [0.28, 0.04, 0.03]);
    P.arms = [-1, 1].map((s) => {
      const a = pivot(root, [s * 0.43, 0.9, 0]);
      part(a, 'cyl', skin, [0, -0.22, 0], [0.1, 0.44, 0.1]);
      part(a, 'sphere', skin, [0, -0.48, 0.02], [0.15, 0.15, 0.15]);
      return a;
    });
    P.arm = P.arms[1];
  },
  Goblins(root, team, P) {
    P.legs = legs(root, 0x4f9a3a, 0.08, 0.2, 0.09);
    const g = mat(0x6fcf4f);
    part(root, 'cyl', mat(TEAM[team]), [0, 0.32, 0], [0.15, 0.22, 0.13]);
    part(root, 'sphere', g, [0, 0.6, 0.02], [0.23, 0.21, 0.21]);
    for (const s of [-1, 1]) part(root, 'cone', g, [s * 0.3, 0.66, 0], [0.07, 0.25, 0.05], [0, 0, -s * 1.3]);
    const white = mat(0xffffff);
    for (const s of [-1, 1]) part(root, 'sphere', white, [s * 0.08, 0.64, 0.18], [0.06, 0.06, 0.04]);
    eyes(root, 0.64, 0.215, 0.08, 0.025);
    P.weapon = pivot(root, [0.2, 0.36, 0.1]);
    part(P.weapon, 'box', mat(0xd5d8de, { metalness: 0.6, roughness: 0.3 }), [0, 0.12, 0], [0.04, 0.24, 0.02]);
    P.weapon.rotation.x = 0.8;
  },
  Musketeer(root, team, P) {
    const purple = mat(0x8a4fc0);
    part(root, 'cone', purple, [0, 0.3, 0], [0.3, 0.6, 0.28]);
    part(root, 'cyl', purple, [0, 0.62, 0], [0.16, 0.26, 0.14]);
    part(root, 'box', mat(TEAM[team]), [0, 0.6, 0.02], [0.36, 0.06, 0.3], [0, 0, 0.5]);
    part(root, 'sphere', mat(SKIN), [0, 0.88, 0.02], [0.17, 0.17, 0.17]);
    part(root, 'sphere', mat(0x4a2a12), [0, 0.88, -0.07], [0.18, 0.17, 0.16]);
    eyes(root, 0.9, 0.17, 0.06);
    const hat = mat(0x4a2a6a);
    part(root, 'cyl', hat, [0, 1.01, 0], [0.32, 0.03, 0.32]);
    part(root, 'cyl', hat, [0, 1.09, 0], [0.16, 0.15, 0.16]);
    part(root, 'cone', mat(0xffd86a), [0.12, 1.2, -0.12], [0.06, 0.32, 0.06], [-0.6, 0, -0.4]);
    P.weapon = pivot(root, [0.18, 0.62, 0.05]);
    part(P.weapon, 'box', mat(0x3a2a20), [0, 0, 0.35], [0.05, 0.06, 0.8]);
    part(P.weapon, 'box', mat(0x8a5a2b), [0, -0.03, 0.05], [0.07, 0.1, 0.25]);
  },
  MiniPekka(root, team, P) {
    P.legs = legs(root, 0x22305a, 0.13, 0.3, 0.15);
    const armor = mat(0x2c3e70, { metalness: 0.55, roughness: 0.35 });
    part(root, 'box', armor, [0, 0.56, 0], [0.52, 0.46, 0.42]);
    for (const s of [-1, 1]) part(root, 'sphere', mat(TEAM[team]), [s * 0.3, 0.76, 0], [0.13, 0.11, 0.15]);
    part(root, 'box', armor, [0, 0.96, 0.02], [0.4, 0.32, 0.38]);
    const horn = mat(0xd9dce4);
    for (const s of [-1, 1]) part(root, 'cone', horn, [s * 0.2, 1.17, 0], [0.06, 0.28, 0.06], [0, 0, -s * 0.5]);
    part(root, 'box', mat(0x9ff3ff, { emissive: 0x39d7ff, emissiveIntensity: 1.6 }), [0, 0.97, 0.2], [0.28, 0.05, 0.02]);
    P.arm = pivot(root, [0.33, 0.7, 0]);
    P.weapon = pivot(P.arm, [0, -0.1, 0.1]);
    part(P.weapon, 'box', mat(0xc9e6ff, { metalness: 0.75, roughness: 0.2 }), [0, 0.38, 0], [0.1, 0.72, 0.035]);
    part(P.weapon, 'box', mat(0x22305a), [0, 0, 0], [0.26, 0.06, 0.08]);
    P.weapon.rotation.x = 0.8;
  },
  Minions(root, team, P) {
    const body = mat(0x5c7cf0);
    part(root, 'sphere', body, [0, 0, 0], [0.26, 0.25, 0.26]);
    part(root, 'box', mat(TEAM[team]), [0, -0.15, 0], [0.3, 0.08, 0.28]);
    const horn = mat(0xeeeeee);
    for (const s of [-1, 1]) part(root, 'cone', horn, [s * 0.1, 0.27, 0.05], [0.04, 0.14, 0.04], [0, 0, -s * 0.4]);
    const eye = mat(0xffee66, { emissive: 0xffd400, emissiveIntensity: 0.6 });
    for (const s of [-1, 1]) part(root, 'sphere', eye, [s * 0.08, 0.04, 0.22], [0.05, 0.05, 0.03]);
    const wingMat = mat(0x3b3f9e, { side: THREE.DoubleSide });
    P.wings = [-1, 1].map((s) => {
      const w = pivot(root, [s * 0.18, 0.08, -0.05]);
      part(w, 'box', wingMat, [s * 0.28, 0, -0.06], [0.5, 0.02, 0.3]);
      return w;
    });
    P.air = true;
  },
  Fireball(root) {
    part(root, 'sphere', mat(0xff7a1a, { emissive: 0xff5a00, emissiveIntensity: 2 }), [0, 0, 0], [0.45, 0.45, 0.45]);
    part(root, 'sphere', mat(0xffe08a, { emissive: 0xffc040, emissiveIntensity: 2.5 }), [0.08, 0.08, 0.15], [0.25, 0.25, 0.25]);
  },
};

// A troop model. Returns { group, body, parts, mats }.
export function makeUnit(name, team, { ghost = false, ring = true } = {}) {
  const group = new THREE.Group();
  const body = new THREE.Group();
  group.add(body);
  const P = {};
  BUILDERS[name](body, team, P);
  if (ring && name !== 'Fireball') {
    const r = new THREE.Mesh(G.ring, new THREE.MeshBasicMaterial({
      color: TEAM[team], transparent: true, opacity: 0.85, depthWrite: false,
    }));
    r.rotation.x = -Math.PI / 2;
    r.scale.setScalar(0.42);
    r.position.y = 0.02;
    group.add(r);
    P.ring = r;
  }
  const mats = [];
  group.traverse((o) => {
    if (o.isMesh) {
      o.material = o.material.clone();
      mats.push(o.material);
      if (ghost) {
        o.material.transparent = true;
        o.material.opacity = 0.5;
        o.material.depthWrite = false;
        o.castShadow = false;
      }
    }
  });
  return { group, body, parts: P, mats };
}

// ------------------------------------------------------------------ towers
function crenellations(root, half, y, n, m) {
  const step = (2 * half) / (n - 1);
  const s = Math.min(0.42, step * 0.55);
  for (let i = 0; i < n; i++) {
    const t = -half + i * step;
    for (const [x, z] of [[t, -half], [t, half], [-half, t], [half, t]]) {
      part(root, 'box', m, [x, y, z], [s, 0.42, s]);
    }
  }
}

function crown(root, pos, size) {
  const g = mat(GOLD, { metalness: 0.7, roughness: 0.3 });
  const c = pivot(root, pos);
  part(c, 'cyl', g, [0, 0, 0], [size, size * 0.5, size]);
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    part(c, 'cone', g, [Math.cos(a) * size * 0.85, size * 0.5, Math.sin(a) * size * 0.85], [size * 0.22, size * 0.6, size * 0.22]);
  }
  return c;
}

export function makeTower(isKing, team) {
  const group = new THREE.Group();
  const P = {};
  const stone = mat(STONE, { roughness: 0.85 });
  const dark = mat(STONE_DARK, { roughness: 0.9 });
  const light = mat(STONE_LIGHT, { roughness: 0.8 });
  const teamM = mat(TEAM[team]);
  const W = isKing ? 3.8 : 2.6;
  const H = isKing ? 2.0 : 1.6;
  part(group, 'box', dark, [0, 0.15, 0], [W + 0.35, 0.3, W + 0.35]);
  part(group, 'box', stone, [0, 0.3 + H / 2, 0], [W, H, W]);
  for (const y of [0.8, 1.3, 1.8].slice(0, isKing ? 3 : 2)) {
    part(group, 'box', dark, [0, y, 0], [W + 0.04, 0.05, W + 0.04]);
  }
  const top = 0.3 + H;
  part(group, 'box', light, [0, top + 0.12, 0], [W + 0.3, 0.25, W + 0.3]);
  crenellations(group, (W + 0.3) / 2 - 0.2, top + 0.45, isKing ? 5 : 4, light);
  // front door + banners (front = +Z, which faces the owner's side)
  part(group, 'box', mat(0x5a3a1e), [0, 0.7, W / 2 + 0.01], [0.7, 0.8, 0.05]);
  part(group, 'cyl', mat(0x5a3a1e), [0, 1.1, W / 2 + 0.01], [0.35, 0.05, 0.35], [Math.PI / 2, 0, 0]);
  for (const s of [-1, 1]) {
    part(group, 'box', teamM, [s * W * 0.3, 1.1, W / 2 + 0.02], [0.45, 0.75, 0.04]);
    part(group, 'cone', teamM, [s * W * 0.3, 0.63, W / 2 + 0.02], [0.23, 0.2, 0.02], [Math.PI, 0, 0]);
  }
  part(group, 'cyl', mat(TEAM_DARK[team]), [0, top + 0.27, 0], [W * 0.4, 0.08, W * 0.4]);
  part(group, 'cyl', teamM, [0, top + 0.33, 0], [W * 0.38, 0.06, W * 0.38]);
  if (isKing) {
    // corner turrets with team roofs
    for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const h = W / 2 + 0.05;
      part(group, 'cyl', light, [x * h, top + 0.45, z * h], [0.4, 0.9, 0.4]);
      part(group, 'cone', teamM, [x * h, top + 1.15, z * h], [0.5, 0.6, 0.5]);
    }
    // cannon (aims at targets) and the king behind it
    P.aim = pivot(group, [0, top + 0.6, -0.2]);
    part(P.aim, 'box', mat(0x4a4a52), [0, -0.1, 0], [0.7, 0.3, 0.8]);
    part(P.aim, 'cyl', mat(0x2b2b31, { metalness: 0.6, roughness: 0.35 }), [0, 0.12, -0.55], [0.2, 1.0, 0.2], [Math.PI / 2, 0, 0]);
    const king = pivot(group, [0, top + 0.95, 0.75]);
    part(king, 'sphere', mat(TEAM[team]), [0, -0.1, 0], [0.55, 0.4, 0.45]);
    part(king, 'sphere', mat(SKIN), [0, 0.42, 0], [0.4, 0.4, 0.38]);
    part(king, 'sphere', mat(0xf2f2f2), [0, 0.2, -0.18], [0.36, 0.3, 0.25]);
    part(king, 'sphere', mat(0xe58a7a), [0, 0.38, -0.36], [0.07, 0.07, 0.07]);
    P.eyesOpen = new THREE.Group();
    P.eyesClosed = new THREE.Group();
    king.add(P.eyesOpen, P.eyesClosed);
    for (const s of [-1, 1]) {
      part(P.eyesOpen, 'sphere', mat(0x111111), [s * 0.13, 0.5, -0.33], [0.05, 0.06, 0.04]);
      part(P.eyesClosed, 'box', mat(0x111111), [s * 0.13, 0.48, -0.35], [0.12, 0.025, 0.02]);
    }
    crown(king, [0, 0.82, 0], 0.3);
    P.king = king;
  } else {
    P.aim = pivot(group, [0, top + 0.4, 0]);
    part(P.aim, 'cone', teamM, [0, 0.3, 0], [0.32, 0.6, 0.32]);
    part(P.aim, 'sphere', mat(SKIN), [0, 0.75, 0], [0.21, 0.21, 0.21]);
    part(P.aim, 'sphere', mat(0xec5fa8), [0, 0.8, 0.07], [0.23, 0.21, 0.22]);
    crown(P.aim, [0, 1.0, 0], 0.12);
    part(P.aim, 'halfTorus', mat(0x7a4a22), [0.2, 0.55, -0.25], [0.3, 0.3, 0.6], [0, Math.PI / 2, Math.PI / 2]);
  }
  group.traverse((o) => { if (o.isMesh) o.material = o.material.clone(); });
  const mats = [];
  group.traverse((o) => { if (o.isMesh) mats.push(o.material); });
  return { group, parts: P, mats, height: top + (isKing ? 2.0 : 1.4) };
}

export function makeRubble(isKing, team) {
  const group = new THREE.Group();
  const W = isKing ? 3.8 : 2.6;
  const scorch = new THREE.Mesh(G.disc, new THREE.MeshBasicMaterial({ color: 0x2a2a20, transparent: true, opacity: 0.45, depthWrite: false }));
  scorch.rotation.x = -Math.PI / 2;
  scorch.scale.setScalar(W * 0.75);
  scorch.position.y = 0.03;
  group.add(scorch);
  const stone = mat(STONE, { roughness: 0.9 });
  const dark = mat(STONE_DARK, { roughness: 0.9 });
  let seed = isKing ? 7 : 3;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 14; i++) {
    const r = (0.25 + rnd() * 0.45) * (isKing ? 1.2 : 1);
    const a = rnd() * Math.PI * 2;
    const d = rnd() * W * 0.45;
    part(group, 'dodeca', rnd() < 0.5 ? stone : dark, [Math.cos(a) * d, r * 0.6, Math.sin(a) * d], [r, r * 0.8, r], [rnd(), rnd(), rnd()]);
  }
  part(group, 'box', mat(TEAM_DARK[team]), [0.3, 0.12, 0.2], [0.8, 0.06, 0.5], [0, 0.6, 0.15]);
  return group;
}

// ------------------------------------------------------------------ the fly
export function makeFly() {
  const group = new THREE.Group();
  const body = new THREE.Group();
  group.add(body);
  const P = {};
  const tan = mat(0xa8743f, { roughness: 0.55 });
  const light = mat(0xc99a5e, { roughness: 0.55 });
  const dark = mat(0x3a2412, { roughness: 0.6 });
  part(body, 'sphere', tan, [0, 0, 0], [0.5, 0.45, 0.58]);                    // thorax
  part(body, 'sphere', mat(0x8f5f30), [0, 0.18, -0.05], [0.42, 0.3, 0.48]);    // darker back
  part(body, 'sphere', light, [0, -0.04, -0.82], [0.4, 0.37, 0.62]);          // abdomen
  for (const [z, s] of [[-0.62, 1], [-0.86, 0.95], [-1.1, 0.75]]) {
    part(body, 'sphere', dark, [0, 0.03, z], [0.405 * s, 0.375 * s, 0.07]);   // stripes
  }
  P.head = pivot(body, [0, 0.05, 0.6]);
  part(P.head, 'sphere', tan, [0, 0, 0], [0.32, 0.29, 0.26]);
  const eye = mat(0xc41e1e, { roughness: 0.25, metalness: 0.15, flatShading: true });
  for (const s of [-1, 1]) {
    part(P.head, 'ico', eye, [s * 0.24, 0.04, 0.06], [0.2, 0.25, 0.22]);
    part(P.head, 'cyl', dark, [s * 0.07, 0.22, 0.22], [0.015, 0.16, 0.015], [0.6, 0, s * 0.3]);
  }
  part(P.head, 'cyl', dark, [0, -0.25, 0.12], [0.04, 0.18, 0.04], [0.3, 0, 0]);
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xdcecff, transparent: true, opacity: 0.38, roughness: 0.15, metalness: 0.3,
    side: THREE.DoubleSide, depthWrite: false,
  });
  P.wings = [-1, 1].map((s) => {
    const w = pivot(body, [s * 0.12, 0.38, -0.05]);
    const m = new THREE.Mesh(G.disc, wingMat);
    m.rotation.x = -Math.PI / 2;
    m.rotation.z = s * 0.35;
    m.scale.set(0.42, 1.0, 1);
    m.position.set(s * 0.42, 0, -0.62);
    w.add(m);
    return w;
  });
  P.legs = [];
  for (const z of [0.3, 0.02, -0.25]) {
    for (const s of [-1, 1]) {
      const hip = pivot(body, [s * 0.25, -0.25, z]);
      hip.rotation.z = s * 0.9;
      part(hip, 'cyl', dark, [0, -0.25, 0], [0.035, 0.5, 0.035]);
      const knee = pivot(hip, [0, -0.5, 0]);
      knee.rotation.z = -s * 1.3;
      part(knee, 'cyl', dark, [0, -0.25, 0], [0.028, 0.5, 0.028]);
      P.legs.push(hip);
    }
  }
  group.traverse((o) => { if (o.isMesh && o.material !== wingMat) { o.castShadow = true; } });
  P.wings.forEach((w) => w.children.forEach((m) => { m.castShadow = false; }));
  return { group, body, parts: P };
}

// ------------------------------------------------------------------ card portraits
let portraitRenderer = null;

export function makePortrait(name, w = 140, h = 168) {
  if (!portraitRenderer) {
    portraitRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    portraitRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  }
  const r = portraitRenderer;
  r.setPixelRatio(2);
  r.setSize(w, h, false);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1.5, 3, 4);
  scene.add(sun);
  const layouts = {
    Goblins: [[-0.32, 0, 0], [0.32, 0, 0], [0, 0, 0.3]],
    Archers: [[-0.25, 0, 0], [0.25, 0, 0]],
    Minions: [[-0.35, 0.35, 0], [0.35, 0.35, 0], [0, 0.6, 0.2]],
  };
  const spots = layouts[name] || [[0, 0, 0]];
  const box = new THREE.Box3();
  for (const p of spots) {
    const u = makeUnit(name, 0, { ring: false });
    u.group.position.set(...p);
    if (name === 'Minions') u.group.scale.setScalar(0.9);
    scene.add(u.group);
  }
  box.setFromObject(scene);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const cam = new THREE.PerspectiveCamera(30, w / h, 0.1, 50);
  const d = Math.max(size.y, size.x * h / w) * 2.0 + 0.6;
  cam.position.set(center.x + d * 0.25, center.y + d * 0.3, center.z + d);
  cam.lookAt(center);
  r.render(scene, cam);
  const url = r.domElement.toDataURL('image/png');
  scene.traverse((o) => { if (o.isMesh) o.material.dispose(); });
  return url;
}

export { G };
