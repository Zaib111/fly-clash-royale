// Arena: painted ground, animated river, bridges, walls, and scenery around it.
import * as THREE from 'three';
import { mat } from './models.js';

const PX = 40; // texture pixels per tile

function rng(seed) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

function groundTexture(A, maxAniso) {
  const W = A.w * PX, H = A.h * PX;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  const r = rng(11);
  // checkered grass with a little per-tile variation
  for (let x = 0; x < A.w; x++) {
    for (let y = 0; y < A.h; y++) {
      const base = (x + y) % 2 ? [112, 186, 72] : [124, 198, 80];
      const j = (r() - 0.5) * 8;
      g.fillStyle = `rgb(${base[0] + j},${base[1] + j},${base[2] + j * 0.5})`;
      g.fillRect(x * PX, y * PX, PX, PX);
    }
  }
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = r() < 0.5 ? 'rgba(255,255,255,0.05)' : 'rgba(0,40,0,0.06)';
    g.fillRect(r() * W, r() * H, 2, 3);
  }
  const cy = (y) => (A.h - y) * PX; // world y -> canvas y (top = far side)
  const lane = (x0, y0, x1, y1) => {
    const X = Math.min(x0, x1) * PX, Y = cy(Math.max(y0, y1));
    const w = Math.abs(x1 - x0) * PX, h = Math.abs(y1 - y0) * PX;
    g.beginPath();
    g.roundRect(X, Y, w, h, 14);
    g.fill();
  };
  for (const pass of [0, 1]) {
    g.fillStyle = pass ? '#d8b97c' : '#b8955d';
    const pad = pass ? 0 : 0.12;
    for (const [back, edge] of [[3, A.river[0]], [A.h - 3, A.river[1]]]) {
      const lo = Math.min(back, edge), hi = Math.max(back, edge);
      for (const bx of A.bridges) lane(bx - 0.9 - pad, lo - (back < 16 ? 0.9 : 0) - pad, bx + 0.9 + pad, hi + (back > 16 ? 0.9 : 0) + pad);
      lane(A.bridges[0] - 0.9 - pad, back - 0.9 - pad, A.bridges[1] + 0.9 + pad, back + 0.9 + pad);
    }
  }
  // dirt speckles on the lanes
  g.globalCompositeOperation = 'source-atop';
  for (let i = 0; i < 4000; i++) {
    g.fillStyle = r() < 0.5 ? 'rgba(120,90,50,0.10)' : 'rgba(255,240,200,0.10)';
    g.fillRect(r() * W, r() * H, 3, 2);
  }
  g.globalCompositeOperation = 'source-over';
  // river bed + banks
  g.fillStyle = '#5f8f3c';
  g.fillRect(0, cy(A.river[1] + 0.25), W, (A.river[1] - A.river[0] + 0.5) * PX);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = maxAniso;
  return t;
}

function waterTexture() {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, 0, 64);
  grd.addColorStop(0, '#2f86d6'); grd.addColorStop(0.5, '#4aa6ee'); grd.addColorStop(1, '#2f86d6');
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 64);
  g.strokeStyle = 'rgba(210,240,255,0.55)';
  g.lineWidth = 2.2;
  const r = rng(5);
  for (let i = 0; i < 14; i++) {
    const y = 6 + r() * 52, x = r() * 256, len = 18 + r() * 30;
    g.beginPath();
    g.moveTo(x, y);
    g.quadraticCurveTo(x + len / 2, y - 3, x + len, y);
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(4, 1);
  return t;
}

function outerGrassTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#5c9a3a';
  g.fillRect(0, 0, 256, 256);
  const r = rng(23);
  for (let i = 0; i < 2600; i++) {
    const v = r();
    g.strokeStyle = v < 0.5 ? 'rgba(140,200,90,0.5)' : 'rgba(40,90,30,0.5)';
    g.lineWidth = 1.2;
    const x = r() * 256, y = r() * 256;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + (r() - 0.5) * 3, y - 4 - r() * 4); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(36, 36);
  return t;
}

// World coords (x right, y toward the far side) -> local scene coords.
export const toLocal = (A, x, y, h = 0) => new THREE.Vector3(x - A.w / 2, h, A.h / 2 - y);

export function buildArena(world, A, renderer) {
  const aniso = renderer.capabilities.getMaxAnisotropy();
  // surrounding meadow
  const outer = new THREE.Mesh(new THREE.PlaneGeometry(240, 240), new THREE.MeshStandardMaterial({ map: outerGrassTexture(), roughness: 1 }));
  outer.rotation.x = -Math.PI / 2;
  outer.position.y = -0.05;
  outer.receiveShadow = true;
  world.add(outer);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(A.w, A.h), new THREE.MeshStandardMaterial({ map: groundTexture(A, aniso), roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  world.add(ground);

  // river
  const water = waterTexture();
  const rw = A.river[1] - A.river[0];
  const river = new THREE.Mesh(new THREE.PlaneGeometry(A.w, rw), new THREE.MeshStandardMaterial({
    map: water, roughness: 0.15, metalness: 0.1, transparent: true, opacity: 0.95,
  }));
  river.rotation.x = -Math.PI / 2;
  river.position.copy(toLocal(A, A.w / 2, (A.river[0] + A.river[1]) / 2, 0.03));
  river.receiveShadow = true;
  world.add(river);
  const bank = mat(0x6a5a3a, { roughness: 1 });
  for (const y of A.river) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(A.w, 0.12, 0.18), bank);
    b.position.copy(toLocal(A, A.w / 2, y, 0.05));
    b.receiveShadow = true;
    world.add(b);
  }

  // bridges
  const wood = mat(0xa36d3c, { roughness: 0.8 });
  const woodDark = mat(0x6e4523, { roughness: 0.85 });
  for (const bx of A.bridges) {
    const g = new THREE.Group();
    g.position.copy(toLocal(A, bx, (A.river[0] + A.river[1]) / 2, 0));
    const deck = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.2, rw + 0.8), wood);
    deck.position.y = 0.12;
    g.add(deck);
    for (let k = -3; k <= 3; k++) {
      const plank = new THREE.Mesh(new THREE.BoxGeometry(2.62, 0.03, 0.04), woodDark);
      plank.position.set(0, 0.225, k * 0.38);
      g.add(plank);
    }
    for (const s of [-1, 1]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.1, rw + 0.8), woodDark);
      rail.position.set(s * 1.25, 0.6, 0);
      g.add(rail);
      for (const z of [-(rw + 0.6) / 2, 0, (rw + 0.6) / 2]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.5, 0.16), woodDark);
        post.position.set(s * 1.25, 0.4, z);
        g.add(post);
      }
    }
    g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    world.add(g);
  }

  // low stone wall around the arena
  const wallM = mat(0x8d8f99, { roughness: 0.9 });
  const capM = mat(0xb4b6c0, { roughness: 0.85 });
  const T = 0.4;
  for (const [w, d, x, y] of [
    [A.w + 2 * T, T, A.w / 2, -T / 2], [A.w + 2 * T, T, A.w / 2, A.h + T / 2],
    [T, A.h, -T / 2, A.h / 2], [T, A.h, A.w + T / 2, A.h / 2],
  ]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 0.5, d), wallM);
    wall.position.copy(toLocal(A, x, y, 0.25));
    const cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.06, 0.1, d + 0.06), capM);
    cap.position.copy(toLocal(A, x, y, 0.55));
    for (const m of [wall, cap]) { m.castShadow = true; m.receiveShadow = true; world.add(m); }
  }

  // scenery: trees, bushes and rocks outside the walls
  const r = rng(41);
  const trunk = mat(0x6b4426);
  const leaves = [mat(0x3f8a3a), mat(0x4e9c40), mat(0x357a33)];
  const rock = mat(0x9a9aa4, { roughness: 0.9, flatShading: true });
  for (let i = 0; i < 70; i++) {
    let x, y;
    do {
      x = -14 + r() * (A.w + 28);
      y = -12 + r() * (A.h + 24);
    } while (x > -1.5 && x < A.w + 1.5 && y > -1.5 && y < A.h + 1.5);
    const g = new THREE.Group();
    g.position.copy(toLocal(A, x, y, 0));
    const kind = r();
    if (kind < 0.5) {
      const h = 1.2 + r() * 1.2;
      const t = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.22, h, 8), trunk);
      t.position.y = h / 2;
      g.add(t);
      for (let k = 0; k < 3; k++) {
        const s = 0.9 + r() * 0.7;
        const l = new THREE.Mesh(new THREE.IcosahedronGeometry(s, 1), leaves[k % 3]);
        l.position.set((r() - 0.5) * 0.8, h + 0.3 + k * 0.35, (r() - 0.5) * 0.8);
        g.add(l);
      }
    } else if (kind < 0.85) {
      for (let k = 0; k < 3; k++) {
        const s = 0.35 + r() * 0.35;
        const b = new THREE.Mesh(new THREE.IcosahedronGeometry(s, 1), leaves[k % 3]);
        b.position.set((r() - 0.5) * 0.9, s * 0.7, (r() - 0.5) * 0.9);
        g.add(b);
      }
    } else {
      const s = 0.3 + r() * 0.5;
      const b = new THREE.Mesh(new THREE.DodecahedronGeometry(s, 0), rock);
      b.position.y = s * 0.5;
      b.rotation.set(r(), r(), r());
      g.add(b);
    }
    g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    world.add(g);
  }

  return {
    ground,
    update(t) {
      water.offset.x = (t * 0.06) % 1;
      water.offset.y = Math.sin(t * 0.7) * 0.02;
    },
  };
}
