// The fly's brain as a glowing point cloud at the neurons' real 3D positions (FlyWire).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

export const GROUP_COLORS = [
  [0.42, 0.52, 1.0],   // central brain
  [0.25, 0.8, 1.0],    // eyes & optic lobes
  [0.35, 1.0, 0.55],   // other senses
  [1.0, 0.35, 0.85],   // descending & motor
];
const HOT = [1.0, 0.86, 0.45];

function dotTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.35, 'rgba(255,255,255,0.6)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export class BrainView {
  constructor(canvas, buffer, count) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
    this.camera.position.set(0, 0.25, 2.45);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.7;
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 6;

    const pos = new Float32Array(buffer, 0, count * 3);
    const group = new Uint8Array(buffer, count * 12, count);
    this.count = count;
    this.group = group;
    this.level = new Float32Array(count);    // displayed activity (smoothed)
    this.target = new Float32Array(count);   // activity each neuron is easing toward
    this.next = new Float32Array(count);     // newest activity from the server, revealed in stages
    this.byGroup = [0, 1, 2, 3].map((g) => Int32Array.from([...group.keys()].filter((i) => group[i] === g)));
    this.reveal = null;                      // {t0, done: [bool x4]}
    this.clock = 0;
    const tex = dotTexture();
    // two layers: everything, plus bigger dots for the output (descending/motor) neurons
    this.layers = [];
    for (const big of [false, true]) {
      const idx = [];
      for (let i = 0; i < count; i++) if ((group[i] === 3) === big) idx.push(i);
      const geo = new THREE.BufferGeometry();
      const p = new Float32Array(idx.length * 3);
      idx.forEach((i, k) => p.set(pos.subarray(i * 3, i * 3 + 3), k * 3));
      geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
      const col = new THREE.BufferAttribute(new Float32Array(idx.length * 3), 3);
      col.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('color', col);
      const material = new THREE.PointsMaterial({
        size: big ? 0.06 : 0.024, map: tex, vertexColors: true, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
      });
      const pts = new THREE.Points(geo, material);
      this.scene.add(pts);
      this.layers.push({ idx: Int32Array.from(idx), col });
    }
    // signal pulses: eyes -> central brain -> descending/motor
    this.maxPulses = 160;
    this.pulses = [];
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.maxPulses * 3), 3));
    pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.maxPulses * 3), 3));
    this.pulseGeo = pg;
    this.scene.add(new THREE.Points(pg, new THREE.PointsMaterial({
      size: 0.07, map: tex, vertexColors: true, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    })));
    this.pos = pos;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  point(group) {
    const ids = this.byGroup[group];
    const i = ids[Math.floor(Math.random() * ids.length)];
    return new THREE.Vector3(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
  }

  // A new decision: reveal activity as a wave (senses -> central -> motor) and fire pulses.
  decide(nPulses = 40) {
    this.reveal = { t0: this.clock, done: [false, false, false, false] };
    for (let k = 0; k < nPulses && this.byGroup[1].length && this.byGroup[3].length; k++) {
      if (this.pulses.length >= this.maxPulses) this.pulses.shift();
      this.pulses.push({
        a: this.point(Math.random() < 0.8 ? 1 : 2), b: this.point(0), c: this.point(3),
        t0: this.clock + Math.random() * 0.15, dur: 0.55 + Math.random() * 0.25,
      });
    }
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // back off far enough that the whole brain (about +-1 wide) fits this panel's shape
    const halfFov = Math.tan((this.camera.fov * Math.PI) / 360);
    const dist = Math.min(4, Math.max(1.6, 0.85 / (halfFov * this.camera.aspect), 0.62 / halfFov));
    this.camera.position.setLength(dist);
  }

  setActivity(b64) {
    const bin = atob(b64);
    const n = Math.min(bin.length, this.count);
    for (let i = 0; i < n; i++) this.next[i] = bin.charCodeAt(i) / 255;
    this.decide();
  }

  update(dt) {
    this.clock += dt;
    if (this.reveal) {
      const DELAY = [0.2, 0.0, 0.0, 0.42];   // central, eyes, senses, motor
      const el = this.clock - this.reveal.t0;
      for (let g = 0; g < 4; g++) {
        if (!this.reveal.done[g] && el >= DELAY[g]) {
          for (const i of this.byGroup[g]) this.target[i] = this.next[i];
          this.reveal.done[g] = true;
        }
      }
      if (this.reveal.done.every(Boolean)) this.reveal = null;
    }
    const pp = this.pulseGeo.attributes.position.array, pc = this.pulseGeo.attributes.color.array;
    pp.fill(0); pc.fill(0);
    this.pulses = this.pulses.filter((p) => this.clock - p.t0 < p.dur);
    this.pulses.forEach((p, k) => {
      const t = Math.max(0, (this.clock - p.t0) / p.dur);
      if (t <= 0) return;
      const v = t < 0.5 ? p.a.clone().lerp(p.b, t * 2) : p.b.clone().lerp(p.c, (t - 0.5) * 2);
      v.y += Math.sin(t * Math.PI) * 0.08;
      pp.set([v.x, v.y, v.z], k * 3);
      const warm = t < 0.5 ? [0.5, 0.9, 1.0] : [1.0, 0.55, 0.95];   // cyan from the eyes, pink into motor
      const f = Math.sin(t * Math.PI) * 1.4;
      pc.set([warm[0] * f, warm[1] * f, warm[2] * f], k * 3);
    });
    this.pulseGeo.setDrawRange(0, this.pulses.length);   // don't draw idle pulse slots
    this.pulseGeo.attributes.position.needsUpdate = true;
    this.pulseGeo.attributes.color.needsUpdate = true;
    const k = Math.min(1, dt * 7);
    const lv = this.level, tg = this.target, grp = this.group;
    for (let i = 0; i < this.count; i++) lv[i] += (tg[i] - lv[i]) * k;
    for (const L of this.layers) {
      const arr = L.col.array;
      const idx = L.idx;
      for (let k2 = 0; k2 < idx.length; k2++) {
        const i = idx[k2];
        const a = Math.min(1, lv[i]);
        const base = GROUP_COLORS[grp[i]];
        const dim = 0.06;
        const hot = a * a * 1.3;          // strongly active neurons flare toward white-gold
        arr[k2 * 3] = base[0] * (dim + a * 1.1) + HOT[0] * hot;
        arr[k2 * 3 + 1] = base[1] * (dim + a * 1.1) + HOT[1] * hot;
        arr[k2 * 3 + 2] = base[2] * (dim + a * 1.1) + HOT[2] * hot;
      }
      L.col.needsUpdate = true;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
