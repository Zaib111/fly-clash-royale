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
    this.camera.position.set(0, 0.35, 3.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.7;
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 6;

    const pos = new Float32Array(buffer, 0, count * 3);
    const group = new Uint8Array(buffer, count * 12, count);
    this.count = count;
    this.group = group;
    this.level = new Float32Array(count);    // displayed activity (smoothed)
    this.target = new Float32Array(count);   // latest activity from the server
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
        size: big ? 0.045 : 0.016, map: tex, vertexColors: true, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
      });
      const pts = new THREE.Points(geo, material);
      this.scene.add(pts);
      this.layers.push({ idx: Int32Array.from(idx), col });
    }
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setActivity(b64) {
    const bin = atob(b64);
    const n = Math.min(bin.length, this.count);
    for (let i = 0; i < n; i++) this.target[i] = bin.charCodeAt(i) / 255;
  }

  update(dt) {
    const k = Math.min(1, dt * 7);
    const lv = this.level, tg = this.target, grp = this.group;
    for (let i = 0; i < this.count; i++) lv[i] += (tg[i] - lv[i]) * k;
    for (const L of this.layers) {
      const arr = L.col.array;
      const idx = L.idx;
      for (let k2 = 0; k2 < idx.length; k2++) {
        const i = idx[k2];
        const a = Math.min(1, lv[i] * 1.6);
        const base = GROUP_COLORS[grp[i]];
        const dim = 0.1;
        arr[k2 * 3] = base[0] * (dim + a * 0.9) + HOT[0] * a * a * 0.8;
        arr[k2 * 3 + 1] = base[1] * (dim + a * 0.9) + HOT[1] * a * a * 0.8;
        arr[k2 * 3 + 2] = base[2] * (dim + a * 0.9) + HOT[2] * a * a * 0.8;
      }
      L.col.needsUpdate = true;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
