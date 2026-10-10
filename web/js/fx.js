// Combat effects driven by sim events, timed in game seconds.
import * as THREE from 'three';
import { toLocal } from './arena.js';

const SPEED = 14; // projectile tiles / game-second (visual only)
const sphere = new THREE.SphereGeometry(1, 16, 12);
const cyl = new THREE.CylinderGeometry(1, 1, 1, 8);
const cone = new THREE.ConeGeometry(1, 1, 8);
const ring = new THREE.RingGeometry(0.85, 1, 48);
const dodeca = new THREE.DodecahedronGeometry(1, 0);

function basic(color, opacity = 1, additive = false) {
  return new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}

export class Effects {
  constructor(world, A) {
    this.world = world;
    this.A = A;
    this.items = [];
  }

  clear() {
    for (const it of this.items) this.world.remove(it.obj);
    this.items = [];
  }

  add(obj, t0, dur, update) {
    obj.visible = false;
    this.world.add(obj);
    this.items.push({ obj, t0, dur, update });
  }

  projectile(t0, style, x, y, h0, tx, ty, h1 = 0.7) {
    const from = toLocal(this.A, x, y, h0);
    const to = toLocal(this.A, tx, ty, h1);
    const dist = from.distanceTo(to);
    const dur = Math.max(0.07, dist / SPEED);
    const arc = style === 'cannon' ? 1.2 : style === 'arrow' ? 0.6 : 0.15;
    let obj;
    if (style === 'arrow') {
      obj = new THREE.Group();
      const shaft = new THREE.Mesh(cyl, new THREE.MeshStandardMaterial({ color: 0x7a5230 }));
      shaft.scale.set(0.03, 0.7, 0.03);
      shaft.rotation.x = Math.PI / 2;
      const tip = new THREE.Mesh(cone, new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.6 }));
      tip.scale.set(0.07, 0.16, 0.07);
      tip.rotation.x = Math.PI / 2;
      tip.position.z = 0.4;
      obj.add(shaft, tip);
    } else if (style === 'cannon') {
      obj = new THREE.Mesh(sphere, new THREE.MeshStandardMaterial({ color: 0x2b2b30, metalness: 0.5, roughness: 0.3 }));
      obj.scale.setScalar(0.24);
      obj.castShadow = true;
    } else if (style === 'bullet') {
      obj = new THREE.Mesh(sphere, basic(0xffe27a, 1, true));
      obj.scale.set(0.09, 0.09, 0.35);
    } else {
      obj = new THREE.Mesh(sphere, basic(0xc58bff, 0.95, true));
      obj.scale.setScalar(0.17);
    }
    const prev = new THREE.Vector3();
    this.add(obj, t0, dur, (p) => {
      prev.copy(obj.position);
      obj.position.lerpVectors(from, to, p);
      obj.position.y += Math.sin(p * Math.PI) * arc;
      if (p > 0.02 && style !== 'cannon') {
        const ahead = obj.position.clone().multiplyScalar(2).sub(prev);   // a point further along the path
        obj.lookAt(this.world.localToWorld(ahead));
      }
      if (p >= 1) this.spark(t0 + dur, tx, ty, h1);
    });
  }

  spark(t0, x, y, h) {
    const m = new THREE.Mesh(sphere, basic(0xfff6c8, 0.9, true));
    const at = toLocal(this.A, x, y, h);
    this.add(m, t0, 0.18, (p) => {
      m.position.copy(at);
      m.scale.setScalar(0.12 + p * 0.35);
      m.material.opacity = 0.9 * (1 - p);
    });
  }

  slash(t0, x, y, big) {
    const m = new THREE.Mesh(ring, basic(0xffffff, 0.9, true));
    const at = toLocal(this.A, x, y, 0.6);
    const s = big ? 0.9 : 0.6;
    this.add(m, t0, 0.22, (p) => {
      m.position.copy(at);
      m.rotation.set(-Math.PI / 2 + 0.6, p * 2.5, 0);
      m.scale.setScalar(s * (0.6 + p * 0.6));
      m.material.opacity = 0.9 * (1 - p);
    });
  }

  fireball(t0, dur, x, y, tx, ty) {
    const g = new THREE.Group();
    const core = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ color: 0xffd27a }));
    core.scale.setScalar(0.42);
    const glow = new THREE.Mesh(sphere, basic(0xff6a1a, 0.55, true));
    glow.scale.setScalar(0.8);
    const light = new THREE.PointLight(0xff8a30, 25, 10, 2);
    g.add(core, glow, light);
    const from = toLocal(this.A, x, y, 4.5);
    const to = toLocal(this.A, tx, ty, 0.3);
    const trail = [];
    const trailGroup = new THREE.Group();
    for (let i = 0; i < 8; i++) {
      const tm = new THREE.Mesh(sphere, basic(i < 3 ? 0xffb040 : 0xff5a10, 0.6, true));
      trailGroup.add(tm);
      trail.push(tm);
    }
    this.add(trailGroup, t0, dur, () => {});
    this.add(g, t0, dur, (p) => {
      g.position.lerpVectors(from, to, p);
      g.position.y += Math.sin(p * Math.PI) * 5;
      glow.scale.setScalar(0.8 + Math.sin(p * 40) * 0.08);
      trail.forEach((tm, i) => {
        const q = Math.max(0, p - (i + 1) * 0.025);
        tm.position.lerpVectors(from, to, q);
        tm.position.y += Math.sin(q * Math.PI) * 5;
        tm.scale.setScalar(0.38 * (1 - i / 9));
        tm.material.opacity = 0.5 * (1 - i / 8);
      });
    });
  }

  explosion(t0, x, y, radius, big = false) {
    const at = toLocal(this.A, x, y, 0.2);
    const ball = new THREE.Mesh(sphere, basic(0xffa040, 0.85, true));
    const core = new THREE.Mesh(sphere, basic(0xfff0b0, 0.9, true));
    const shock = new THREE.Mesh(ring, basic(0xffe6a0, 0.8, true));
    const light = new THREE.PointLight(0xff9040, 60, radius * 5, 2);
    const g = new THREE.Group();
    g.add(ball, core, shock, light);
    shock.rotation.x = -Math.PI / 2;
    const dur = big ? 0.9 : 0.55;
    this.add(g, t0, dur, (p) => {
      g.position.copy(at);
      const e = 1 - Math.pow(1 - p, 3);
      ball.scale.setScalar(radius * (0.3 + 0.7 * e));
      ball.scale.y *= 0.7;
      ball.material.opacity = 0.85 * (1 - p);
      core.scale.setScalar(radius * 0.45 * (1 - p));
      shock.scale.setScalar(radius * (0.4 + 1.1 * e));
      shock.material.opacity = 0.8 * (1 - p);
      light.intensity = 60 * (1 - p);
    });
    // smoke puffs rising after the flash
    for (let i = 0; i < (big ? 10 : 5); i++) {
      const a = (i / (big ? 10 : 5)) * Math.PI * 2;
      const s = new THREE.Mesh(sphere, new THREE.MeshStandardMaterial({ color: 0x777777, transparent: true, opacity: 0.6, depthWrite: false }));
      const off = new THREE.Vector3(Math.cos(a) * radius * 0.5, 0, Math.sin(a) * radius * 0.5);
      this.add(s, t0 + 0.1, 1.1, (p) => {
        s.position.copy(at).add(off).addScaledVector(off, p * 0.6);
        s.position.y = 0.4 + p * 1.6;
        s.scale.setScalar(radius * 0.25 * (0.6 + p));
        s.material.opacity = 0.55 * (1 - p);
      });
    }
    if (big) this.debris(t0, x, y);
  }

  debris(t0, x, y) {
    const at = toLocal(this.A, x, y, 1.5);
    const m = new THREE.MeshStandardMaterial({ color: 0x8d8f99, roughness: 0.9 });
    for (let i = 0; i < 12; i++) {
      const d = new THREE.Mesh(dodeca, m);
      d.castShadow = true;
      const v = new THREE.Vector3((Math.random() - 0.5) * 6, 4 + Math.random() * 4, (Math.random() - 0.5) * 6);
      const s = 0.15 + Math.random() * 0.25;
      this.add(d, t0, 1.0, (p) => {
        const t = p * 1.0;
        d.position.copy(at).addScaledVector(v, t);
        d.position.y = Math.max(0.1, at.y + v.y * t - 12 * t * t);
        d.rotation.set(t * 8, t * 6, 0);
        d.scale.setScalar(s * (1 - p * 0.5));
      });
    }
  }

  puff(t0, x, y, h = 0.5) {
    const at = toLocal(this.A, x, y, h);
    for (let i = 0; i < 5; i++) {
      const m = new THREE.Mesh(sphere, new THREE.MeshStandardMaterial({ color: 0xf2f2f6, transparent: true, opacity: 0.85, depthWrite: false }));
      const off = new THREE.Vector3((Math.random() - 0.5) * 0.7, Math.random() * 0.4, (Math.random() - 0.5) * 0.7);
      this.add(m, t0, 0.5, (p) => {
        m.position.copy(at).addScaledVector(off, 1 + p * 1.5);
        m.scale.setScalar(0.18 + p * 0.35);
        m.material.opacity = 0.85 * (1 - p);
      });
    }
  }

  deployRing(t0, x, y, radius, color) {
    const m = new THREE.Mesh(ring, basic(color, 0.8, true));
    const at = toLocal(this.A, x, y, 0.04);
    m.rotation.x = -Math.PI / 2;
    this.add(m, t0, 0.5, (p) => {
      m.position.copy(at);
      m.scale.setScalar(radius * (0.4 + p * 1.2));
      m.material.opacity = 0.8 * (1 - p);
    });
  }

  update(now) {
    const keep = [];
    for (const it of this.items) {
      const p = (now - it.t0) / it.dur;
      if (p < 0) { keep.push(it); continue; }
      it.obj.visible = true;
      it.update(Math.min(1, p));
      if (p >= 1) {
        this.world.remove(it.obj);
        it.obj.traverse((o) => { if (o.isMesh && o.material.dispose) o.material.dispose(); });
      } else {
        keep.push(it);
      }
    }
    this.items = keep;
  }
}
