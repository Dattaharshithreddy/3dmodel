// Reel studio: renders the current model into a short vertical/square/wide video
// with different looks (3D, toon, pixel, sketch, hologram) and camera moves.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const BGS = {
  studio: ['#1b1f2a', '#07080b'], sunset: ['#ff9a5a', '#7b2ff7'], neon: ['#2b0f4c', '#0a0218'],
  ocean: ['#3fd0ff', '#0b3a8a'], white: ['#ffffff', '#e6e8ee'], green: ['#00b140', '#00b140'],
};
const ease = { out3: t => 1 - Math.pow(1 - t, 3), inOut: t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  bounce: t => { const n = 7.5625, d = 2.75; if (t < 1 / d) return n * t * t; if (t < 2 / d) return n * (t -= 1.5 / d) * t + .75; if (t < 2.5 / d) return n * (t -= 2.25 / d) * t + .9375; return n * (t -= 2.625 / d) * t + .984375; } };
const clamp01 = x => Math.max(0, Math.min(1, x));

function toonGradient(steps) {
  const data = new Uint8Array(steps); for (let i = 0; i < steps; i++) data[i] = Math.round(40 + (215 * i) / (steps - 1));
  const tex = new THREE.DataTexture(data, steps, 1, THREE.RedFormat);
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true; return tex;
}
const outlineMaterial = (thickness, color) => new THREE.ShaderMaterial({
  uniforms: { thick: { value: thickness }, col: { value: new THREE.Color(color) } },
  vertexShader: 'uniform float thick; void main(){ vec3 p = position + normal * thick; gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }',
  fragmentShader: 'uniform vec3 col; void main(){ gl_FragColor = vec4(col, 1.0); }',
  side: THREE.BackSide,
});
const holoMaterial = () => new THREE.ShaderMaterial({
  uniforms: { col: { value: new THREE.Color('#38e8ff') } },
  vertexShader: 'varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }',
  fragmentShader: 'uniform vec3 col; varying vec3 vN; varying vec3 vV; void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.0); gl_FragColor = vec4(col * (0.2 + f * 1.4), 0.15 + f * 0.85); }',
  transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
});

export class ReelStudio {
  constructor(outCanvas) {
    this.out = outCanvas; this.ctx = outCanvas.getContext('2d');
    this.gl = document.createElement('canvas');
    this.renderer = new THREE.WebGLRenderer({ canvas: this.gl, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);
    this.envTex = new THREE.PMREMGenerator(this.renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 1000);
    this.opts = {};
  }

  setup(model, opts) {
    this.opts = opts;
    const [W, H] = opts.format.split('x').map(Number);
    this.W = W; this.H = H;
    this.out.width = W; this.out.height = H;
    this.px = opts.look === 'pixel' ? Math.max(6, Math.round(W / 130)) : 1;
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(Math.round(W / this.px), Math.round(H / this.px), false);
    this.renderer.toneMapping = opts.look === 'real' ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    this.camera.aspect = W / H; this.camera.updateProjectionMatrix();

    const scene = this.scene = new THREE.Scene();
    if (opts.look === 'real') scene.environment = this.envTex;
    this.amb = new THREE.AmbientLight(0xffffff, opts.look === 'real' ? 0.2 : 0.9); scene.add(this.amb);
    this.key = new THREE.DirectionalLight(0xffffff, opts.look === 'real' ? 1.6 : 1.8); this.key.position.set(2, 4, 3); scene.add(this.key);
    const rim = new THREE.DirectionalLight(0x9db8ff, 0.8); rim.position.set(-3, 2, -3); scene.add(rim);

    const clone = model.clone(true);
    const box = new THREE.Box3().setFromObject(clone);
    this.size = box.getSize(new THREE.Vector3());
    this.R = this.size.length() / 2 || 1;
    this.centerY = box.getCenter(new THREE.Vector3()).y;
    this.pivot = new THREE.Group(); this.pivot.add(clone); scene.add(this.pivot);
    clone.updateMatrixWorld(true);

    const look = opts.look;
    const grad = toonGradient(look === 'pixel' ? 4 : look === 'sketch' ? 2 : 3);
    const meshes = []; clone.traverse(o => { if (o.isMesh) meshes.push(o); });
    for (const m of meshes) {
      if (look === 'real') continue;
      const orig = [].concat(m.material)[0] || {};
      if (!m.geometry.attributes.normal) { m.geometry = m.geometry.clone(); m.geometry.computeVertexNormals(); }
      if (look === 'holo') { m.material = holoMaterial(); continue; }
      m.material = new THREE.MeshToonMaterial({
        color: look === 'sketch' ? new THREE.Color('#ffffff') : (orig.color?.clone() || new THREE.Color('#ffffff')),
        map: look === 'sketch' ? null : orig.map || null, vertexColors: !!orig.vertexColors && look !== 'sketch', gradientMap: grad,
      });
      const ws = m.getWorldScale(new THREE.Vector3()).x || 1;
      const thick = (look === 'sketch' ? 0.012 : look === 'pixel' ? 0.012 : 0.008) * this.R * 2 / ws;
      const hull = new THREE.Mesh(m.geometry, outlineMaterial(thick, look === 'sketch' ? '#1d1b18' : '#0b0b12'));
      m.add(hull);
    }
    // camera distance that fits the model's bounding sphere in the narrowest field of view
    const v = THREE.MathUtils.degToRad(this.camera.fov), h = 2 * Math.atan(Math.tan(v / 2) * this.camera.aspect);
    this.fitDist = this.R / Math.sin(Math.min(v, h) / 2) * 1.02;
  }

  // t: 0..1 through the clip
  frame(t, time = t * (this.opts.duration || 8)) {
    const { look, motion } = this.opts, R = this.R, cam = this.camera, p = this.pivot;
    let ang = 0, dist = this.fitDist * 0.82, elev = 0.18, lift = 0, keyI = 1, fade = 1;
    p.rotation.set(0, 0, 0); p.position.set(0, 0, 0);
    if (motion === 'turntable') { p.rotation.y = t * Math.PI * 2; }
    else if (motion === 'orbit') { ang = -0.6 + ease.inOut(t) * Math.PI * 1.3; dist = this.fitDist * (1.15 - 0.4 * ease.inOut(t)); elev = 0.35 - 0.2 * t; }
    else if (motion === 'reveal') { keyI = ease.out3(clamp01(t / 0.45)); fade = clamp01(t / 0.25); p.rotation.y = -0.9 + 1.3 * ease.out3(t); dist = this.fitDist * (1.1 - 0.3 * ease.out3(t)); elev = 0.05 + 0.12 * t; }
    else if (motion === 'float') { lift = (0.06 + 0.04 * Math.sin(time * 2.1)) * R; p.rotation.y = Math.sin(t * Math.PI * 2) * 0.7; p.rotation.z = Math.sin(time * 1.3) * 0.04; }
    else if (motion === 'drop') { const d = clamp01(t / 0.35); lift = (1 - ease.bounce(d)) * R * 2.2; p.rotation.y = ease.out3(clamp01((t - 0.3) / 0.7)) * Math.PI * 1.1; }
    p.position.y = lift;
    this.key.intensity = (look === 'real' ? 1.6 : 1.8) * keyI; this.amb.intensity = (look === 'real' ? 0.2 : 0.9) * (0.25 + 0.75 * keyI);
    const aimUp = (this.H > this.W ? 0.22 : 0.12) * R, target = new THREE.Vector3(0, this.centerY + aimUp, 0);
    cam.position.set(Math.sin(ang) * dist * Math.cos(elev), target.y + Math.sin(elev) * dist, Math.cos(ang) * dist * Math.cos(elev));
    cam.lookAt(target);
    this.renderer.render(this.scene, cam);
    this.composite(t, time, fade, lift);
  }

  composite(t, time, fade, lift) {
    const { ctx, W, H } = this, o = this.opts, s = W / 1080;
    // background
    let bg = BGS[o.bg] || BGS.studio;
    if (o.look === 'sketch') bg = ['#f6f1e6', '#ebe3d2'];
    if (o.look === 'holo' && o.bg !== 'green') bg = ['#04121c', '#000307'];
    const g = ctx.createLinearGradient(0, 0, W * 0.3, H); g.addColorStop(0, bg[0]); g.addColorStop(1, bg[1]);
    ctx.globalAlpha = 1; ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    if (o.bg !== 'green' && o.look !== 'sketch') {
      const glow = ctx.createRadialGradient(W / 2, H * 0.55, 0, W / 2, H * 0.55, Math.max(W, H) * 0.6);
      glow.addColorStop(0, 'rgba(255,255,255,0.10)'); glow.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);
    }
    // contact shadow
    if (o.bg !== 'green') {
      const c = new THREE.Vector3(0, 0, 0).project(this.camera), e = new THREE.Vector3(this.size.x * 0.55, 0, 0).project(this.camera);
      const cx = (c.x + 1) / 2 * W, cy = (1 - c.y) / 2 * H, rx = Math.max(10, Math.abs(e.x - c.x) / 2 * W), k = 1 / (1 + lift / this.R * 1.5);
      const sh = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx * k);
      sh.addColorStop(0, `rgba(0,0,0,${0.45 * k})`); sh.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save(); ctx.translate(cx, cy); ctx.scale(1, 0.28); ctx.translate(-cx, -cy); ctx.fillStyle = sh; ctx.beginPath(); ctx.arc(cx, cy, rx * k, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
    // 3D layer
    ctx.imageSmoothingEnabled = this.px === 1;
    ctx.drawImage(this.gl, 0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    // look-specific overlays
    if (o.look === 'holo') {
      ctx.fillStyle = 'rgba(56,232,255,0.06)';
      for (let y = (time * 60) % 6; y < H; y += 6 * s) ctx.fillRect(0, y, W, 2 * s);
      if (Math.sin(time * 37) > 0.97) { ctx.fillStyle = 'rgba(56,232,255,0.08)'; ctx.fillRect(0, 0, W, H); }
    }
    if (o.look === 'sketch') { ctx.strokeStyle = 'rgba(60,50,40,0.05)'; ctx.lineWidth = 1; for (let i = 0; i < 60; i++) { const y = (i * 97.13) % H; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y + 30 * s); ctx.stroke(); } }
    if (o.look === 'pixel') { ctx.fillStyle = 'rgba(0,0,0,0.12)'; for (let y = 0; y < H; y += this.px * 2) ctx.fillRect(0, y, W, this.px * 0.6); }
    // fade (reveal)
    if (fade < 1) { ctx.fillStyle = `rgba(0,0,0,${1 - fade})`; ctx.fillRect(0, 0, W, H); }
    // captions
    const dark = o.bg === 'white' || o.look === 'sketch';
    const a = clamp01((t - 0.08) / 0.12) * (o.loopFade ? clamp01((1 - t) / 0.06) : 1);
    const top = H > W ? H * 0.11 : H * 0.12;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const font = o.look === 'pixel' ? '"Courier New", monospace' : 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
    if (o.title) {
      ctx.globalAlpha = a; ctx.font = `800 ${Math.round(84 * s)}px ${font}`;
      ctx.shadowColor = dark ? 'transparent' : 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 24 * s;
      ctx.fillStyle = dark ? '#16161a' : '#ffffff';
      wrap(ctx, o.title, W / 2, top - (1 - a) * 30 * s, W * 0.86, 92 * s);
    }
    if (o.subtitle) {
      ctx.globalAlpha = clamp01((t - 0.15) / 0.12); ctx.font = `500 ${Math.round(42 * s)}px ${font}`;
      ctx.fillStyle = dark ? '#4a4a55' : 'rgba(255,255,255,0.85)';
      wrap(ctx, o.subtitle, W / 2, top + (o.title ? 120 * s * Math.ceil(ctx.measureText(o.title).width / (W * 0.86) + 0.01) : 0), W * 0.86, 50 * s);
    }
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  }

  pickMime() {
    const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    return types.find(m => window.MediaRecorder?.isTypeSupported?.(m)) || '';
  }

  record(onProgress) {
    return new Promise((resolve, reject) => {
      if (!window.MediaRecorder || !this.out.captureStream) return reject(new Error('Video recording is not supported on this device.'));
      const mime = this.pickMime(), dur = (this.opts.duration || 8) * 1000;
      const stream = this.out.captureStream(30);
      const rec = new MediaRecorder(stream, { mimeType: mime || undefined, videoBitsPerSecond: 10_000_000 });
      const chunks = [];
      rec.ondataavailable = e => e.data.size && chunks.push(e.data);
      rec.onstop = () => resolve(new Blob(chunks, { type: (mime || 'video/webm').split(';')[0] }));
      rec.onerror = e => reject(e.error || new Error('Recording failed'));
      this.frame(0);
      rec.start(250);
      const t0 = performance.now();
      this.recording = true;
      const step = () => {
        const el = performance.now() - t0, t = Math.min(1, el / dur);
        this.frame(t, el / 1000); onProgress?.(t);
        if (t < 1) requestAnimationFrame(step); else { this.recording = false; setTimeout(() => rec.stop(), 150); }
      };
      requestAnimationFrame(step);
    });
  }
}

function wrap(ctx, text, x, y, maxW, lineH) {
  const words = text.split(/\s+/); let line = '', yy = y;
  for (const w of words) {
    const test = line ? line + ' ' + w : w;
    if (ctx.measureText(test).width > maxW && line) { ctx.fillText(line, x, yy); line = w; yy += lineH; } else line = test;
  }
  if (line) ctx.fillText(line, x, yy);
}
