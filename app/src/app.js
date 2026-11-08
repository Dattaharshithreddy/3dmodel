import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter } from 'three/addons/exporters/OBJExporter.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { ReelStudio } from './reels.js';

const $ = s => document.querySelector(s);
const Cap = window.Capacitor;
const NATIVE = !!Cap?.isNativePlatform?.();
const state = { mode: 'text', image: null, current: null, model: null, polling: null };

// =============================================================== settings
const SKEY = 'model-forge-settings';
const settings = (() => { try { return JSON.parse(localStorage.getItem(SKEY)) || {}; } catch { return {}; } })();
const saveSettings = () => { try { localStorage.setItem(SKEY, JSON.stringify(settings)); } catch {} };
const serverBase = () => (settings.server || '').trim().replace(/\/+$/, '');

// =============================================================== networking
// On Android we use the native HTTP stack (no CORS limits, works with any server).
const b64ToBuf = b64 => { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; };
const bufToB64 = buf => { const u = new Uint8Array(buf); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };

async function http(url, { method = 'GET', headers = {}, json, binary } = {}) {
  if (json !== undefined) headers = { ...headers, 'Content-Type': 'application/json' };
  if (NATIVE) {
    const r = await Cap.Plugins.CapacitorHttp.request({ url, method, headers, data: json,
      responseType: binary ? 'arraybuffer' : 'json', connectTimeout: 30000, readTimeout: 120000 });
    let data = r.data;
    if (r.status >= 400) {
      if (typeof data === 'string') { try { data = JSON.parse(data); } catch {} }
      throw new Error(`HTTP ${r.status}: ${data?.detail || data?.error || data?.message || (typeof data === 'string' ? data.slice(0, 200) : '')}`);
    }
    if (binary) return typeof data === 'string' ? b64ToBuf(data) : data;
    if (typeof data === 'string') { try { data = JSON.parse(data); } catch {} }
    return data;
  }
  const r = await fetch(url, { method, headers, body: json !== undefined ? JSON.stringify(json) : undefined });
  if (!r.ok) {
    const t = await r.text(); let d; try { d = JSON.parse(t); } catch {}
    throw new Error(`HTTP ${r.status}: ${d?.detail || d?.error || d?.message || t.slice(0, 200)}`);
  }
  return binary ? r.arrayBuffer() : r.json();
}

// =============================================================== engines
// Each engine: create(req) -> taskId ; status(taskId) -> {status, progress, stage, modelUrl, thumbnail, canRefine, error}
const authH = () => settings.token ? { Authorization: `Bearer ${settings.token}` } : {};
const ENGINES = {
  server: {
    label: 'My AI server (Hunyuan3D-2)',
    ready: () => !!serverBase(),
    async create(req) {
      if (req.mode === 'refine') throw new Error('Your server textures models directly — tick "Generate textures" and generate again.');
      const d = await http(`${serverBase()}/generate`, { method: 'POST', headers: authH(),
        json: { mode: req.mode, prompt: req.prompt, image: req.image, texture: req.texture, polycount: req.polycount } });
      return d.task_id;
    },
    async status(id) {
      const d = await http(`${serverBase()}/status/${id}`, { headers: authH() });
      return { status: d.status, progress: d.progress, stage: d.stage, error: d.error,
        modelUrl: d.model_url && serverBase() + d.model_url, thumbnail: d.thumbnail_url && serverBase() + d.thumbnail_url };
    },
    headers: authH,
  },
  meshy: {
    label: 'Meshy (paid API)',
    ready: () => !!settings.meshy,
    h: () => ({ Authorization: `Bearer ${settings.meshy}` }),
    async create(req) {
      const B = 'https://api.meshy.ai/openapi';
      if (req.mode === 'image') {
        const d = await http(`${B}/v1/image-to-3d`, { method: 'POST', headers: this.h(),
          json: { image_url: req.image, should_texture: req.texture, enable_pbr: req.texture, should_remesh: true, target_polycount: req.polycount } });
        return `image:${d.result}`;
      }
      const body = req.mode === 'refine'
        ? { mode: 'refine', preview_task_id: req.previewTaskId.replace(/^text:/, ''), enable_pbr: true }
        : { mode: 'preview', prompt: req.prompt, art_style: 'realistic', should_remesh: true, target_polycount: req.polycount };
      const d = await http(`${B}/v2/text-to-3d`, { method: 'POST', headers: this.h(), json: body });
      return `text:${d.result}`;
    },
    async status(id) {
      const [kind, raw] = id.split(':');
      const d = await http(kind === 'text' ? `https://api.meshy.ai/openapi/v2/text-to-3d/${raw}` : `https://api.meshy.ai/openapi/v1/image-to-3d/${raw}`, { headers: this.h() });
      const map = { PENDING: 'queued', IN_PROGRESS: 'running', SUCCEEDED: 'succeeded' };
      return { status: map[d.status] || 'failed', progress: d.progress || 0, modelUrl: d.model_urls?.glb, thumbnail: d.thumbnail_url,
        error: d.task_error?.message || (map[d.status] ? undefined : d.status), canRefine: kind === 'text' && d.mode === 'preview' };
    },
  },
  tripo: {
    label: 'Tripo (paid API, text only)',
    ready: () => !!settings.tripo,
    h: () => ({ Authorization: `Bearer ${settings.tripo}` }),
    async create(req) {
      if (req.mode === 'image') throw new Error('Tripo engine supports text prompts only in this app.');
      const body = req.mode === 'refine' ? { type: 'refine_model', draft_model_task_id: req.previewTaskId }
        : { type: 'text_to_model', prompt: req.prompt, face_limit: req.polycount, texture: req.texture };
      const d = await http('https://api.tripo3d.ai/v2/openapi/task', { method: 'POST', headers: this.h(), json: body });
      if (d.code !== 0) throw new Error(d.message || 'Tripo error');
      return d.data.task_id;
    },
    async status(id) {
      const d = (await http(`https://api.tripo3d.ai/v2/openapi/task/${id}`, { headers: this.h() })).data;
      const map = { queued: 'queued', running: 'running', success: 'succeeded' }, o = d.output || {};
      return { status: map[d.status] || 'failed', progress: d.progress || 0, modelUrl: o.pbr_model || o.model || o.base_model,
        thumbnail: o.rendered_image, error: map[d.status] ? undefined : d.status, canRefine: d.type === 'text_to_model' };
    },
  },
  demo: {
    label: 'Demo (offline shapes)',
    ready: () => true,
    tasks: {},
    async create(req) { const id = 'demo-' + Date.now().toString(36); this.tasks[id] = { t: Date.now() }; return id; },
    async status(id) {
      const t = this.tasks[id]; const p = t ? Math.min(100, Math.round((Date.now() - t.t) / 40)) : 100;
      return { status: p >= 100 ? 'succeeded' : 'running', progress: p, stage: 'Building demo shape', demo: true };
    },
  },
};

function renderEngines(preferBest = false) {
  const sel = $('#provider'), prev = preferBest ? null : sel.value;
  sel.innerHTML = Object.entries(ENGINES).map(([k, e]) =>
    `<option value="${k}" ${e.ready() ? '' : 'disabled'}>${e.label}${e.ready() ? '' : ' — set up in ⚙'}</option>`).join('');
  sel.value = prev && ENGINES[prev]?.ready() ? prev : Object.keys(ENGINES).find(k => ENGINES[k].ready());
}

// =============================================================== 3D viewer
const container = $('#viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f14);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
camera.position.set(2.5, 1.6, 2.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(3, 5, 2); scene.add(sun);
const grid = new THREE.GridHelper(4, 20, 0x3a3f50, 0x222632); scene.add(grid);
const resize = () => { const { clientWidth: w, clientHeight: h } = container; if (!w || !h) return; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); };
new ResizeObserver(resize).observe(container); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

function frame(obj) {
  const box = new THREE.Box3().setFromObject(obj), size = box.getSize(new THREE.Vector3()).length() || 1, c = box.getCenter(new THREE.Vector3());
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.7, 0.45, 0.7).normalize().multiplyScalar(size * 1.4));
  camera.near = size / 100; camera.far = size * 100; camera.updateProjectionMatrix();
  grid.position.y = box.min.y;
}
function showModel(obj) {
  if (state.model) scene.remove(state.model);
  const box = new THREE.Box3().setFromObject(obj);
  obj.scale.multiplyScalar(2 / (box.getSize(new THREE.Vector3()).length() || 1));
  box.setFromObject(obj);
  obj.position.sub(box.getCenter(new THREE.Vector3())).add(new THREE.Vector3(0, box.getSize(new THREE.Vector3()).y / 2, 0));
  state.model = obj; scene.add(obj); frame(obj); applyWire(); $('#empty').hidden = true;
  let tris = 0, verts = 0;
  obj.traverse(o => { if (o.isMesh) { const g = o.geometry; verts += g.attributes.position.count; tris += (g.index ? g.index.count : g.attributes.position.count) / 3; } });
  $('#stats').innerHTML = `${Math.round(tris).toLocaleString()} triangles<br>${verts.toLocaleString()} vertices`;
}
let wire = false;
const applyWire = () => state.model?.traverse(o => { if (o.isMesh) [].concat(o.material).forEach(m => m.wireframe = wire); });
document.querySelectorAll('.toolbar [data-act]').forEach(b => b.onclick = () => {
  const a = b.dataset.act;
  if (a === 'rotate') { controls.autoRotate = !controls.autoRotate; b.classList.toggle('on', controls.autoRotate); }
  if (a === 'wire') { wire = !wire; b.classList.toggle('on', wire); applyWire(); }
  if (a === 'grid') { grid.visible = !grid.visible; b.classList.toggle('on', grid.visible); }
  if (a === 'reset' && state.model) frame(state.model);
});

// =============================================================== export / save
const slug = s => (s || 'model').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'model';
async function saveFile(data, name, mime) {
  if (NATIVE) {
    const { Filesystem, Share } = Cap.Plugins;
    const b64 = typeof data === 'string' ? btoa(unescape(encodeURIComponent(data))) : bufToB64(data);
    const { uri } = await Filesystem.writeFile({ path: name, data: b64, directory: 'CACHE' });
    await Share.share({ title: name, url: uri, dialogTitle: 'Save or share your 3D model' });
    return;
  }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type: mime })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
// Game-ready copy: real-world height in meters, pivot at the feet (bottom-center), chosen up-axis.
function exportable() {
  const root = new THREE.Group(), inner = new THREE.Group(), copy = state.model.clone(true);
  copy.position.set(0, 0, 0); copy.rotation.set(0, 0, 0); copy.scale.copy(state.model.scale);
  inner.add(copy); root.add(inner); root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(copy), size = box.getSize(new THREE.Vector3());
  copy.position.set(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
  const h = Math.max(0.01, Number($('#height').value) || 1);
  inner.scale.setScalar(h / (size.y || 1));
  if ($('#upaxis').value === 'z') root.rotation.x = Math.PI / 2;
  root.name = slug(state.current?.prompt); root.updateMatrixWorld(true);
  return root;
}
document.querySelectorAll('[data-export]').forEach(b => b.onclick = async () => {
  if (!state.model) return showError('Nothing to export yet.');
  const fmt = b.dataset.export, name = slug(state.current?.prompt), obj = exportable();
  try {
    if (fmt === 'glb') await saveFile(await new GLTFExporter().parseAsync(obj, { binary: true }), `${name}.glb`, 'model/gltf-binary');
    else if (fmt === 'obj') await saveFile(new OBJExporter().parse(obj), `${name}.obj`, 'text/plain');
    else await saveFile(new STLExporter().parse(obj, { binary: true }).buffer, `${name}.stl`, 'model/stl');
  } catch (e) { if (!/cancel/i.test(e.message)) showError('Export failed: ' + e.message); }
});

// =============================================================== demo shapes (offline)
function hash(str) { let h = 2166136261; for (const c of str) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; }
function rng(seed) { return () => ((seed = Math.imul(seed ^ (seed >>> 15), seed | 1) ^ (seed + Math.imul(seed ^ (seed >>> 7), seed | 61))) >>> 0) / 4294967296; }
function demoModel(prompt) {
  const r = rng(hash(prompt)), p = prompt.toLowerCase(), hue = r();
  const mat = (h = hue, l = 0.55) => new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(h, 0.6, l), roughness: 0.45 });
  const g = new THREE.Group();
  if (/tree|plant|pine/.test(p)) {
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 0.6, 10), mat(0.08, 0.3)); trunk.position.y = 0.3; g.add(trunk);
    for (let i = 0; i < 3; i++) { const c = new THREE.Mesh(new THREE.ConeGeometry(0.5 - i * 0.12, 0.6, 12), mat(0.3, 0.35)); c.position.y = 0.7 + i * 0.3; g.add(c); }
  } else if (/vase|cup|bottle|pot|bowl/.test(p)) {
    const pts = []; for (let i = 0; i <= 24; i++) { const t = i / 24; pts.push(new THREE.Vector2(0.25 + 0.2 * Math.sin(t * Math.PI) + 0.05 * Math.sin(t * 9 * r()), t * 1.2)); }
    g.add(new THREE.Mesh(new THREE.LatheGeometry(pts, 64), Object.assign(mat(), { side: THREE.DoubleSide })));
  } else if (/house|building|tower|castle|home/.test(p)) {
    const base = new THREE.Mesh(new THREE.BoxGeometry(1, 0.7, 0.8), mat(0.1, 0.7)); base.position.y = 0.35; g.add(base);
    const roof = new THREE.Mesh(new THREE.ConeGeometry(0.8, 0.5, 4), mat(0.0, 0.4)); roof.position.y = 0.95; roof.rotation.y = Math.PI / 4; g.add(roof);
  } else {
    const geo = mergeVertices(new THREE.IcosahedronGeometry(0.8, 40)), pos = geo.attributes.position, v = new THREE.Vector3();
    const f = [1 + r() * 3, 2 + r() * 4, 4 + r() * 6], a = [0.18, 0.08, 0.03], ph = [r() * 6, r() * 6, r() * 6];
    for (let i = 0; i < pos.count; i++) {
      const n = v.fromBufferAttribute(pos, i).normalize(); let d = 1;
      for (let k = 0; k < 3; k++) d += a[k] * Math.sin(n.x * f[k] + ph[k]) * Math.sin(n.y * f[k] * 1.3 + ph[k]) * Math.sin(n.z * f[k] * 0.7 + ph[k]);
      pos.setXYZ(i, n.x * 0.8 * d, n.y * 0.8 * d, n.z * 0.8 * d);
    }
    geo.computeVertexNormals(); g.add(new THREE.Mesh(geo, mat()));
  }
  return g;
}

// =============================================================== inputs
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
  state.mode = t.dataset.mode;
  $('#text-input').hidden = state.mode !== 'text';
  $('#image-input').hidden = state.mode !== 'image';
});
const setPoly = n => { $('#poly').value = n; $('#poly-val').textContent = Number(n).toLocaleString(); };
$('#poly').oninput = e => { setPoly(e.target.value); $('#usage').value = 'custom'; };
const USAGE = { mobile: { poly: 8000, tex: true }, pc: { poly: 30000, tex: true }, hero: { poly: 80000, tex: true }, print: { poly: 150000, tex: false } };
$('#usage').onchange = e => { const u = USAGE[e.target.value]; if (u) { setPoly(u.poly); $('#texture').checked = u.tex; } };
const drop = $('#drop'), file = $('#file');
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); loadImage(e.dataTransfer.files[0]); };
file.onchange = () => loadImage(file.files[0]);
function loadImage(f) {
  if (!f || !f.type.startsWith('image/')) return showError('Please choose an image.');
  // Re-encode to PNG at max 1024px: keeps uploads small and handles HEIC/large camera photos.
  const url = URL.createObjectURL(f), img = new Image();
  img.onload = () => {
    const s = Math.min(1, 1024 / Math.max(img.width, img.height)), c = document.createElement('canvas');
    c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    state.image = c.toDataURL('image/png');
    $('#preview').src = state.image; $('#preview').hidden = false; $('#drop-hint').hidden = true; URL.revokeObjectURL(url);
  };
  img.onerror = () => showError('Could not read that image.');
  img.src = url;
}

// =============================================================== generation
const showError = msg => { $('#error').textContent = msg || ''; $('#error').hidden = !msg; };

async function startJob(req, label) {
  showError(''); $('#generate').disabled = true; $('#refine').hidden = true;
  try {
    const id = await ENGINES[req.provider].create(req);
    const job = { id, provider: req.provider, prompt: label, mode: req.mode, created: Date.now(), status: 'queued',
      thumb: req.mode === 'image' ? await tinyThumb(req.image) : null };
    saveHistory(job); poll(job);
  } catch (e) { showError(e.message); $('#generate').disabled = false; }
}
$('#generate').onclick = () => {
  const provider = $('#provider').value;
  if (!provider) return openSettings();
  const base = { provider, texture: $('#texture').checked, polycount: Number($('#poly').value) };
  if (state.mode === 'text') {
    const prompt = $('#prompt').value.trim();
    if (!prompt) return showError('Type a prompt first.');
    const art = $('#art').value, game = /mobile|pc/.test($('#usage').value) ? 'game asset' : '';
    startJob({ ...base, mode: 'text', prompt: [prompt, art, game].filter(Boolean).join(', ') }, prompt);
  } else {
    if (!state.image) return showError('Pick a photo first.');
    startJob({ ...base, mode: 'image', image: state.image }, 'Photo → 3D');
  }
};
$('#refine').onclick = () => { const c = state.current; if (c) startJob({ provider: c.provider, mode: 'refine', previewTaskId: c.id }, `${c.prompt} (textured)`); };

function poll(job) {
  clearTimeout(state.polling); state.current = job; $('#job').hidden = false;
  let failures = 0;
  const tick = async () => {
    try {
      const s = await ENGINES[job.provider].status(job.id);
      failures = 0;
      $('#job-status').textContent = s.stage || { queued: 'Queued…', running: 'Generating…', succeeded: 'Done', failed: 'Failed' }[s.status] || s.status;
      $('#job-pct').textContent = `${Math.round(s.progress || 0)}%`; $('#job-bar').style.width = `${s.progress || 0}%`;
      if (s.status === 'succeeded') {
        Object.assign(job, { status: 'succeeded', modelUrl: s.modelUrl, demo: s.demo, canRefine: s.canRefine, thumb: job.thumb || s.thumbnail });
        saveHistory(job); $('#generate').disabled = false; openJob(job); return;
      }
      if (s.status === 'failed') { job.status = 'failed'; saveHistory(job); $('#generate').disabled = false; return showError(s.error || 'Generation failed'); }
    } catch (e) {
      if (++failures >= 5) { $('#generate').disabled = false; return showError(e.message); }
    }
    state.polling = setTimeout(tick, job.provider === 'demo' ? 400 : 3000);
  };
  tick();
}

async function openJob(job) {
  state.current = job; showError('');
  $('#refine').hidden = !job.canRefine;
  if (job.status !== 'succeeded') return poll(job);
  if (job.demo) { showModel(demoModel(job.prompt)); $('#job-status').textContent = 'Done'; return; }
  try {
    $('#job-status').textContent = 'Downloading model…';
    const buf = await http(job.modelUrl, { binary: true, headers: job.provider === 'server' ? authH() : {} });
    const gltf = await new GLTFLoader().parseAsync(buf, '');
    showModel(gltf.scene); $('#job-status').textContent = 'Done';
  } catch (e) { showError(`Could not load model (cloud links expire, and Colab servers stop when closed): ${e.message}`); }
}

// =============================================================== history
const HKEY = 'model-forge-history';
const getHistory = () => { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } };
function saveHistory(job) {
  const h = getHistory().filter(j => j.id !== job.id); h.unshift(job);
  try { localStorage.setItem(HKEY, JSON.stringify(h.slice(0, 50))); } catch {}
  renderHistory();
}
function renderHistory() {
  const ul = $('#history'); ul.innerHTML = '';
  const h = getHistory();
  if (!h.length) { ul.innerHTML = '<li class="s" style="cursor:default">No models yet</li>'; return; }
  for (const j of h) {
    const li = document.createElement('li');
    li.innerHTML = `${j.thumb ? '<img alt="">' : '<div class="ph">◇</div>'}
      <div class="meta"><div class="t"></div><div class="s"></div></div><button class="del" title="Remove">✕</button>`;
    if (j.thumb) li.querySelector('img').src = j.thumb;
    li.querySelector('.t').textContent = j.prompt;
    li.querySelector('.s').textContent = `${ENGINES[j.provider]?.label.split(' (')[0] || j.provider} · ${j.status} · ${new Date(j.created).toLocaleDateString()}`;
    li.onclick = () => openJob(j);
    li.querySelector('.del').onclick = e => { e.stopPropagation(); try { localStorage.setItem(HKEY, JSON.stringify(getHistory().filter(x => x.id !== j.id))); } catch {} renderHistory(); };
    ul.appendChild(li);
  }
}
function tinyThumb(src) {
  return new Promise(res => { const img = new Image(); img.onload = () => { const c = document.createElement('canvas'); c.width = c.height = 80;
    const s = Math.max(80 / img.width, 80 / img.height); c.getContext('2d').drawImage(img, (80 - img.width * s) / 2, (80 - img.height * s) / 2, img.width * s, img.height * s);
    res(c.toDataURL('image/jpeg', 0.7)); }; img.onerror = () => res(null); img.src = src; });
}

// =============================================================== settings dialog
const dlg = $('#settings');
function openSettings() {
  $('#s-server').value = settings.server || ''; $('#s-token').value = settings.token || '';
  $('#s-meshy').value = settings.meshy || ''; $('#s-tripo').value = settings.tripo || '';
  $('#s-result').textContent = ''; dlg.showModal();
}
$('#open-settings').onclick = openSettings;
dlg.addEventListener('close', () => {
  if (dlg.returnValue !== 'save') return;
  Object.assign(settings, { server: $('#s-server').value.trim(), token: $('#s-token').value.trim(),
    meshy: $('#s-meshy').value.trim(), tripo: $('#s-tripo').value.trim() });
  saveSettings(); renderEngines(true);
});
$('#s-test').onclick = async () => {
  const out = $('#s-result'), url = $('#s-server').value.trim().replace(/\/+$/, '');
  if (!url) { out.textContent = 'Enter the server URL first.'; out.className = 'small error'; return; }
  out.textContent = 'Connecting…'; out.className = 'small';
  try {
    const d = await http(`${url}/health`);
    out.textContent = `✓ Connected — GPU: ${d.gpu || 'none'}, textures: ${d.texture ? 'on' : 'off'}`; out.className = 'small ok';
  } catch (e) { out.textContent = '✗ ' + e.message; out.className = 'small error'; }
};

// =============================================================== reel studio
const reelDlg = $('#reel');
let studio, previewRaf = null, recording = false;
const reelOpts = () => ({ format: $('#r-format').value, look: $('#r-look').value, motion: $('#r-motion').value, bg: $('#r-bg').value,
  duration: Number($('#r-dur').value), title: $('#r-title').value.trim(), subtitle: $('#r-sub').value.trim() });
function reelPreview() {
  cancelAnimationFrame(previewRaf);
  studio.setup(state.model, reelOpts());
  const t0 = performance.now();
  const loop = () => {
    if (!reelDlg.open || recording) return;
    const el = (performance.now() - t0) / 1000, d = studio.opts.duration;
    studio.frame((el % d) / d, el); previewRaf = requestAnimationFrame(loop);
  };
  loop();
}
$('#open-reel').onclick = () => {
  if (!state.model) return showError('Generate or open a model first, then make a reel.');
  studio ||= new ReelStudio($('#reel-canvas'));
  if (!$('#r-title').value && state.current?.prompt) $('#r-title').value = state.current.prompt.slice(0, 40);
  $('#r-job').hidden = true; reelDlg.showModal(); reelPreview();
};
reelDlg.querySelectorAll('select, input').forEach(el => el.addEventListener('change', () => !recording && reelPreview()));
reelDlg.addEventListener('cancel', e => { if (recording) e.preventDefault(); });
$('#r-record').onclick = async () => {
  if (recording) return;
  recording = true; cancelAnimationFrame(previewRaf);
  $('#r-record').disabled = $('#r-close').disabled = true; $('#r-job').hidden = false; $('#r-status').textContent = 'Recording…';
  try {
    studio.setup(state.model, reelOpts());
    const blob = await studio.record(t => { $('#r-pct').textContent = `${Math.round(t * 100)}%`; $('#r-bar').style.width = `${t * 100}%`; });
    $('#r-status').textContent = 'Saving…';
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
    await saveFile(await blob.arrayBuffer(), `${slug(studio.opts.title || state.current?.prompt)}-reel.${ext}`, blob.type);
    $('#r-status').textContent = ext === 'mp4' ? 'Saved ✓' : 'Saved ✓ (WebM — CapCut/Instagram accept it; convert to MP4 if an app refuses)';
  } catch (e) { if (!/cancel/i.test(e.message)) $('#r-status').textContent = 'Failed: ' + e.message; }
  recording = false; $('#r-record').disabled = $('#r-close').disabled = false; reelPreview();
};

renderEngines(); renderHistory();
