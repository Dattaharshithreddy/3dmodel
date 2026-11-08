// Copies src/ and the parts of three.js the app uses into www/ (no bundler needed).
import fs from 'node:fs';
import path from 'node:path';
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const www = path.join(root, 'www');
fs.rmSync(www, { recursive: true, force: true });
fs.cpSync(path.join(root, 'src'), www, { recursive: true });
const three = path.join(root, 'node_modules', 'three');
const dst = path.join(www, 'vendor', 'three');
fs.mkdirSync(path.join(dst, 'build'), { recursive: true });
fs.copyFileSync(path.join(three, 'build', 'three.module.js'), path.join(dst, 'build', 'three.module.js'));
for (const dir of ['controls', 'loaders', 'exporters', 'environments', 'utils', 'libs'])
  fs.cpSync(path.join(three, 'examples', 'jsm', dir), path.join(dst, 'examples', 'jsm', dir), { recursive: true });
// drop big things we never use
for (const junk of ['libs/draco', 'libs/basis', 'libs/ammo.wasm.js', 'libs/ammo.wasm.wasm', 'libs/rhino3dm', 'libs/opentype.module.js', 'libs/lottie_canvas.module.js'])
  fs.rmSync(path.join(dst, 'examples', 'jsm', junk), { recursive: true, force: true });
console.log('Built www/');
