"""
Model Forge MCP server: lets Claude turn photos (or text) into 3D models with YOUR Model Forge AI server.

Tools
  image_to_3d     one photo  -> textured 3D model (.glb)
  photos_to_3d    front (+ back/left/right) photos -> more accurate model (Hunyuan3D-2mv)
  text_to_3d      description -> 3D model
  check_3d_job    resume waiting on a job that was still running
  preview_3d_model render any .glb from 4 angles so Claude can inspect it
  server_status   GPU / queue / features of the AI server

Every finished model comes back with a 4-view preview image, so Claude can check the shape and retry if needed.

Run
  Local (Claude Code / Claude Desktop):  python modelforge_mcp.py            (stdio)
  Remote (claude.ai custom connector):   python modelforge_mcp.py --http      (streamable HTTP)

Environment
  MODELFORGE_URL        AI server, e.g. http://1.2.3.4:8000 or https://1-2-3-4.sslip.io   (required)
  MODELFORGE_TOKEN      the server's API_TOKEN (if it has one)
  MODELFORGE_OUTPUT_DIR where .glb files are saved in local mode (default: ./models)
  MODELFORGE_PUBLIC_URL public base URL of the AI server, used for download links (default: MODELFORGE_URL)
  MCP_PORT / MCP_PATH   HTTP mode: listen port (default 8765) and URL path (default /mcp)
"""
import asyncio, base64, io, ipaddress, json, os, re, socket, sys, time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageOps
from mcp.server.fastmcp import Context, FastMCP, Image

REMOTE = "--http" in sys.argv
SERVER = os.environ.get("MODELFORGE_URL", "").rstrip("/")
TOKEN = os.environ.get("MODELFORGE_TOKEN", "")
PUBLIC = os.environ.get("MODELFORGE_PUBLIC_URL", SERVER).rstrip("/")
OUT_DIR = Path(os.environ.get("MODELFORGE_OUTPUT_DIR", "models")).expanduser().resolve()
MAX_IMAGE_BYTES = 25 * 1024 * 1024

INSTRUCTIONS = """Model Forge turns photos or text into 3D models (.glb) using the user's own GPU server (open-source Hunyuan3D-2).
- One photo: image_to_3d. Several photos of the same object (front/back/left/right): photos_to_3d - more accurate.
- Pass images as a local file path (e.g. an uploaded file's path), an https URL, or a data URL.
- Each result includes a 4-view preview: look at it. If the shape is wrong (missing parts, merged background, broken
  geometry), say so and suggest a better photo (single object, plain background, whole object in frame) or more views.
- Generation takes ~1-4 minutes (the first run after the server starts can take ~10). If a call returns status "running",
  call check_3d_job with the task_id.
- max_triangles guide: mobile game 8000, PC/console 30000, hero/cinematic 80000, 3D printing 150000 (texture=false)."""

kwargs = dict(instructions=INSTRUCTIONS)
if REMOTE:
    kwargs.update(host="0.0.0.0", port=int(os.environ.get("MCP_PORT", 8765)),
                  streamable_http_path=os.environ.get("MCP_PATH", "/mcp"), stateless_http=True)
    try:  # behind a reverse proxy the Host header is the public name; the secret URL path is the protection
        from mcp.server.transport_security import TransportSecuritySettings
        kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    except Exception:
        pass
mcp = FastMCP("model-forge", **kwargs)


# ----------------------------------------------------------------------------- helpers
def _headers():
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def _need_server():
    if not SERVER:
        raise ValueError("MODELFORGE_URL is not set. Point it at your Model Forge AI server, e.g. http://<ec2-ip>:8000")


async def _api(method, path, **kw):
    _need_server()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=120)) as c:
        r = await c.request(method, SERVER + path, headers=_headers(), **kw)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error") or r.json().get("detail")
        except Exception:
            msg = r.text[:300]
        hint = {401: " (wrong MODELFORGE_TOKEN)", 502: " (AI server not running)"}.get(r.status_code, "")
        raise RuntimeError(f"AI server error HTTP {r.status_code}: {msg}{hint}")
    return r


def _check_public_url(url):
    """Remote mode: refuse URLs that point into the server's own network (metadata service, localhost, VPC)."""
    host = urlparse(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve host {host!r}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("That URL points to a private network address and is not allowed.")


async def _load_image(src: str) -> str:
    """Accepts a file path, http(s) URL, data URL or raw base64. Returns a PNG data URL (max 1024 px, EXIF-rotated)."""
    src = (src or "").strip()
    if not src:
        raise ValueError("Empty image")
    if src.startswith("data:"):
        raw = base64.b64decode(src.split(",", 1)[-1])
    elif re.match(r"^https?://", src, re.I):
        if REMOTE:
            _check_public_url(src)
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            r = await c.get(src)
            r.raise_for_status()
            raw = r.content
    elif not REMOTE and Path(src).expanduser().exists():
        raw = Path(src).expanduser().read_bytes()
    elif re.fullmatch(r"[A-Za-z0-9+/=\s]{200,}", src):
        raw = base64.b64decode(src)
    else:
        if REMOTE:
            raise ValueError("In this remote connector, pass the image as an https URL or data URL (local file paths are not accessible).")
        raise ValueError(f"Image not found: {src!r} (give a file path, https URL or data URL)")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 25 MB")
    img = ImageOps.exif_transpose(PILImage.open(io.BytesIO(raw)))
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def _progress(ctx, p, msg):
    if ctx is None:
        return
    try:
        await ctx.report_progress(p, 100, msg)
    except TypeError:
        await ctx.report_progress(p, 100)
    except Exception:
        pass


async def _wait(task_id, ctx, wait_seconds):
    deadline = time.monotonic() + max(5, wait_seconds)
    last = None
    while True:
        s = (await _api("GET", f"/status/{task_id}")).json()
        if s.get("stage") != last:
            last = s.get("stage")
            await _progress(ctx, s.get("progress") or 0, last or s.get("status"))
        if s["status"] in ("succeeded", "failed") or time.monotonic() > deadline:
            return s
        await asyncio.sleep(3)


# ----------------------------------------------------------------------------- preview renderer (no GPU / OpenGL needed)
def _render_views(glb: bytes, size=384):
    import trimesh
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb")
    parts = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    V, F, C = [], [], []
    off = 0
    for m in parts:
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        try:
            vis = m.visual.to_color() if m.visual.kind == "texture" else m.visual
            fc = np.asarray(vis.face_colors)[:, :3].astype(float)
        except Exception:
            fc = np.full((len(m.faces), 3), 200.0)
        V.append(m.vertices); F.append(m.faces + off); C.append(fc); off += len(m.vertices)
    if not V:
        raise ValueError("Model has no triangles")
    V, F, C = np.vstack(V), np.vstack(F), np.vstack(C)
    lo, hi = V.min(0), V.max(0)
    dims = hi - lo
    V = (V - (lo + hi) / 2) / (np.linalg.norm(dims) / 2 or 1)
    light = np.array([0.45, 0.65, 1.0]); light /= np.linalg.norm(light)
    views = [("front", 0, 0), ("right", 90, 0), ("back", 180, 0), ("3/4 top", -40, 28)]
    tiles = []
    for name, yaw, pitch in views:
        a, b = np.radians(yaw), np.radians(pitch)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        P = V @ (Rx @ Ry).T
        tri = P[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        vis = n[:, 2] > 0
        shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        col = np.clip(C * shade[:, None], 0, 255).astype(np.uint8)
        order = np.argsort(tri[:, :, 2].mean(1))
        img = PILImage.new("RGB", (size, size), (34, 37, 46))
        d = ImageDraw.Draw(img)
        s = size * 0.46
        xy = np.stack([size / 2 + tri[:, :, 0] * s, size / 2 - tri[:, :, 1] * s], -1)
        for i in order:
            if vis[i]:
                d.polygon([tuple(p) for p in xy[i]], fill=tuple(col[i]))
        d.text((8, 6), name, fill=(200, 205, 220))
        tiles.append(img)
    grid = PILImage.new("RGB", (size * 2 + 6, size * 2 + 6), (14, 15, 20))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 2) * (size + 6), (i // 2) * (size + 6)))
    buf = io.BytesIO()
    grid.save(buf, "PNG")
    stats = {"triangles": int(len(F)), "vertices": int(len(V)),
             "proportions_w_h_d": [round(float(x / (dims.max() or 1)), 3) for x in dims]}
    return buf.getvalue(), stats


async def _finish(s, task_id, label):
    """Turn a server status into tool output: summary text + saved file + preview image."""
    if s["status"] == "failed":
        return f"❌ Generation failed: {s.get('error') or 'unknown error'} (task {task_id})"
    if s["status"] != "succeeded":
        return json.dumps({"status": "running", "task_id": task_id, "progress": s.get("progress"), "stage": s.get("stage"),
                           "next": "Call check_3d_job with this task_id to keep waiting."})
    glb = (await _api("GET", s["model_url"])).content
    info = {"status": "succeeded", "task_id": task_id, "download_url": PUBLIC + s["model_url"], "size_mb": round(len(glb) / 1e6, 2)}
    if not REMOTE:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "model"
        path = OUT_DIR / f"{name}-{task_id[:6]}.glb"
        path.write_bytes(glb)
        info["saved_to"] = str(path)
    out = []
    try:
        png, stats = await asyncio.to_thread(_render_views, glb)
        info.update(stats)
        out = [json.dumps(info, indent=1), Image(data=png, format="png")]
    except Exception as e:
        info["preview_error"] = str(e)
        out = [json.dumps(info, indent=1)]
    return out


async def _run(body, label, ctx, wait_seconds):
    task_id = (await _api("POST", "/generate", json=body)).json()["task_id"]
    await _progress(ctx, 1, "Queued on GPU server")
    s = await _wait(task_id, ctx, wait_seconds)
    return await _finish(s, task_id, label)


# ----------------------------------------------------------------------------- tools
@mcp.tool()
async def image_to_3d(image: str, texture: bool = True, max_triangles: int = 30000, wait_seconds: int = 300,
                      ctx: Context = None):
    """Create a 3D model (.glb) from ONE photo of an object.

    image: local file path, https URL, or data URL. Best: single object, centered, plain background, whole object visible.
    texture: paint colors/materials (False = grey shape, faster; good for 3D printing).
    max_triangles: polygon budget (8000 mobile game, 30000 PC, 80000 hero, 150000 printing).
    wait_seconds: how long to wait before returning a task_id to poll with check_3d_job."""
    body = {"mode": "image", "image": await _load_image(image), "texture": texture, "polycount": max_triangles}
    return await _run(body, Path(image).stem if not image.startswith(("data:", "http")) else "photo-model", ctx, wait_seconds)


@mcp.tool()
async def photos_to_3d(front: str, back: str = "", left: str = "", right: str = "", texture: bool = True,
                       max_triangles: int = 30000, wait_seconds: int = 300, ctx: Context = None):
    """Create a MORE ACCURATE 3D model from several photos of the same object (multi-view, Hunyuan3D-2mv).

    front is required; add back/left/right when available (all 4 is best). Photos should be taken at the same distance
    and height, turning the object ~90 degrees between shots. Each value: file path, https URL or data URL.
    Textures are painted from the front photo."""
    views = {"front": front, "back": back, "left": left, "right": right}
    images = {k: await _load_image(v) for k, v in views.items() if v}
    if len(images) < 2:
        raise ValueError("Give at least two views (front + back is the most useful), or use image_to_3d for a single photo.")
    body = {"mode": "multiview", "images": images, "texture": texture, "polycount": max_triangles}
    return await _run(body, f"multiview-{len(images)}", ctx, wait_seconds)


@mcp.tool()
async def text_to_3d(prompt: str, style: str = "", texture: bool = True, max_triangles: int = 30000,
                     wait_seconds: int = 300, ctx: Context = None):
    """Create a 3D model from a text description (the server first imagines an image, then builds 3D from it).

    prompt: describe ONE object, e.g. "a wooden treasure chest with gold trim". For characters add "full body, T-pose".
    style: optional, e.g. "low poly", "stylized hand-painted game art", "cartoon", "realistic"."""
    full = ", ".join(x for x in [prompt.strip(), style.strip()] if x)
    return await _run({"mode": "text", "prompt": full, "texture": texture, "polycount": max_triangles}, prompt, ctx, wait_seconds)


@mcp.tool()
async def check_3d_job(task_id: str, wait_seconds: int = 240, ctx: Context = None):
    """Check on (and keep waiting for) a generation that returned status "running"."""
    s = await _wait(task_id, ctx, wait_seconds)
    return await _finish(s, task_id, f"model-{task_id[:6]}")


@mcp.tool()
async def preview_3d_model(model: str):
    """Render a .glb model (file path or https URL) from 4 angles and return the image plus triangle count and proportions."""
    if re.match(r"^https?://", model, re.I):
        if REMOTE:
            _check_public_url(model)
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            r = await c.get(model)
            r.raise_for_status()
            glb = r.content
    elif not REMOTE:
        glb = Path(model).expanduser().read_bytes()
    else:
        raise ValueError("Pass an https URL to the .glb")
    png, stats = await asyncio.to_thread(_render_views, glb)
    return [json.dumps(stats), Image(data=png, format="png")]


@mcp.tool()
async def server_status():
    """Show whether the Model Forge AI server is reachable, its GPU, queue length and features."""
    return (await _api("GET", "/health")).json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if REMOTE else "stdio")
