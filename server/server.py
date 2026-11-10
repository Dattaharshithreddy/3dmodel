"""
Model Forge AI server — your own text-to-3D / image-to-3D engine.

Runs Tencent's open-source Hunyuan3D-2 models on your GPU:
  text  -> image (HunyuanDiT) -> 3D shape (Hunyuan3D-DiT) -> texture (Hunyuan3D-Paint)
  image ->                       3D shape                  -> texture

API (used by the Model Forge app):
  GET  /health                   -> {"ok": true, "texture": bool, ...}
  POST /generate                 {mode: "text"|"image"|"multiview", prompt?, image? (data URL),
                                  images? ({front, left?, back?, right?} data URLs), texture?, polycount?}
                                 -> {"task_id": "..."}
  GET  /status/{task_id}         -> {status: queued|running|succeeded|failed, progress, stage, model_url?, error?}
  GET  /models/{task_id}.glb     -> the model file

Environment variables:
  SHAPE_MODEL   default "tencent/Hunyuan3D-2mini"   (or "tencent/Hunyuan3D-2" for higher quality, more VRAM)
  SHAPE_SUBFOLDER default "hunyuan3d-dit-v2-mini" (use "hunyuan3d-dit-v2-0" with tencent/Hunyuan3D-2)
  MV_MODEL      default "tencent/Hunyuan3D-2mv"   multi-view model (front/left/back/right photos)
  MV_SUBFOLDER  default "hunyuan3d-dit-v2-mv"     ("hunyuan3d-dit-v2-mv-turbo" = faster, slightly less detail)
  IMAGE_MODEL   default "stabilityai/stable-diffusion-xl-base-1.0"  (2D art mode; any SDXL model id works)
  TEXTURE       "1" to enable texture painting (default: on if the texture module is installed)
  LOW_VRAM      "1" to offload models to CPU between steps (slower, fits 12-16 GB GPUs)
  API_TOKEN     if set, clients must send "Authorization: Bearer <token>"
  PORT          default 8000
"""
import base64, io, json, os, threading, time, traceback, uuid, queue
from urllib.parse import parse_qs
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

OUT = Path(os.environ.get("OUTPUT_DIR", Path(__file__).parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
SHAPE_MODEL = os.environ.get("SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
SHAPE_SUBFOLDER = os.environ.get("SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-mini")
TEXTURE_MODEL = os.environ.get("TEXTURE_MODEL", "tencent/Hunyuan3D-2")
MV_MODEL = os.environ.get("MV_MODEL", "tencent/Hunyuan3D-2mv")
MV_SUBFOLDER = os.environ.get("MV_SUBFOLDER", "hunyuan3d-dit-v2-mv")
MV_VIEWS = ("front", "left", "back", "right")
T2I_MODEL = os.environ.get("T2I_MODEL", "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")   # 2D art generator
IMAGE_VAE = os.environ.get("IMAGE_VAE", "madebyollin/sdxl-vae-fp16-fix")
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"
API_TOKEN = os.environ.get("API_TOKEN", "")


# ----------------------------------------------------------------------------- models (lazy)
def _model_cached(repo, subfolder=None):
    """True if the model is already on disk. hy3dgen first looks in $HY3DGEN_MODELS (default ~/.cache/hy3dgen) and
    otherwise downloads into the Hugging Face cache (~/.cache/huggingface/hub/models--org--name/snapshots/<rev>/)."""
    import glob
    base = os.path.expanduser(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen"))
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub")
    candidates = [os.path.join(base, repo, subfolder or "")]
    candidates += glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*", subfolder or ""))
    return any(os.path.isdir(c) and any(os.scandir(c)) for c in candidates)


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.on_status = lambda msg: None   # set per job: shows "Downloading…" in the app
        self._shape = self._mv = self._paint = self._t2i = self._rembg = self._img = None
        try:  # textures need the compiled CUDA rasterizer (its kernel links against torch: import torch first)
            import torch  # noqa: F401
            import custom_rasterizer  # noqa: F401
            import hy3dgen.texgen  # noqa: F401
            default_tex = "1"
        except Exception:
            default_tex = "0"
        self.texture_available = os.environ.get("TEXTURE", default_tex) == "1"

    def _free(self, attr):
        if LOW_VRAM and getattr(self, attr) is not None:
            setattr(self, attr, None)
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def image2d(self):
        """SDXL text-to-image for 2D art mode."""
        if self._img is None:
            for other in ("_shape", "_mv", "_paint"):
                self._free(other)
            import torch
            from diffusers import AutoencoderKL, StableDiffusionXLPipeline
            from huggingface_hub import try_to_load_from_cache
            if not isinstance(try_to_load_from_cache(IMAGE_MODEL, "model_index.json"), str):
                self.on_status("Downloading 2D art AI model (first time only, ~7 GB)")
            vae = AutoencoderKL.from_pretrained(IMAGE_VAE, torch_dtype=torch.float16)
            pipe = StableDiffusionXLPipeline.from_pretrained(IMAGE_MODEL, vae=vae, torch_dtype=torch.float16,
                                                             variant="fp16", use_safetensors=True)
            from diffusers import StableDiffusionXLImg2ImgPipeline
            redraw = StableDiffusionXLImg2ImgPipeline(**pipe.components)   # shares the same weights (photo -> 2D art)
            if not LOW_VRAM:
                pipe.to("cuda")
            self._img = {"draw": pipe, "redraw": redraw}
        return self._img

    def art_pipe(self, kind):
        p = self.image2d()[kind]
        if LOW_VRAM:
            p.enable_model_cpu_offload()   # re-attaches the offload hooks to whichever of the two is used
        return p

    def multiview(self):
        if self._mv is None:
            self._free("_img")
            self._free("_shape")  # on small GPUs keep only one shape model in memory
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self._free("_paint")
            if not _model_cached(MV_MODEL, MV_SUBFOLDER):
                self.on_status("Downloading multi-view AI model (first time only, ~5–10 min)")
            self._mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                MV_MODEL, subfolder=MV_SUBFOLDER, use_safetensors=True)
        return self._on_gpu(self._mv)

    def shape(self):
        if self._shape is None:
            self._free("_mv")
            self._free("_paint")
            self._free("_img")
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            if not _model_cached(SHAPE_MODEL, SHAPE_SUBFOLDER):
                self.on_status("Downloading 3D shape AI model (first time only, ~5 min)")
            self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=True)
        return self._on_gpu(self._shape)

    # Hunyuan3D's shape pipeline has a broken enable_model_cpu_offload (it uses a missing .components), so on
    # small GPUs we park the shape model in CPU RAM while textures are painted and bring it back afterwards.
    def _on_gpu(self, pipe):
        import torch
        if torch.cuda.is_available() and getattr(pipe, "device", None) is not None and pipe.device.type != "cuda":
            pipe.to("cuda")
        return pipe

    def park_shape_models(self):
        if not LOW_VRAM:
            return
        import gc, torch
        for p in (self._shape, self._mv):
            if p is not None and hasattr(p, "to"):
                p.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def paint(self):
        if self._paint is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            if not _model_cached(TEXTURE_MODEL, "hunyuan3d-paint-v2-0"):
                self.on_status("Downloading texture AI models (first time only, ~5–10 min)")
            self._paint = Hunyuan3DPaintPipeline.from_pretrained(TEXTURE_MODEL)
            if LOW_VRAM and hasattr(self._paint, "enable_model_cpu_offload"):
                self._paint.enable_model_cpu_offload()
        return self._paint

    def t2i(self):
        if self._t2i is None:
            from hy3dgen.text2image import HunyuanDiTPipeline
            self._t2i = HunyuanDiTPipeline(T2I_MODEL)
        return self._t2i

    def rembg(self):
        if self._rembg is None:
            from hy3dgen.rembg import BackgroundRemover
            self._rembg = BackgroundRemover()
        return self._rembg


engine = Engine()
tasks: dict[str, dict] = {}
jobs: "queue.Queue[str]" = queue.Queue()
TASKS_FILE = OUT / "tasks.json"
_save_lock = threading.Lock()


def save_tasks():
    """Keep job status on disk so finished models survive a server restart."""
    with _save_lock:
        data = {k: {kk: vv for kk, vv in v.items() if kk != "req"} for k, v in tasks.items()}
        tmp = TASKS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(TASKS_FILE)


def load_tasks():
    if not TASKS_FILE.exists():
        return
    try:
        data = json.loads(TASKS_FILE.read_text())
    except Exception:
        return
    for k, v in data.items():
        if v.get("status") in ("queued", "running"):
            v.update(status="failed", error="The server restarted while working on this. Tap Generate again — "
                                            "finished steps (the 3D shape) are reused, so it continues faster.")
        tasks[k] = v


def fingerprint(req) -> str:
    """Same inputs + settings -> same id (texture on/off excluded: the shape is shared)."""
    import hashlib
    key = {k: req.get(k) for k in ("mode", "prompt", "image", "images", "polycount", "steps", "seed", "quality",
                                       "style", "aspect", "transparent", "negative", "strength")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]


# Shape quality presets: more diffusion steps + a finer 3D grid = sharper details (slower, more GPU memory).
QUALITY = {
    "fast":  {"steps": 20, "octree": 256, "chunks": 20000},
    "high":  {"steps": 30, "octree": 384, "chunks": 20000},
    "ultra": {"steps": 50, "octree": 512, "chunks": 8000},
}


# 2D art presets
ART_STYLES = {
    "": "",
    "realistic": "photorealistic, ultra detailed, sharp focus, professional photography, natural lighting",
    "3d": "3D render, Pixar style character, soft studio lighting, octane render, highly detailed",
    "anime": "anime style, clean line art, vibrant colors, detailed illustration",
    "cartoon": "cartoon style, bold outlines, flat vibrant colors, vector illustration",
    "pixel": "pixel art, 16-bit game sprite, crisp pixels, limited color palette",
    "concept": "concept art, digital painting, dramatic lighting, highly detailed, artstation quality",
    "sticker": "sticker design, thick white outline, flat colors, simple cute illustration",
    "icon": "game icon, centered single object, glossy, highly detailed, dark simple background",
}
ART_SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}
ART_STEPS = {"fast": 20, "high": 30, "ultra": 45}
ART_NEGATIVE = "blurry, low quality, lowres, jpeg artifacts, deformed, extra limbs, bad anatomy, watermark, text, signature, cropped"


def make_2d(t, req, task_id):
    import torch
    w, h = ART_SIZES.get(req.get("aspect") or "square", ART_SIZES["square"])
    style = ART_STYLES.get(req.get("style") or "", "")
    prompt = ", ".join(x for x in [req["prompt"].strip(), style,
                                   "isolated on plain white background, full view" if req.get("transparent") else ""] if x)
    negative = ", ".join(x for x in [ART_NEGATIVE, (req.get("negative") or "").strip()] if x)
    set_stage(t, "Loading 2D art AI", 5)
    steps = ART_STEPS.get(req.get("quality") or "high", 30)
    gen = torch.Generator("cpu").manual_seed(int(req.get("seed") or 1234))
    src = None
    if req.get("image"):
        # Photo -> 2D art: redraw the uploaded picture in the chosen style, keeping its layout.
        src = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(req["image"].split(",", 1)[-1]))))
        if src.mode in ("RGBA", "LA", "P"):
            rgba = src.convert("RGBA")
            src = Image.new("RGB", rgba.size, "white")
            src.paste(rgba, mask=rgba.split()[3])
        src = src.convert("RGB")
        scale = (1024 * 1024 / (src.width * src.height)) ** 0.5      # ~1 megapixel, same proportions as the photo
        w = max(512, min(1536, round(src.width * scale / 64) * 64))
        h = max(512, min(1536, round(src.height * scale / 64) * 64))
        src = src.resize((w, h), Image.LANCZOS)
    strength = min(0.95, max(0.15, float(req.get("strength") or 0.68)))
    total = max(1, int(steps * strength)) if src is not None else steps
    pipe = engine.art_pipe("redraw" if src is not None else "draw")
    set_stage(t, "Redrawing your photo" if src is not None else "Drawing", 15)

    def on_step(p, i, ts, kw):
        t.update(progress=15 + int(70 * min(1, (i + 1) / total)))
        return kw

    if src is not None:
        image = pipe(prompt=prompt, negative_prompt=negative, image=src, strength=strength, num_inference_steps=steps,
                     guidance_scale=8.0, generator=gen, callback_on_step_end=on_step).images[0]
    else:
        image = pipe(prompt=prompt, negative_prompt=negative, width=w, height=h, num_inference_steps=steps,
                     guidance_scale=6.5, generator=gen, callback_on_step_end=on_step).images[0]
    if req.get("transparent"):
        set_stage(t, "Removing background", 90)
        image = engine.rembg()(image.convert("RGBA"))
    set_stage(t, "Saving", 95)
    image.save(OUT / f"{task_id}.png")
    thumb = image.convert("RGBA").copy()
    thumb.thumbnail((256, 256))
    thumb.save(OUT / f"{task_id}_input.png")
    engine._free("_img")   # LOW_VRAM: make room for 3D models again
    t.update(status="succeeded", progress=100, stage="Done", image_url=f"/images/{task_id}.png",
             thumbnail_url=f"/inputs/{task_id}.png", textured=bool(req.get("texture", True)))


def matte(mesh):
    """Hunyuan's texture export leaves metallicFactor unset, and glTF treats unset as 100% metal, which makes skin and
    cloth look washed-out and mirror-like in viewers and game engines. Make the material non-metallic."""
    try:
        v = getattr(mesh, "visual", None)
        if getattr(v, "kind", None) == "texture" and getattr(v, "material", None) is not None:
            pbr = v.material.to_pbr() if hasattr(v.material, "to_pbr") else v.material
            pbr.metallicFactor = 0.0
            pbr.roughnessFactor = 0.85
            v.material = pbr
    except Exception:
        traceback.print_exc()
    return mesh


def set_stage(t, stage, progress):
    t.update(stage=stage, progress=progress, status="running")
    save_tasks()


def run(task_id: str):
    import torch
    t = tasks[task_id]
    req = t["req"]
    engine.on_status = lambda msg: (t.update(stage=msg), save_tasks())
    try:
        def decode(data_url):
            return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1])))

        def cutout(img):
            img = img.convert("RGBA")
            return engine.rembg()(img) if img.getextrema()[3][0] == 255 else img  # skip if already transparent

        gen = torch.manual_seed(int(req.get("seed") or 1234))
        q = QUALITY.get(req.get("quality") or "high", QUALITY["high"])
        steps = int(req.get("steps") or q["steps"])
        if req["mode"] == "2d":
            make_2d(t, req, task_id)
            return
        fp = t["fp"]
        shape_file, cut_file = OUT / f"shape_{fp}.glb", OUT / f"cut_{fp}.png"
        mesh = None
        if shape_file.exists() and cut_file.exists():
            # Resume: the 3D shape for exactly these inputs was already made -> skip straight to textures.
            set_stage(t, "Reusing the finished 3D shape", 55)
            import trimesh
            mesh = trimesh.load(str(shape_file), force="mesh")
            image = Image.open(cut_file)
            Image.open(cut_file).save(OUT / f"{task_id}_input.png")

        if mesh is None:
            # 1. input image(s)
            if req["mode"] == "text":
                set_stage(t, "Imagining the object (text → image)", 5)
                prompt = req["prompt"].strip() + ", single object, centered, white background, 3D render, full view"
                image = engine.t2i()(prompt)
                if LOW_VRAM:
                    engine._t2i = None
                    torch.cuda.empty_cache()
            elif req["mode"] == "multiview":
                views = {k: decode(v) for k, v in (req.get("images") or {}).items() if k in MV_VIEWS and v}
                image = views["front"]
            else:
                image = decode(req["image"])
            image.save(OUT / f"{task_id}_input.png")

            set_stage(t, "Removing background", 15)
            if req["mode"] == "multiview":
                views = {k: cutout(v) for k, v in views.items()}
                image = views["front"]
                # 2. geometry from all views
                set_stage(t, f"Sculpting 3D shape from {len(views)} photos", 25)
                mesh = engine.multiview()(image=views, num_inference_steps=steps, octree_resolution=q["octree"],
                                          num_chunks=q["chunks"], generator=gen, output_type="trimesh")[0]
            else:
                image = cutout(image)
                # 2. geometry
                set_stage(t, "Sculpting 3D shape", 25)
                mesh = engine.shape()(image=image, num_inference_steps=steps, octree_resolution=q["octree"],
                                      num_chunks=q["chunks"], generator=gen)[0]

            set_stage(t, "Cleaning mesh", 55)
            from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=int(req.get("polycount") or 40000))

            # save the finished shape + cut-out photo so a retry can resume from here
            mesh.export(str(shape_file))
            image.save(cut_file)

        # 3. texture (if it fails, still deliver the grey shape instead of losing the whole job)
        warning = None
        if req.get("texture", True) and engine.texture_available:
            set_stage(t, "Painting textures", 65)
            engine.park_shape_models()
            try:
                mesh = engine.paint()(mesh, image=image)
                engine._free("_paint")   # LOW_VRAM: free RAM for the next job's shape model (reload takes ~1 min)
            except Exception as e:
                traceback.print_exc()
                warning = f"Textures failed ({str(e)[:200]}); here is the untextured shape. Tap Generate to retry — the shape is reused."
        elif req.get("texture", True):
            warning = "Textures are not available on this server; the model is untextured."

        set_stage(t, "Saving", 95)
        matte(mesh).export(str(OUT / f"{task_id}.glb"))
        t.update(status="succeeded", progress=100, stage="Done", model_url=f"/models/{task_id}.glb",
                 thumbnail_url=f"/inputs/{task_id}.png", textured=warning is None and bool(req.get("texture", True)))
        if warning:
            t["warning"] = warning
    except Exception as e:
        traceback.print_exc()
        t.update(status="failed", error=str(e)[:500])
    finally:
        t.pop("req", None)
        save_tasks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def worker():
    while True:
        tid = jobs.get()
        with engine.lock:  # one GPU job at a time
            run(tid)


load_tasks()
threading.Thread(target=worker, daemon=True).start()

# ----------------------------------------------------------------------------- API
app = FastAPI(title="Model Forge AI server")


@app.middleware("http")
async def auth(request: Request, call_next):
    if API_TOKEN and request.method != "OPTIONS" and request.url.path != "/health" \
            and not request.url.path.startswith(("/models/", "/inputs/", "/images/")):  # unguessable ids act as share links
        if request.headers.get("authorization") != f"Bearer {API_TOKEN}":
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        return await call_next(request)
    except Exception as e:  # keep CORS headers on crashes so the app can show the message
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"Server error: {e}"}, status_code=500)


# Added last = outermost, so every response (401s and errors included) carries CORS headers.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def bad(msg):
    raise HTTPException(400, msg)


def to_int(v, name, default):
    if v in (None, ""):
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        bad(f"{name} must be a number, got {v!r:.40}")


def to_bool(v, default=True):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(v)


async def read_body(request: Request) -> dict:
    """Accept JSON (also double-encoded JSON) or form-encoded bodies, whatever the client sends."""
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    body = None
    try:
        body = json.loads(raw or b"{}")
        if isinstance(body, str):  # double-encoded
            body = json.loads(body)
    except Exception:
        if "form" in ctype:
            body = {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace")).items()}
    if not isinstance(body, dict):
        bad(f"Expected a JSON object in the request body (content-type: {ctype or 'none'}, {len(raw)} bytes)")
    if isinstance(body.get("images"), str):
        try:
            body["images"] = json.loads(body["images"])
        except Exception:
            bad("images must be an object like {front, left, back, right}")
    return body


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse({"error": str(exc.detail), "detail": str(exc.detail)}, status_code=exc.status_code)


@app.get("/health")
def health():
    import torch
    return {"ok": True, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "texture": engine.texture_available, "shape_model": SHAPE_MODEL, "multiview": True, "mv_model": MV_MODEL, "qualities": list(QUALITY), "art2d": True,
            "queue": jobs.qsize(), "auth": bool(API_TOKEN), "version": 2}


@app.post("/generate")
async def generate(request: Request):
    b = await read_body(request)
    print("generate request:", {k: (type(v).__name__, len(v) if isinstance(v, (str, dict)) else v) for k, v in b.items()}, flush=True)
    mode = str(b.get("mode") or "").strip().lower()
    if mode not in ("text", "image", "multiview", "2d"):
        bad(f"mode must be text, image, multiview or 2d (got {b.get('mode')!r})")
    req = {"mode": mode, "prompt": str(b.get("prompt") or "").strip(), "image": b.get("image"),
           "texture": to_bool(b.get("texture")), "polycount": to_int(b.get("polycount"), "polycount", 40000),
           "steps": to_int(b.get("steps"), "steps", None), "seed": to_int(b.get("seed"), "seed", None),
           "quality": str(b.get("quality") or "high").lower()}
    if req["quality"] not in QUALITY:
        bad("quality must be fast, high or ultra")
    if mode == "2d":
        req.update(style=str(b.get("style") or "").lower(), aspect=str(b.get("aspect") or "square").lower(),
                   transparent=to_bool(b.get("transparent"), False), negative=str(b.get("negative") or "")[:500])
        if req["style"] not in ART_STYLES:
            bad(f"style must be one of: {', '.join(k for k in ART_STYLES if k)}")
        if req["aspect"] not in ART_SIZES:
            bad("aspect must be square, portrait or landscape")
        try:
            req["strength"] = float(b.get("strength") or 0.68)
        except (TypeError, ValueError):
            bad("strength must be a number between 0.15 and 0.95")
        if not isinstance(req["image"], str) or not req["image"]:
            req["image"] = None
        if not req["prompt"] and req["image"] and not req["style"]:
            bad("Pick a style or describe how to redraw the photo")
    if mode == "text" and not req["prompt"] or mode == "2d" and not req["prompt"] and not req.get("image"):
        bad("Type a prompt first")
    if mode == "image" and not (isinstance(req["image"], str) and req["image"]):
        bad("No photo was received — pick a photo and try again")
    if mode == "multiview":
        imgs = b.get("images") if isinstance(b.get("images"), dict) else {}
        imgs = {k: v for k, v in imgs.items() if k in MV_VIEWS and isinstance(v, str) and v}
        if "front" not in imgs:
            bad("Multi-view needs at least a front photo")
        req["images"] = imgs
    fp = fingerprint(req)
    key = f"{fp}:{int(req['texture'])}"
    if not b.get("fresh"):
        for old_id, old in sorted(tasks.items(), key=lambda kv: -kv[1].get("created", 0)):
            if old.get("key") != key:
                continue
            if old["status"] in ("queued", "running"):
                return {"task_id": old_id, "reused": "running"}      # reconnect instead of queueing a duplicate
            if old["status"] == "succeeded" and ((OUT / f"{old_id}.glb").exists() or (OUT / f"{old_id}.png").exists()) and old.get("textured", True) == req["texture"]:
                return {"task_id": old_id, "reused": "finished"}     # same inputs already done -> instant result
            break
    tid = uuid.uuid4().hex[:24]  # 96 random bits: download links are shareable
    tasks[tid] = {"status": "queued", "progress": 0, "stage": "Waiting in queue", "created": time.time(),
                  "req": req, "fp": fp, "key": key}
    save_tasks()
    jobs.put(tid)
    return {"task_id": tid}


@app.get("/status/{tid}")
def status(tid: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "unknown task")
    return {k: v for k, v in t.items() if k not in ("req", "fp", "key")}


@app.get("/models/{name}")
def model(name: str):
    f = OUT / Path(name).name
    if not f.exists() or f.suffix != ".glb":
        raise HTTPException(404)
    return FileResponse(f, media_type="model/gltf-binary")


@app.get("/images/{name}")
def art_image(name: str):
    f = OUT / Path(name).name
    if not f.exists() or f.suffix != ".png" or f.stem.endswith("_input") or f.stem.startswith("cut_"):
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


@app.get("/inputs/{name}")
def inputs(name: str):
    f = OUT / (Path(name).stem + "_input.png")
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
