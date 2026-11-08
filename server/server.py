"""
Model Forge AI server — your own text-to-3D / image-to-3D engine.

Runs Tencent's open-source Hunyuan3D-2 models on your GPU:
  text  -> image (HunyuanDiT) -> 3D shape (Hunyuan3D-DiT) -> texture (Hunyuan3D-Paint)
  image ->                       3D shape                  -> texture

API (used by the Model Forge app):
  GET  /health                   -> {"ok": true, "texture": bool, ...}
  POST /generate                 {mode: "text"|"image", prompt?, image? (data URL), texture?, polycount?}
                                 -> {"task_id": "..."}
  GET  /status/{task_id}         -> {status: queued|running|succeeded|failed, progress, stage, model_url?, error?}
  GET  /models/{task_id}.glb     -> the model file

Environment variables:
  SHAPE_MODEL   default "tencent/Hunyuan3D-2mini"   (or "tencent/Hunyuan3D-2" for higher quality, more VRAM)
  SHAPE_SUBFOLDER default "hunyuan3d-dit-v2-mini" (use "hunyuan3d-dit-v2-0" with tencent/Hunyuan3D-2)
  TEXTURE       "1" to enable texture painting (default: on if the texture module is installed)
  LOW_VRAM      "1" to offload models to CPU between steps (slower, fits 12-16 GB GPUs)
  API_TOKEN     if set, clients must send "Authorization: Bearer <token>"
  PORT          default 8000
"""
import base64, io, os, threading, time, traceback, uuid, queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

OUT = Path(os.environ.get("OUTPUT_DIR", Path(__file__).parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
SHAPE_MODEL = os.environ.get("SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
SHAPE_SUBFOLDER = os.environ.get("SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-mini")
TEXTURE_MODEL = os.environ.get("TEXTURE_MODEL", "tencent/Hunyuan3D-2")
T2I_MODEL = os.environ.get("T2I_MODEL", "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"
API_TOKEN = os.environ.get("API_TOKEN", "")


# ----------------------------------------------------------------------------- models (lazy)
class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self._shape = self._paint = self._t2i = self._rembg = None
        try:
            import hy3dgen.texgen  # noqa: F401
            default_tex = "1"
        except Exception:
            default_tex = "0"
        self.texture_available = os.environ.get("TEXTURE", default_tex) == "1"

    def shape(self):
        if self._shape is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=True)
            if LOW_VRAM and hasattr(self._shape, "enable_model_cpu_offload"):
                self._shape.enable_model_cpu_offload()
        return self._shape

    def paint(self):
        if self._paint is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
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


def set_stage(t, stage, progress):
    t.update(stage=stage, progress=progress, status="running")


def run(task_id: str):
    import torch
    t = tasks[task_id]
    req = t["req"]
    try:
        # 1. input image
        if req["mode"] == "text":
            set_stage(t, "Imagining the object (text → image)", 5)
            prompt = req["prompt"].strip() + ", single object, centered, white background, 3D render, full view"
            image = engine.t2i()(prompt)
            if LOW_VRAM:
                engine._t2i = None
                torch.cuda.empty_cache()
        else:
            raw = req["image"].split(",", 1)[-1]
            image = Image.open(io.BytesIO(base64.b64decode(raw)))
        image.save(OUT / f"{task_id}_input.png")

        set_stage(t, "Removing background", 15)
        image = image.convert("RGBA")
        if image.getextrema()[3][0] == 255:  # no transparency yet
            image = engine.rembg()(image)

        # 2. geometry
        set_stage(t, "Sculpting 3D shape", 25)
        mesh = engine.shape()(image=image, num_inference_steps=int(req.get("steps") or 30),
                              octree_resolution=380, generator=torch.manual_seed(int(req.get("seed") or 1234)))[0]

        set_stage(t, "Cleaning mesh", 55)
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=int(req.get("polycount") or 40000))

        # 3. texture
        if req.get("texture", True) and engine.texture_available:
            set_stage(t, "Painting textures", 65)
            mesh = engine.paint()(mesh, image=image)

        set_stage(t, "Saving", 95)
        mesh.export(str(OUT / f"{task_id}.glb"))
        t.update(status="succeeded", progress=100, stage="Done", model_url=f"/models/{task_id}.glb",
                 thumbnail_url=f"/inputs/{task_id}.png")
    except Exception as e:
        traceback.print_exc()
        t.update(status="failed", error=str(e)[:500])
    finally:
        t.pop("req", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def worker():
    while True:
        tid = jobs.get()
        with engine.lock:  # one GPU job at a time
            run(tid)


threading.Thread(target=worker, daemon=True).start()

# ----------------------------------------------------------------------------- API
app = FastAPI(title="Model Forge AI server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def auth(request: Request, call_next):
    if API_TOKEN and request.method != "OPTIONS" and request.url.path not in ("/health",):
        if request.headers.get("authorization") != f"Bearer {API_TOKEN}":
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


class GenerateReq(BaseModel):
    mode: str
    prompt: str | None = None
    image: str | None = None
    texture: bool = True
    polycount: int | None = 40000
    steps: int | None = 30
    seed: int | None = None


@app.get("/health")
def health():
    import torch
    return {"ok": True, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "texture": engine.texture_available, "shape_model": SHAPE_MODEL, "queue": jobs.qsize(),
            "auth": bool(API_TOKEN)}


@app.post("/generate")
def generate(body: GenerateReq):
    if body.mode not in ("text", "image"):
        raise HTTPException(400, "mode must be text or image")
    if body.mode == "text" and not (body.prompt or "").strip():
        raise HTTPException(400, "prompt required")
    if body.mode == "image" and not body.image:
        raise HTTPException(400, "image required")
    tid = uuid.uuid4().hex[:12]
    tasks[tid] = {"status": "queued", "progress": 0, "stage": "Waiting in queue", "created": time.time(),
                  "req": body.model_dump()}
    jobs.put(tid)
    return {"task_id": tid}


@app.get("/status/{tid}")
def status(tid: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "unknown task")
    return {k: v for k, v in t.items() if k != "req"}


@app.get("/models/{name}")
def model(name: str):
    f = OUT / Path(name).name
    if not f.exists() or f.suffix != ".glb":
        raise HTTPException(404)
    return FileResponse(f, media_type="model/gltf-binary")


@app.get("/inputs/{name}")
def inputs(name: str):
    f = OUT / (Path(name).stem + "_input.png")
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
