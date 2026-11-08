#!/usr/bin/env bash
# Installs Hunyuan3D-2 (open-source image/text -> 3D model by Tencent) and this server.
# Needs Linux/WSL + NVIDIA GPU (>= 8 GB VRAM for shapes, ~16 GB with textures) + CUDA PyTorch.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"   # set PYTHON=/path/to/python to install into a specific environment
if [ ! -d Hunyuan3D-2 ]; then git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2; fi
"$PY" -m pip install -r Hunyuan3D-2/requirements.txt
"$PY" -m pip install -e Hunyuan3D-2
"$PY" -m pip install -r requirements.txt
if [ "${SKIP_TEXTURE:-0}" != "1" ]; then
  # Texture painting needs two small CUDA extensions compiled.
  (cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && "$PY" -m pip install --no-build-isolation .)
  (cd Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && "$PY" -m pip install --no-build-isolation .)
fi
echo "Done. Start with:  $PY server.py"
