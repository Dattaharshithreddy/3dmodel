#!/usr/bin/env bash
# Installs Hunyuan3D-2 (open-source image/text -> 3D model by Tencent) and this server.
# Needs Linux/WSL + NVIDIA GPU (>= 8 GB VRAM for shapes, ~16 GB with textures) + CUDA PyTorch.
set -e
cd "$(dirname "$0")"
if [ ! -d Hunyuan3D-2 ]; then git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2; fi
pip install -r Hunyuan3D-2/requirements.txt
pip install -e Hunyuan3D-2
pip install -r requirements.txt
if [ "${SKIP_TEXTURE:-0}" != "1" ]; then
  # Texture painting needs two small CUDA extensions compiled.
  (cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && pip install --no-build-isolation .)
  (cd Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && pip install --no-build-isolation .)
fi
echo "Done. Start with:  python server.py"
