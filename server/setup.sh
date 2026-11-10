#!/usr/bin/env bash
# Installs Hunyuan3D-2 (open-source image/text -> 3D by Tencent) and this server into the Python given by $PYTHON.
# Needs Linux + NVIDIA GPU (>= 8 GB VRAM for shapes, ~16 GB with textures) + CUDA PyTorch.
# Textures need a CUDA compiler; if none is installed, NVIDIA's nvcc is fetched from pip automatically.
# SKIP_TEXTURE=1 skips texture support (grey models, faster install).
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"   # set PYTHON=/path/to/python to install into a specific environment
if [ ! -d Hunyuan3D-2 ]; then git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2; fi
"$PY" -m pip install -r Hunyuan3D-2/requirements.txt
"$PY" -m pip install -e Hunyuan3D-2
"$PY" -m pip install -r requirements.txt

[ "${SKIP_TEXTURE:-0}" = "1" ] && { echo "Skipping textures (SKIP_TEXTURE=1)."; exit 0; }

# ---- CUDA compiler for the texture rasterizer --------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null; then export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda
  else
    TV=$("$PY" -c "import torch; print(torch.version.cuda or '')")
    if [ -z "$TV" ]; then echo "⚠️  PyTorch has no CUDA support — skipping textures."; exit 0; fi
    echo "No CUDA toolkit found — installing NVIDIA's nvcc $TV from pip…"
    "$PY" -m pip install -q "nvidia-cuda-nvcc-cu12==${TV}.*" "nvidia-cuda-runtime-cu12==${TV}.*" "nvidia-cuda-cccl-cu12==${TV}.*" || true
    NV=$("$PY" -c "import nvidia, os; print(list(nvidia.__path__)[0])")
    CH="$(cd "$(dirname "$PY")/.." && pwd)/cuda_home"
    rm -rf "$CH"; mkdir -p "$CH/include" "$CH/lib64"
    ln -s "$NV/cuda_nvcc/bin" "$CH/bin"
    [ -d "$NV/cuda_nvcc/nvvm" ] && ln -s "$NV/cuda_nvcc/nvvm" "$CH/nvvm"
    for d in "$NV/cuda_runtime/include" "$NV/cuda_nvcc/include" "$NV/cuda_cccl/include"; do
      [ -d "$d" ] && for f in "$d"/*; do ln -sfn "$f" "$CH/include/$(basename "$f")"; done
    done
    for f in "$NV"/cuda_runtime/lib/*; do ln -sf "$f" "$CH/lib64/"; done
    [ -e "$CH/lib64/libcudart.so" ] || ln -s "$CH/lib64/libcudart.so.12" "$CH/lib64/libcudart.so"
    export CUDA_HOME="$CH"
  fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
# Compile only for this machine's GPU (faster; also works when no GPU is visible, e.g. T4 = 7.5)
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  export TORCH_CUDA_ARCH_LIST="${CC:-7.5}"
fi
echo "Building texture extensions with CUDA_HOME=$CUDA_HOME for GPU arch $TORCH_CUDA_ARCH_LIST"
TEX_OK=1
(cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && "$PY" -m pip install --no-build-isolation .) || TEX_OK=0
# Optional speed-up; there is a pure-Python fallback if this fails.
(cd Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && "$PY" -m pip install --no-build-isolation .) \
  || echo "(mesh_processor C++ speed-up not built — using the Python fallback)"
if [ "$TEX_OK" = 1 ] && "$PY" -c "import custom_rasterizer" 2>/dev/null; then
  echo "✅ Textures enabled."
else
  echo "⚠️  Texture extension failed to build — the server will make untextured (grey) models. Shapes work normally."
fi
echo "Done. Start with:  $PY server.py"
