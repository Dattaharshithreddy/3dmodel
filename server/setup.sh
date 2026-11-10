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
# NVIDIA's pip packages don't ship nvcc, so when no toolkit is present we install NVIDIA's official
# "cuda-toolkit-12-X" package (compiler + headers only, never touches the GPU driver).
install_cuda_toolkit() {
  local TV MIN m v
  TV=$("$PY" -c "import torch; print(torch.version.cuda or '')")
  [ -z "$TV" ] && { echo "⚠️  PyTorch has no CUDA support — skipping textures."; return 1; }
  [ "${TV%%.*}" = "12" ] || { echo "⚠️  Unsupported torch CUDA $TV — skipping textures."; return 1; }
  MIN=${TV#*.}; MIN=${MIN%%.*}
  echo "No CUDA toolkit found — installing NVIDIA CUDA toolkit 12.x (~4 GB, compiler only)…"
  if command -v dnf >/dev/null; then
    . /etc/os-release
    case "$ID" in amzn) REPO=amzn2023 ;; *) REPO="rhel${VERSION_ID%%.*}" ;; esac
    sudo curl -fsSL -o /etc/yum.repos.d/cuda-$REPO.repo \
      "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-$REPO.repo" || return 1
    sudo dnf -q makecache >/dev/null 2>&1 || true
    # versions NVIDIA offers for this OS, e.g. "12-8 12-9"; prefer the one closest to (and not below) torch's
    AVAIL=$(sudo dnf -q list --available 'cuda-toolkit-12-*' 2>/dev/null | grep -oE '^cuda-toolkit-12-[0-9]+\.' | grep -oE '12-[0-9]+' | sort -t- -k2 -n -u)
    echo "CUDA toolkits available: ${AVAIL:-none}"
    for m in $(echo "$AVAIL" | cut -d- -f2); do
      if [ "$m" -ge "$MIN" ]; then sudo dnf install -y "cuda-toolkit-12-$m" && v="12.$m"; break; fi
    done
  elif command -v apt-get >/dev/null; then
    . /etc/os-release
    UBU="ubuntu${VERSION_ID//./}"
    curl -fsSL -o /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/$UBU/x86_64/cuda-keyring_1.1-1_all.deb" || return 1
    sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null && sudo apt-get update -qq
    for m in $(seq "$MIN" 12); do
      if apt-cache show "cuda-toolkit-12-$m" >/dev/null 2>&1; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-12-$m" && v="12.$m"; break
      fi
    done
  fi
  [ -n "${v:-}" ] && [ -x "/usr/local/cuda-$v/bin/nvcc" ] || { echo "⚠️  Could not install the CUDA toolkit."; return 1; }
  export CUDA_HOME="/usr/local/cuda-$v"
}
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null; then export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda
  else
    found=""; for d in /usr/local/cuda-12.*; do [ -x "$d/bin/nvcc" ] && found="$d"; done
    if [ -n "$found" ]; then export CUDA_HOME="$found"
    else install_cuda_toolkit || { echo "⚠️  Textures disabled — the server will make grey models. Shapes work normally."; exit 0; }
    fi
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
