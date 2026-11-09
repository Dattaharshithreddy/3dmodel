#!/usr/bin/env bash
# Model Forge AI server — one-command install on an AWS EC2 GPU instance.
# Tested target: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)" on g5.xlarge or g4dn.xlarge.
# Usage (on the instance):
#   git clone https://github.com/Dattaharshithreddy/3dmodel && bash 3dmodel/server/ec2/install.sh
set -euo pipefail
SERVER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SERVER_DIR"

echo "== Checking GPU"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "❌ No NVIDIA GPU found. Use a GPU instance type (g5.xlarge or g4dn.xlarge) with the Deep Learning AMI."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

echo "== Checking memory"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
SWAP_GB=$(awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "RAM: ${RAM_GB} GB, swap: ${SWAP_GB} GB"
if [ "$RAM_GB" -lt 30 ] && [ "$SWAP_GB" -lt 16 ] && [ ! -f /swapfile ]; then
  # Small-RAM instances (e.g. g4dn.xlarge = 16 GB) offload models from the GPU to RAM; swap prevents crashes.
  FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$FREE_GB" -gt 70 ]; then
    echo "Adding 32 GB swap file…"
    sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  else
    echo "⚠️  Only ${FREE_GB} GB disk free — skipping swap. Increase the EBS volume to 100 GB for more stability."
  fi
fi

echo "== Choosing Python"
if [ -x /opt/pytorch/bin/python ]; then PY=/opt/pytorch/bin/python          # Deep Learning AMI (2024+)
elif [ -x /opt/conda/envs/pytorch/bin/python ]; then PY=/opt/conda/envs/pytorch/bin/python
else PY=$(command -v python3); fi
echo "Using $PY"
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "Installing CUDA PyTorch…"
  "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo "== Installing Hunyuan3D-2 + server (10–20 min)"
PYTHON="$PY" bash ./setup.sh

echo "== Configuring service"
LOW=0; [ "${VRAM_MB:-0}" -lt 20000 ] && LOW=1      # 16 GB cards (g4dn) need CPU offloading for textures
if [ -f /etc/modelforge.env ] && grep -q '^API_TOKEN=' /etc/modelforge.env; then
  TOKEN=$(grep '^API_TOKEN=' /etc/modelforge.env | cut -d= -f2-)
else
  TOKEN=$(openssl rand -hex 12)
fi
sudo tee /etc/modelforge.env >/dev/null <<ENV
PORT=8000
LOW_VRAM=$LOW
API_TOKEN=$TOKEN
OUTPUT_DIR=$SERVER_DIR/outputs
ENV
sudo chmod 600 /etc/modelforge.env
sudo tee /etc/systemd/system/modelforge.service >/dev/null <<UNIT
[Unit]
Description=Model Forge AI 3D server
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$SERVER_DIR
EnvironmentFile=/etc/modelforge.env
ExecStart=$PY $SERVER_DIR/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now modelforge
sudo systemctl restart modelforge

echo "== Waiting for the server"
for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
curl -s localhost:8000/health && echo || { echo "❌ Server not responding. Logs:"; sudo journalctl -u modelforge -n 40 --no-pager; exit 1; }

PUBLIC_IP=$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/public-ipv4 || true)
cat <<DONE

✅ Model Forge server is running and starts automatically on boot.

   In the app → ⚙ Settings:
     My AI server URL:  http://${PUBLIC_IP:-<your-instance-public-IP>}:8000
     Server token:      $TOKEN

   Make sure the instance's Security Group allows inbound TCP 8000.
   The first generation downloads the AI models (~15 GB) and takes several minutes.

   Logs:     sudo journalctl -u modelforge -f
   Restart:  sudo systemctl restart modelforge
   Update:   cd $(dirname "$SERVER_DIR") && git pull && sudo systemctl restart modelforge
DONE
