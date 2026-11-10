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

echo "== System packages"
if command -v dnf >/dev/null; then        # Amazon Linux 2023
  sudo dnf install -y -q git gcc gcc-c++ make mesa-libGL glib2 tar >/dev/null || true
elif command -v apt-get >/dev/null; then  # Ubuntu
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential libgl1 libglib2.0-0 >/dev/null || true
fi

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

echo "== Python environment (separate from any other app on this machine)"
if [ -x /opt/pytorch/bin/python ]; then
  PY=/opt/pytorch/bin/python                       # AWS Deep Learning AMI: ready-made CUDA PyTorch
else
  BASE=""
  for c in /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10; do
    [ -x "$c" ] && { BASE="$c"; break; }
  done
  if [ -z "$BASE" ]; then
    if command -v dnf >/dev/null; then sudo dnf install -y -q python3.11 python3.11-devel python3.11-pip >/dev/null; BASE=/usr/bin/python3.11
    else sudo apt-get install -y -qq python3-venv python3-dev >/dev/null; BASE=$(command -v python3); fi
  fi
  if command -v dnf >/dev/null; then sudo dnf install -y -q "$(basename "$BASE")-devel" >/dev/null 2>&1 || true; fi
  VENV="$HOME/modelforge-venv"
  [ -x "$VENV/bin/python" ] || "$BASE" -m venv "$VENV"
  PY="$VENV/bin/python"
  "$PY" -m pip install -q --upgrade pip setuptools wheel
fi
echo "Using $PY"
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "Installing CUDA PyTorch (~3 GB)…"
  "$PY" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi
"$PY" -m pip install -q pybind11 ninja

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

REPO_DIR="$(dirname "$SERVER_DIR")"
IMDS_TOKEN=$(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
PUBLIC_IP=$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || true)
HOST="${PUBLIC_IP//./-}.sslip.io"     # free DNS name that points at this IP, so we can get an HTTPS certificate

echo "== MCP server (lets Claude call your GPU server)"
if [ -f /etc/modelforge-mcp.env ] && grep -q '^MCP_SECRET=' /etc/modelforge-mcp.env; then
  MCP_SECRET=$(grep '^MCP_SECRET=' /etc/modelforge-mcp.env | cut -d= -f2-)
else
  MCP_SECRET=$(openssl rand -hex 16)
fi
"$PY" -m venv "$REPO_DIR/mcp/.venv"
"$REPO_DIR/mcp/.venv/bin/pip" install -q --upgrade pip
"$REPO_DIR/mcp/.venv/bin/pip" install -q -r "$REPO_DIR/mcp/requirements.txt"
sudo tee /etc/modelforge-mcp.env >/dev/null <<ENV
MODELFORGE_URL=http://127.0.0.1:8000
MODELFORGE_TOKEN=$TOKEN
MODELFORGE_PUBLIC_URL=https://$HOST
MCP_PORT=8765
MCP_PATH=/mcp/$MCP_SECRET
MCP_SECRET=$MCP_SECRET
ENV
sudo chmod 600 /etc/modelforge-mcp.env
sudo tee /etc/systemd/system/modelforge-mcp.service >/dev/null <<UNIT
[Unit]
Description=Model Forge MCP server (remote, for Claude)
After=modelforge.service

[Service]
User=$(id -un)
EnvironmentFile=/etc/modelforge-mcp.env
ExecStart=$REPO_DIR/mcp/.venv/bin/python $REPO_DIR/mcp/modelforge_mcp.py --http
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

echo "== HTTPS (Caddy + free Let's Encrypt certificate for $HOST)"
if ! command -v caddy >/dev/null && [ ! -x /usr/local/bin/caddy ]; then
  curl -fsSL -o /tmp/caddy.tgz https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz
  sudo tar -xzf /tmp/caddy.tgz -C /usr/local/bin caddy && rm -f /tmp/caddy.tgz
fi
# The public IP can change after stop/start, so the Caddyfile is regenerated on every boot.
sudo tee /usr/local/bin/modelforge-caddyfile >/dev/null <<'GEN'
#!/usr/bin/env bash
T=$(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
IP=$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/public-ipv4)
mkdir -p /etc/caddy
cat > /etc/caddy/Caddyfile <<CF
${IP//./-}.sslip.io {
	handle /mcp/* {
		reverse_proxy 127.0.0.1:8765
	}
	handle {
		reverse_proxy 127.0.0.1:8000
	}
}
CF
sed -i "s#^MODELFORGE_PUBLIC_URL=.*#MODELFORGE_PUBLIC_URL=https://${IP//./-}.sslip.io#" /etc/modelforge-mcp.env 2>/dev/null || true
GEN
sudo chmod +x /usr/local/bin/modelforge-caddyfile
sudo tee /etc/systemd/system/modelforge-https.service >/dev/null <<'UNIT'
[Unit]
Description=Model Forge HTTPS (Caddy)
After=network-online.target
Wants=network-online.target

[Service]
Environment=XDG_DATA_HOME=/var/lib XDG_CONFIG_HOME=/etc
ExecStartPre=/usr/local/bin/modelforge-caddyfile
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now modelforge-https modelforge-mcp
sudo systemctl restart modelforge-https modelforge-mcp
sleep 3
HTTPS_OK=0
for i in $(seq 1 20); do curl -sf -m 5 "https://$HOST/health" >/dev/null && { HTTPS_OK=1; break; }; sleep 3; done

if [ "$HTTPS_OK" = 1 ]; then APP_URL="https://$HOST"; else APP_URL="http://${PUBLIC_IP:-<public-ip>}:8000"; fi
cat <<DONE

✅ Model Forge is running and starts automatically on boot.

  📱 App (⚙ Settings)
     My AI server URL:  $APP_URL
     Server token:      $TOKEN

  🤖 Claude (MCP)
     claude.ai → Settings → Connectors → Add custom connector, URL:
        https://$HOST/mcp/$MCP_SECRET/
     Claude Code on your computer:
        claude mcp add model-forge -e MODELFORGE_URL=$APP_URL -e MODELFORGE_TOKEN=$TOKEN -- bash <path-to>/3dmodel/mcp/run.sh

  Security group must allow inbound TCP 80 and 443 (HTTPS) and 8000 (plain HTTP).
DONE
[ "$HTTPS_OK" = 1 ] || echo "  ⚠️  HTTPS not ready yet (open ports 80 + 443, then: sudo systemctl restart modelforge-https). Plain HTTP works meanwhile."
cat <<DONE

  Keep the token and connector URL private — they give access to your GPU server.
  The first generation downloads the AI models (~15 GB) and takes several minutes.

  Logs:     sudo journalctl -u modelforge -f        (AI)   |  -u modelforge-mcp  |  -u modelforge-https
  Update:   cd $REPO_DIR && git pull && sudo systemctl restart modelforge modelforge-mcp
DONE
