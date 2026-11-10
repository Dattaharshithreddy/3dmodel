#!/usr/bin/env bash
# Replace the server token and the Claude connector secret with new random ones (use if they were ever shared).
set -euo pipefail
TOKEN=$(openssl rand -hex 12)
SECRET=$(openssl rand -hex 16)
sudo sed -i "s#^API_TOKEN=.*#API_TOKEN=$TOKEN#" /etc/modelforge.env
if [ -f /etc/modelforge-mcp.env ]; then
  sudo sed -i "s#^MODELFORGE_TOKEN=.*#MODELFORGE_TOKEN=$TOKEN#; s#^MCP_SECRET=.*#MCP_SECRET=$SECRET#; s#^MCP_PATH=.*#MCP_PATH=/mcp/$SECRET#" /etc/modelforge-mcp.env
fi
sudo systemctl restart modelforge
sudo systemctl restart modelforge-mcp 2>/dev/null || true
HOST=$(sudo grep '^MODELFORGE_PUBLIC_URL=' /etc/modelforge-mcp.env 2>/dev/null | cut -d= -f2-)
cat <<DONE
✅ New secrets are active (old ones no longer work).

  App → ⚙ Settings → Server token:   $TOKEN
  Claude connector URL:               ${HOST:-https://<your-host>}/mcp/$SECRET/

  Update the app and Claude with these. Don't share them in screenshots.
DONE
