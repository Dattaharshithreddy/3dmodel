#!/usr/bin/env bash
# Starts the Model Forge MCP server, creating its Python environment on first run.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${MODELFORGE_VENV:-$DIR/.venv}"
PY="${PYTHON:-python3}"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" >&2
  "$VENV/bin/pip" install -q --upgrade pip >&2
fi
if ! "$VENV/bin/python" -c "import mcp.server.fastmcp, httpx, trimesh, PIL, numpy" 2>/dev/null; then
  "$VENV/bin/pip" install -q -r "$DIR/requirements.txt" >&2
fi
exec "$VENV/bin/python" "$DIR/modelforge_mcp.py" "$@"
