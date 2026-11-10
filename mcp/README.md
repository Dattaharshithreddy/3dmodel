# Model Forge MCP server: let Claude make 3D models

With this MCP server connected, you can send Claude photos and ask for a 3D model. Claude sends them to **your** Model Forge GPU server, waits, checks a 4-angle preview of the result, and gives you the `.glb` file.

> "Here are front, side and back photos of my chair. Make a game-ready model, about 30k triangles."

| Tool | What it does |
|---|---|
| `image_to_3d` | One photo → 3D model |
| `photos_to_3d` | Front + back/left/right photos → more accurate model (multi-view) |
| `text_to_3d` | Description → 3D model |
| `check_3d_job` | Keep waiting on a long job |
| `preview_3d_model` | Render any `.glb` from 4 angles (Claude uses this to check quality) |
| `server_status` | GPU, queue, features |

You need a running Model Forge AI server first (EC2, Colab or your PC). See the main README.

## Option A: claude.ai, desktop and mobile apps (remote connector)
The EC2 installer (`server/ec2/install.sh`) already runs this MCP server online and prints a **connector URL** like `https://13-233-10-20.sslip.io/mcp/<secret>/`.

1. On claude.ai, go to **Settings → Connectors → Add custom connector**, paste the URL, and save.
2. In a chat, enable **model-forge** from the tools menu.

On claude.ai, the tools need an **https link** to each image, because Claude can't hand a chat attachment to a remote connector. Upload your photos somewhere that gives a direct link (for example GitHub, an S3 bucket, or Imgur), then paste the links. For attached photos, use Option B.

## Option B: Claude Code (photos you upload are passed directly by file path)
In this repository, `.mcp.json` already registers the server. Set two environment variables, then start Claude Code in the repo:
```bash
export MODELFORGE_URL=https://13-233-10-20.sslip.io     # your server
```
```bash
export MODELFORGE_TOKEN=<your server token>
```
Anywhere else, add it once (it needs Python 3.10+; the first start installs its dependencies):
```bash
claude mcp add model-forge -e MODELFORGE_URL=https://… -e MODELFORGE_TOKEN=… -- bash /path/to/3dmodel/mcp/run.sh
```
Models are saved to `./models/` (change this with `MODELFORGE_OUTPUT_DIR`).

## Option C: Claude Desktop (local files)
In `claude_desktop_config.json`:
```json
{ "mcpServers": { "model-forge": {
    "command": "bash", "args": ["/path/to/3dmodel/mcp/run.sh"],
    "env": { "MODELFORGE_URL": "https://…", "MODELFORGE_TOKEN": "…", "MODELFORGE_OUTPUT_DIR": "/Users/me/ModelForge" } } } }
```
Then ask, for example: "Make a 3D model from ~/Pictures/chair.jpg".

## Security
- The connector URL and token give access to your GPU server. Keep them private; to change them, re-run the installer after editing `/etc/modelforge-mcp.env` and `/etc/modelforge.env`.
- In remote mode, the server refuses local file paths and internal network addresses (such as AWS metadata), so only public images can be fetched.
- Model download links contain a long random ID. Anyone with the link can download that model.
