# Run the Model Forge AI server on AWS EC2

## 1. Launch a GPU instance
In the AWS console, go to **EC2 → Launch instance**:

| Setting | Value |
|---|---|
| AMI | **Deep Learning OSS Nvidia Driver AMI GPU PyTorch … (Ubuntu 22.04)**. Search "Deep Learning PyTorch" in the AMI list |
| Instance type | **g5.xlarge** (A10G, 24 GB, recommended, ~$1/hour) or **g4dn.xlarge** (T4, 16 GB, cheaper, ~$0.53/hour, slower textures) |
| Storage | **100 GB** gp3 (the AI models are ~15–25 GB) |
| Key pair | Create or pick one so you can SSH in |
| Security group | Allow **SSH (22)** from *My IP*, and **HTTP (80)**, **HTTPS (443)** and **Custom TCP 8000** from *Anywhere* (80/443 give you a secure https link; 8000 is the plain fallback) |

> ⚠️ Ordinary instance types (t2/t3/m5/c5, free tier) have **no GPU** and won't work.
> New AWS accounts often have a GPU quota of **0**. If launch fails with *"You have requested more vCPU capacity than your current vCPU limit"*, go to **Service Quotas → EC2 → "Running On-Demand G and VT instances"**, request **8**, and wait for approval (usually hours).

## 2. Install (one command)
Connect with **EC2 Instance Connect** (in the console) or SSH, check the GPU with `nvidia-smi`, then run:
```bash
command -v git || sudo dnf install -y git || sudo apt-get install -y git
```
```bash
git clone https://github.com/Dattaharshithreddy/3dmodel && bash 3dmodel/server/ec2/install.sh
```
It works on Amazon Linux 2023 and Ubuntu. It installs Hunyuan3D-2, the server, the **MCP server for Claude**, and a free **HTTPS** certificate (via `<ip>.sslip.io`), and sets everything to start automatically on boot. At the end it prints the **app URL and token** and the **Claude connector URL**.

If `nvidia-smi` says "command not found", the instance has no NVIDIA driver. Launch it again with the *Deep Learning OSS Nvidia Driver AMI GPU PyTorch* image.

## 3. Connect the app
In **⚙ Settings**, paste the URL (`http://<public-ip>:8000`) and the token, tap **Test connection**, then **Save**.

## Costs and tips
- **Stop the instance when you're not using it.** GPU instances bill per hour while running. The server starts automatically when you start the instance again.
- The public IP changes after stop/start. Attach an **Elastic IP** (EC2 → Elastic IPs → Allocate → Associate) to keep the same URL.
- Logs: `sudo journalctl -u modelforge -f`. Update to the latest code: `cd ~/3dmodel && git pull && sudo systemctl restart modelforge`.
- The token protects your server from strangers. To change it, edit `/etc/modelforge.env`, then run `sudo systemctl restart modelforge`.
