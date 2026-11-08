# Model Forge

Your own AI 3D model generator, similar to Meshy and Tripo. Type a description or take a photo, and get a 3D model you can rotate, inspect, and export as **GLB / OBJ / STL** (STL works for 3D printing).

| Part | Folder | What it is |
|---|---|---|
| 📱 Android app | [`app/`](app) | The app you install on your phone (built from HTML/JS with Capacitor) |
| 🧠 AI engine | [`server/`](server) | Your own generation server. Runs Tencent's open-source **Hunyuan3D-2** models on a GPU |
| ☁️ Free GPU | [`colab/`](colab) | A notebook that runs the AI engine on Google Colab's free GPU |

## How it works
```
 Phone app ──(prompt or photo)──► Your AI server (GPU) ──► text→image → image→3D shape → texture painting
     ▲                                                                     │
     └─────────────────────────── .glb 3D model ◄──────────────────────────┘
```
Generating 3D models needs a powerful GPU, the same as Meshy and Tripo use, so a phone can't run the AI itself. The app sends the job to **your** server, which runs free open-source models. No subscription is needed.

## 1. Install the app
Open the [**Releases**](../../releases/latest) page on your Android phone, download `ModelForge-…apk`, and open it. If Android asks, allow **Install unknown apps** for your browser.

Every push to `main` builds a new APK automatically (see [`.github/workflows/android.yml`](.github/workflows/android.yml)), and it installs over the old version.

## 2. Start your AI engine (pick one)

**A. Free: Google Colab** (no GPU of your own needed)
1. Open [`colab/ModelForge_Server.ipynb`](colab/ModelForge_Server.ipynb) in Colab: on colab.research.google.com, choose **File → Open notebook → GitHub**, then paste this repo's URL.
2. Choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
3. Copy the `https://….trycloudflare.com` URL it prints.

**B. Your own PC** (NVIDIA GPU with 8 GB or more of VRAM; 16 GB for textures; Linux or WSL)
```bash
cd server && ./setup.sh && python server.py
```
Then use `http://<your-PC-IP>:8000` as the URL. Your phone must be on the same Wi-Fi.

## 3. Connect and create
In the app, tap **⚙ Settings**, paste the server URL, tap **Test connection**, then **Save**. Now type a prompt or pick a photo and tap **Generate**. Use the buttons on the 3D view to save or share the model.

Optional: you can also paste a paid **Meshy** or **Tripo** API key in Settings to use their cloud engines. **Demo** mode makes simple offline shapes so you can try the app without a server.

## 📸 Multi-view: most accurate models
In **Image to 3D**, switch to **Multi-view** and add photos of the **front** (required) plus the **left, back and right** sides. The server uses **Hunyuan3D-2mv**, which builds the shape from all the photos together, so the sides you can see aren't guessed. This gives the most accurate shapes, which is best for game props and characters based on real objects.

How to shoot:
- Turn the object 90° between shots, or walk around it. Keep the same distance and camera height.
- Keep the whole object in frame, on a plain background, in the same lighting each time.
- Front + back already helps a lot; all 4 is best.
- Textures are painted from the front photo.

On Colab's free T4, the server keeps only one shape model in memory at a time (`LOW_VRAM=1`). The first switch between single-photo and multi-view mode takes about 1–2 minutes while the other model loads.

## 🎮 Making game assets
- **Made for:** pick *Mobile game* (~8k triangles) or *PC / console* (~30k). This sets the polygon budget and asks the AI for a game-asset look.
- **Art style:** Realistic, Stylized, Low-poly, Cartoon, Chibi, or Sci-fi.
- **Export for games:** set the **height in meters** (e.g. 1.8 for a human, 0.5 for a chest) and your engine's up-axis. The model's pivot is placed at its feet, so it stands on the ground when you drop it in.
- **GLB** imports directly into **Godot** and **Blender**, into **Unity** with the free *glTFast* package, and into **Unreal** (5.x has glTF import built in). For **Roblox**, export OBJ and use the 3D Importer.
- **Characters that need to move:** for prompts like "full body, T-pose", import the GLB into Blender, export it as FBX, and upload it to the free **Mixamo** site for automatic rigging and animations.

## 🎬 Making reels
Open a model and tap **🎬 Make a reel from this model**. Then choose:
- **Format:** 9:16 Reel / Short / TikTok, 1:1 post, or 16:9 YouTube (all 1080p)
- **Look:** 3D realistic, 2D toon (cartoon with outlines), 2D pixel art, 2D sketch, or hologram
- **Camera move:** 360° turntable, orbit + zoom, dramatic reveal, floating, or drop & bounce
- **Background, length (5–15 s), title and subtitle**

Tap **Record**, and the video is saved or shared straight from your phone (MP4 where the phone supports it, otherwise WebM). Add music in Instagram, YouTube or CapCut. The *Green screen* background lets you put the model into other videos.

## Tips for good results
- **Photos:** one object, centered, plain background, good lighting.
- **Prompts:** describe one object, e.g. "a wooden treasure chest with gold trim, stylized".
- Turn off **Generate textures** for faster, grey (shape-only) models, which are good for 3D printing.

## Server options (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `SHAPE_MODEL` / `SHAPE_SUBFOLDER` | `tencent/Hunyuan3D-2mini` / `hunyuan3d-dit-v2-mini` | Use `tencent/Hunyuan3D-2` / `hunyuan3d-dit-v2-0` for higher quality (needs more VRAM) |
| `MV_MODEL` / `MV_SUBFOLDER` | `tencent/Hunyuan3D-2mv` / `hunyuan3d-dit-v2-mv` | Multi-view model. `hunyuan3d-dit-v2-mv-turbo` is faster |
| `TEXTURE` | on if installed | `0` = shape only |
| `LOW_VRAM` | `0` | `1` = move models to the CPU between steps (needed on 16 GB GPUs) |
| `API_TOKEN` | – | Require this token (enter the same one in the app's Settings) |

## Developing
```bash
cd app && npm install && npm run serve      # try the app in a desktop browser at http://localhost:5173
cd app && npm run apk                        # build an APK locally (needs Android SDK + JDK 17)
```

### Signing
CI signs the APK with the committed `app/android/app/debug.keystore` (public, for sideloading only). Before publishing on Google Play, create your own release keystore, keep it **out of git**, and add it to the workflow as repository secrets.

## License notes
Hunyuan3D-2 is under the Tencent Hunyuan Community License. Read it before any commercial use; it restricts use in some regions (EU, UK, South Korea).
