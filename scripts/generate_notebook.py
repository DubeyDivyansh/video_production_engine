import json
import os

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {"id": "view-in-github"},
            "source": [
                "<div align=\"center\">\n",
                "\n",
                "<a href=\"https://colab.research.google.com/github/DubeyDivyansh/video_production_engine/blob/main/quick_demo.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
                "\n",
                "# 🎬 Modular AI Video & Voice Production Studio\n",
                "### Free Tesla T4 GPU (16GB VRAM) • Persistent Google Drive • Zero Cloud Cost\n",
                "\n",
                "</div>\n",
                "\n",
                "---\n",
                "\n",
                "### 🌟 Capabilities:\n",
                "1. **Continuous Avatar Knowledge Base:** Store your voice, portrait, and base video in Google Drive. As you upload new media, the system automatically evaluates SNR, sharpness, and motion stability to continuously upgrade your avatar profile without restarting from scratch.\n",
                "2. **Zero-Shot Voice Cloning (F5-TTS):** Generates studio-grade cloned speech from as little as 5 seconds of clean reference voice.\n",
                "3. **Dual Animation Engines:**\n",
                "   - **LatentSync (Default):** Retains 100% of body language, natural gestures, and clothing for whole-body videos.\n",
                "   - **LivePortrait / SadTalker:** High-fidelity facial performance animation from single portrait photos.\n",
                "4. **Studio Post-Production Enhancers:**\n",
                "   - **Background Matting & Replacement (BiRefNet / Chroma Key / Solid Studio Backdrops)**\n",
                "   - **Face Super-Resolution & Restoration (GFPGAN v1.4)**\n",
                "   - **Audio Spectral Denoising & -16 LUFS Mastering (EBU R128)**\n",
                "   - **Dynamic BGM Mixing with Voice Ducking & Styled Subtitle Burn-In**\n",
                "5. **Visual Node UI (ComfyUI Geeky LatentSync):** Optional 1-click Cloudflare tunnel for node-graph experimentation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-0-md"},
            "source": [
                "---\n",
                "## 0. Hardware & GPU Environment Check\n",
                "Verify your runtime has an active NVIDIA Tesla T4 GPU with ~15GB of free VRAM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-0-code"},
            "outputs": [],
            "source": [
                "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-1-md"},
            "source": [
                "---\n",
                "## 1. Mount Google Drive (Persistent Storage)\n",
                "Mount Google Drive to persist your heavy model weights (~6 GB), avatar profiles, and generated deliverables across ephemeral Colab sessions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-1-code"},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "import os\n",
                "\n",
                "try:\n",
                "    drive.mount('/content/drive')\n",
                "    BASE_DIR = '/content/drive/MyDrive/video_gen_platform'\n",
                "    for sub in ['checkpoints', 'avatars', 'inputs', 'outputs']:\n",
                "        os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)\n",
                "    print(f'✅ Persistent Drive storage active at: {BASE_DIR}')\n",
                "except Exception as e:\n",
                "    print('⚠️ Drive mount notice (running in local / ephemeral mode):', e)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-2-md"},
            "source": [
                "---\n",
                "## 2. Clone Repository & Install Dependencies\n",
                "Clones `video_production_engine` from GitHub and installs the unified production dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-2-code"},
            "outputs": [],
            "source": [
                "# 1. Clone repository from GitHub\n",
                "!git clone https://github.com/DubeyDivyansh/video_production_engine.git /content/app 2>/dev/null || (cd /content/app && git pull)\n",
                "%cd /content/app\n",
                "\n",
                "# 2. Install production dependencies\n",
                "!pip install -q -r requirements.txt\n",
                "\n",
                "# 3. Ensure system ffmpeg is up to date\n",
                "!apt-get install -y -q ffmpeg > /dev/null 2>&1\n",
                "\n",
                "import sys\n",
                "if '/content/app' not in sys.path:\n",
                "    sys.path.append('/content/app')\n",
                "\n",
                "print('✅ Production environment ready!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-3-md"},
            "source": [
                "---\n",
                "## 3. Persistent Avatar Knowledge Base (Continuous Improvement)\n",
                "Register your personal photos, videos, and audio samples under your avatar profile (`divyansh`).\n",
                "The engine **automatically scores quality** (Audio SNR, image sharpness, video motion stability), archives raw assets, and updates optimal generation pointers on Google Drive."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-3-code"},
            "outputs": [],
            "source": [
                "from src.avatar.manager import AvatarManager\n",
                "import json\n",
                "\n",
                "# Initialize Avatar Manager connected to Google Drive\n",
                "manager = AvatarManager(use_drive=True)\n",
                "\n",
                "# Download sample user assets for initial registration\n",
                "!wget -q -O /content/my_voice.wav https://files.catbox.moe/dvin9r.wav\n",
                "!wget -q -O /content/my_face.jpeg https://files.catbox.moe/puyuek.jpeg\n",
                "\n",
                "# Register and automatically evaluate quality\n",
                "print('[*] Registering audio asset...')\n",
                "res_audio = manager.register_media('divyansh', '/content/my_voice.wav', media_type='audio')\n",
                "print(f\"    - Quality Score: {res_audio['quality_score']:.2f} | Optimal Slice: {res_audio['optimal_slice']}\")\n",
                "\n",
                "print('[*] Registering portrait image...')\n",
                "res_img = manager.register_media('divyansh', '/content/my_face.jpeg', media_type='image')\n",
                "print(f\"    - Sharpness Score: {res_img['quality_score']:.2f}\")\n",
                "\n",
                "# Inspect Avatar Profile state\n",
                "profile = manager.get_avatar('divyansh')\n",
                "print(f\"\\n🎉 Avatar Profile Active: '{profile.avatar_id}'\")\n",
                "print(f\"   Optimal Audio: {profile.optimal_audio_path} (Score: {profile.optimal_audio_score:.2f})\")\n",
                "print(f\"   Optimal Image: {profile.optimal_image_path} (Score: {profile.optimal_image_score:.2f})\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-4-md"},
            "source": [
                "---\n",
                "## 4. Studio Video Production Run (Zero-Restart End-to-End Pipeline)\n",
                "Run the complete studio production pipeline with 1 call:\n",
                "1. Loads avatar assets from Drive.\n",
                "2. Synthesizes cloned voice with F5-TTS.\n",
                "3. Masters audio (spectral noise removal + -16 LUFS).\n",
                "4. Enforces VRAM fence (`torch.cuda.empty_cache()` + `gc.collect()`).\n",
                "5. Drives whole-body lip-sync (LatentSync with T4 FP16 patch) or facial reenactment.\n",
                "6. Restores and upscales face via GFPGAN.\n",
                "7. Composites custom background (image, video, or solid studio backdrop).\n",
                "8. Burns styled subtitles & mixes background music with automatic speech ducking.\n",
                "9. Saves finished deliverable to Google Drive!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-4-code"},
            "outputs": [],
            "source": [
                "from src.pipeline import VideoProductionPipeline\n",
                "\n",
                "# Initialize unified pipeline\n",
                "pipeline = VideoProductionPipeline(use_drive=True)\n",
                "\n",
                "# Script text for your avatar to deliver\n",
                "SCRIPT = (\n",
                "    \"Hello! Welcome to our autonomous video production studio. \"\n",
                "    \"Everything you are seeing and hearing is generated zero-cost on Google Colab, \"\n",
                "    \"with continuous avatar calibration, studio audio mastering, and background replacement.\"\n",
                ")\n",
                "\n",
                "# Execute Full Production\n",
                "final_video = pipeline.produce_video(\n",
                "    avatar_id='divyansh',                   # Pulls optimal assets from Drive knowledge base\n",
                "    script_text=SCRIPT,\n",
                "    output_name='divyansh_studio_master.mp4',\n",
                "    engine='sadtalker',                    # 'sadtalker' for photo, 'latentsync' for body video\n",
                "    denoise_audio=True,                    # Studio spectral gating + EBU R128 loudness\n",
                "    enhance_face=True,                     # GFPGAN facial super-resolution\n",
                "    color_backdrop='#1A1A2E',              # Modern dark slate studio backdrop\n",
                "    add_subtitles=True,                    # Auto-generate & burn styled subtitles\n",
                ")\n",
                "\n",
                "print(f\"\\n🎬 Production Deliverable Ready at: {final_video}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-5-md"},
            "source": [
                "---\n",
                "## 5. Inline Video Player & Deliverable Verification\n",
                "Preview the generated MP4 deliverable directly inside your Colab notebook."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-5-code"},
            "outputs": [],
            "source": [
                "from IPython.display import HTML, display\n",
                "from base64 import b64encode\n",
                "import os\n",
                "\n",
                "video_to_play = final_video if 'final_video' in locals() and os.path.exists(final_video) else None\n",
                "if not video_to_play:\n",
                "    fallback = '/content/drive/MyDrive/video_gen_platform/outputs/divyansh_studio_master.mp4'\n",
                "    if os.path.exists(fallback):\n",
                "        video_to_play = fallback\n",
                "\n",
                "if video_to_play and os.path.exists(video_to_play):\n",
                "    mp4_bytes = open(video_to_play, 'rb').read()\n",
                "    data_url = 'data:video/mp4;base64,' + b64encode(mp4_bytes).decode()\n",
                "    display(HTML(f'''\n",
                "    <div style=\"text-align: center;\">\n",
                "      <h3>🎉 Rendered Deliverable: {os.path.basename(video_to_play)}</h3>\n",
                "      <video width=640 controls autoplay loop style=\"border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);\">\n",
                "        <source src=\"{data_url}\" type=\"video/mp4\">\n",
                "      </video>\n",
                "    </div>\n",
                "    '''))\n",
                "else:\n",
                "    print('⚠️ No video deliverable found to display. Please run Section 4 first.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-6-md"},
            "source": [
                "---\n",
                "## 6. (Optional) Visual UI: ComfyUI + Geeky LatentSync Wrapper via Cloudflare Tunnel\n",
                "For users who want to experiment with interactive node graphs in browser.\n",
                "Launches ComfyUI and exposes a public HTTPS tunnel."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-6-code"},
            "outputs": [],
            "source": [
                "# 1. Clone ComfyUI and Geeky LatentSync Wrapper if needed\n",
                "%cd /content\n",
                "!git clone https://github.com/comfyanonymous/ComfyUI.git 2>/dev/null || true\n",
                "%cd /content/ComfyUI/custom_nodes\n",
                "!git clone https://github.com/GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper.git 2>/dev/null || true\n",
                "\n",
                "# 2. Install Cloudflare tunnel\n",
                "!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64\n",
                "!chmod +x /usr/local/bin/cloudflared\n",
                "\n",
                "# 3. Launch ComfyUI with low-VRAM flag in background and start tunnel\n",
                "%cd /content/ComfyUI\n",
                "import subprocess\n",
                "import time\n",
                "\n",
                "print('[*] Starting ComfyUI server...')\n",
                "comfy_proc = subprocess.Popen(['python', 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--lowvram'])\n",
                "time.sleep(3)\n",
                "\n",
                "print('[*] Exposing public Cloudflare tunnel...')\n",
                "!cloudflared tunnel --url http://127.0.0.1:8188 > /content/tunnel.log 2>&1 &\n",
                "time.sleep(5)\n",
                "\n",
                "!grep -o 'https://.*\\.trycloudflare\\.com' /content/tunnel.log | head -n 1\n",
                "print('\\n👉 Click the .trycloudflare.com link above to open ComfyUI!')"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    # Write to projects/video_production_engine/quick_demo.ipynb
    with open("quick_demo.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/quick_demo.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully generated quick_demo.ipynb in video_production_engine!")

if __name__ == "__main__":
    create_notebook()
