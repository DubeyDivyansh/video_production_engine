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
                "1. **Continuous Avatar Knowledge Base (`AvatarManager`):** Store your voice, portrait, and base video in Google Drive. As you upload new media, the system automatically evaluates SNR, sharpness, and motion stability to continuously upgrade your avatar profile without restarting from scratch.\n",
                "2. **Dual Speech Synthesis Engines:**\n",
                "   - **Kokoro-82M (Default):** Zero-VRAM / CPU narration in <1 second with 82M parameters and 24kHz studio audio (no reference audio required).\n",
                "   - **F5-TTS (Voice Cloning):** Flow-matching diffusion transformer generating studio cloned speech from 5s reference audio.\n",
                "3. **Multi-Mode Video Generation & Lip-Sync:**\n",
                "   - **LTX-Video 0.9.5 & Wan2.1 (Generative DiT):** Two-Phase Memory Offload strategy (Phase 1 text encoder -> Phase 2 Transformer + VAE) generating high-fidelity cinematic scenes within 16GB T4 VRAM.\n",
                "   - **LatentSync (Default Gesture Avatar):** Retains 100% of body language, natural gestures, and clothing for whole-body videos.\n",
                "   - **LivePortrait / SadTalker:** High-fidelity facial performance animation from single portrait photos.\n",
                "4. **Studio Post-Production Enhancers:**\n",
                "   - **RIFE v4.x (60fps Broadcast Motion Interpolation):** Optical-flow frame interpolation eliminating micro-stutter in seconds.\n",
                "   - **Background Matting & Replacement (BiRefNet / Chroma Key / Solid Studio Backdrops)**\n",
                "   - **Face Super-Resolution & Restoration (GFPGAN v1.4)**\n",
                "   - **Audio Spectral Denoising & -16 LUFS Mastering (EBU R128)**\n",
                "   - **Dynamic BGM Mixing with Voice Ducking & Styled Subtitle Burn-In**\n",
                "5. **Gemini Managed Agent Supervisor (`GeminiAgentSupervisor`):** 5-part cinematic prompt expansion and self-annealing autonomous QA diagnostic loop.\n",
                "6. **Visual Node UI (ComfyUI Geeky LatentSync):** Optional 1-click Cloudflare tunnel for node-graph experimentation."
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
                "Mount Google Drive to persist heavy model checkpoints, avatar profiles, and generated deliverables across ephemeral Colab sessions."
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
                "# 1. Clone repository from GitHub (with public mirror fallback)\n",
                "!git clone https://github.com/DubeyDivyansh/video_production_engine.git /content/app 2>/dev/null || (rm -rf /content/temp_repo && git clone https://github.com/DubeyDivyansh/DubeyDivyansh.git /content/temp_repo && cp -r /content/temp_repo/video_production_engine /content/app)\n",
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
                "Register personal photos, videos, and audio samples under your avatar profile (`divyansh`).\n",
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
                "# 1. Register portrait image\n",
                "print('[*] Registering portrait image...')\n",
                "!wget -q -O /content/my_face.jpeg https://files.catbox.moe/puyuek.jpeg\n",
                "res_img = manager.register_media('divyansh', '/content/my_face.jpeg', media_type='image')\n",
                "print(f\"    - Sharpness Score: {res_img['quality_score']:.2f}\")\n",
                "\n",
                "# 2. Ingest and extract speech audio from Google Drive lecture video\n",
                "DRIVE_LECTURE_URL = 'https://drive.google.com/file/d/1yCoOYz-Q-pzAaDYEWJUV1RihuiYJGQX2/view?usp=sharing'\n",
                "print('\\n[*] Ingesting user lecture video & extracting pristine speech audio from Google Drive...')\n",
                "res_drive = manager.ingest_drive_url(\n",
                "    avatar_id='divyansh',\n",
                "    drive_url_or_id=DRIVE_LECTURE_URL,\n",
                "    media_type='video',\n",
                "    extract_video_audio=True\n",
                ")\n",
                "\n",
                "# 3. Inspect upgraded Avatar Profile state\n",
                "profile = manager.get_avatar('divyansh')\n",
                "print(f\"\\n🎉 Avatar Profile Active: '{profile.avatar_id}'\")\n",
                "print(f\"   Optimal Audio: {profile.optimal_audio_path} (Score: {profile.optimal_audio_score:.2f})\")\n",
                "print(f\"   Optimal Image: {profile.optimal_image_path} (Score: {profile.optimal_image_score:.2f})\")\n",
                "print(f\"   Total Archived Assets: {len(profile.raw_assets)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-4-md"},
            "source": [
                "---\n",
                "## 4. Fast Speech Narration (Kokoro-82M Zero-VRAM Engine)\n",
                "Synthesizes studio-grade 24kHz speech narration with **0 MB VRAM overhead** using Kokoro-82M (Apache 2.0).\n",
                "Runs on CPU in less than 1 second without needing reference audio, leaving all 16GB of GPU VRAM free for video diffusion."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-4-code"},
            "outputs": [],
            "source": [
                "from src.audio.kokoro_engine import KokoroEngine\n",
                "import os\n",
                "import IPython.display as ipd\n",
                "\n",
                "kokoro = KokoroEngine(device=\"cpu\")\n",
                "print(\"[*] Synthesizing speech narration with Kokoro-82M (0 MB VRAM overhead)...\")\n",
                "narration_text = (\n",
                "    \"Welcome to our modular video production studio. \"\n",
                "    \"This high-fidelity speech was synthesized in less than one second using Kokoro-82M on CPU, \"\n",
                "    \"preserving all sixteen gigabytes of GPU VRAM for video diffusion models.\"\n",
                ")\n",
                "\n",
                "out_dir = '/content/drive/MyDrive/video_gen_platform/outputs'\n",
                "os.makedirs(out_dir, exist_ok=True)\n",
                "output_wav = os.path.join(out_dir, 'kokoro_narration.wav')\n",
                "\n",
                "audio_path = kokoro.synthesize(\n",
                "    text=narration_text,\n",
                "    voice=\"af_heart\",\n",
                "    output_path=output_wav,\n",
                ")\n",
                "print(f\"✅ Narration successfully synthesized: {audio_path}\")\n",
                "if os.path.exists(audio_path):\n",
                "    ipd.display(ipd.Audio(audio_path))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-5-md"},
            "source": [
                "---\n",
                "## 5. Diffusion Video Scene Generation (LTX-Video 0.9.5 Two-Phase Offload)\n",
                "Generates high-fidelity cinematic video scenes on Tesla T4 (16GB VRAM) using the **Two-Phase Memory Offload Strategy**:\n",
                "- **Phase 1:** T5-XXL text encoder loads to compute prompt embeddings, followed by an immediate VRAM purge (`del text_encoder; flush_vram()`).\n",
                "- **Phase 2:** Transformer + VAE generate video with model CPU offloading and VAE tiling (`pipe.enable_model_cpu_offload()`, `pipe.vae.enable_tiling()`).\n",
                "Adheres strictly to temporal compression requirements: **8n + 1 frames** (e.g. 41 frames) and dimensions divisible by 32."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-5-code"},
            "outputs": [],
            "source": [
                "from src.video.generative_engine import GenerativeVideoEngine\n",
                "import os\n",
                "\n",
                "gen_engine = GenerativeVideoEngine()\n",
                "print(\"[*] Generating cinematic diffusion video scene with LTX-Video 0.9.5...\")\n",
                "print(\"[*] Strategy: Two-Phase Memory Offload on T4 (Phase 1 text encoder -> Phase 2 Transformer + VAE)\")\n",
                "\n",
                "scene_prompt = \"Cinematic aerial view of futuristic cyberpunk neon city at night, rain reflections, photorealistic 4k\"\n",
                "out_dir = '/content/drive/MyDrive/video_gen_platform/outputs'\n",
                "os.makedirs(out_dir, exist_ok=True)\n",
                "output_scene = os.path.join(out_dir, 'ltx_cyberpunk_scene.mp4')\n",
                "\n",
                "# Generate scene (41 frames = 8n + 1 rule, 512x320 divisible by 32)\n",
                "scene_path = gen_engine.generate_scene(\n",
                "    prompt=scene_prompt,\n",
                "    output_path=output_scene,\n",
                "    model_name=\"ltx-video\",\n",
                "    width=512,\n",
                "    height=320,\n",
                "    num_frames=41,\n",
                "    fps=24,\n",
                "    num_steps=30,\n",
                ")\n",
                "print(f\"✅ LTX-Video scene rendered successfully: {scene_path}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-6-md"},
            "source": [
                "---\n",
                "## 6. Studio Video Production Run (Zero-Restart End-to-End Pipeline)\n",
                "Run the complete multi-mode studio production pipeline with 1 call:\n",
                "1. Loads avatar assets from Drive.\n",
                "2. Synthesizes voice via Kokoro-82M (0 VRAM) or F5-TTS (voice cloning).\n",
                "3. Masters audio (spectral noise removal + -16 LUFS loudness).\n",
                "4. Enforces VRAM fence (`torch.cuda.empty_cache()` + `gc.collect()`).\n",
                "5. Drives animation via LatentSync (whole body), LivePortrait / SadTalker (portrait), or Generative DiT (LTX-Video scene).\n",
                "6. Restores and upscales face via GFPGAN.\n",
                "7. Composites custom background (image, video, or solid studio backdrop).\n",
                "8. Upsamples to 60fps broadcast framerate via RIFE optical flow.\n",
                "9. Burns styled subtitles & mixes background music with automatic speech ducking.\n",
                "10. Saves finished deliverable to Google Drive!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-6-code"},
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
                "    engine='sadtalker',                    # 'sadtalker' for photo, 'latentsync' for body, 'generative' for scene\n",
                "    tts_engine='kokoro',                   # 'kokoro' (0 MB VRAM) or 'f5' (voice cloning)\n",
                "    denoise_audio=True,                    # Studio spectral gating + EBU R128 loudness\n",
                "    enhance_face=True,                     # GFPGAN facial super-resolution\n",
                "    color_backdrop='#1A1A2E',              # Modern dark slate studio backdrop\n",
                "    add_subtitles=True,                    # Auto-generate & burn styled subtitles\n",
                "    interpolate_60fps=True,                # Upsample output video to 60fps via RIFE\n",
                ")\n",
                "\n",
                "print(f\"\\n🎬 Production Deliverable Ready at: {final_video}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-7-md"},
            "source": [
                "---\n",
                "## 7. 60fps Broadcast Motion Interpolation (RIFE v4.x)\n",
                "Upsamples 24fps AI video to smooth **60fps broadcast video** using RIFE optical-flow motion interpolation.\n",
                "Eliminates diffusion micro-stutter and temporal jitter in less than 8 seconds on Tesla T4."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-7-code"},
            "outputs": [],
            "source": [
                "from src.enhancers.motion_interpolator import MotionInterpolator\n",
                "import os\n",
                "\n",
                "interpolator = MotionInterpolator(target_fps=60)\n",
                "print(\"[*] Upsampling video to 60fps broadcast framerate with RIFE v4.x...\")\n",
                "\n",
                "input_candidate = final_video if 'final_video' in locals() and os.path.exists(final_video) else None\n",
                "if not input_candidate:\n",
                "    input_candidate = output_scene if 'output_scene' in locals() and os.path.exists(output_scene) else '/content/drive/MyDrive/video_gen_platform/outputs/divyansh_studio_master.mp4'\n",
                "\n",
                "out_dir = '/content/drive/MyDrive/video_gen_platform/outputs'\n",
                "os.makedirs(out_dir, exist_ok=True)\n",
                "output_60fps = os.path.join(out_dir, 'video_60fps_broadcast.mp4')\n",
                "\n",
                "if os.path.exists(input_candidate):\n",
                "    interp_result = interpolator.interpolate_60fps(\n",
                "        video_path=input_candidate,\n",
                "        output_path=output_60fps,\n",
                "        multiplier=2\n",
                "    )\n",
                "    print(f\"✅ 60fps video interpolated successfully: {interp_result}\")\n",
                "else:\n",
                "    print(f\"⚠️ Target video '{input_candidate}' not found on disk. Run Section 5 or 6 first.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-8-md"},
            "source": [
                "---\n",
                "## 8. Gemini Managed Agent Autonomous QA Supervisor & Scorecard Diagnostic\n",
                "Integrates the **Gemini Agent Supervisor** (`GeminiAgentSupervisor`) to:\n",
                "1. **Deconstruct creative briefs** into the 5-part cinematic formula for Diffusion Transformers (`[Subject] + [Setting] + [Camera Motion] + [Atmosphere] + [Fidelity]`).\n",
                "2. **Diagnose objective QA scorecards** (Audio SNR, face presence %, Laplacian sharpness, AV sync delta) to recommend actionable self-annealing parameter remediation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-8-code"},
            "outputs": [],
            "source": [
                "from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor\n",
                "import json\n",
                "\n",
                "supervisor = GeminiAgentSupervisor()\n",
                "\n",
                "# 1. Expand creative brief into 5-part cinematic formula\n",
                "brief = \"executive presenting quarterly tech earnings in modern glass boardroom\"\n",
                "expanded = supervisor.expand_creative_prompt(brief)\n",
                "print(\"🎬 [Gemini Supervisor] Expanded 5-Part Cinematic Prompt:\")\n",
                "print(f\"   Positive: {expanded['positive_prompt']}\")\n",
                "print(f\"   Negative: {expanded['negative_prompt']}\\n\")\n",
                "\n",
                "# 2. Analyze Quality Audit Scorecard & Generate Autonomous Remediation\n",
                "sample_scorecard = {\n",
                "    \"audio_metrics\": {\"snr_db\": 12.5, \"clipping_ratio\": 0.02, \"wpm\": 145.0},\n",
                "    \"video_metrics\": {\"face_detected_ratio\": 0.65, \"laplacian_sharpness\": 50.0},\n",
                "    \"sync_metrics\": {\"av_duration_delta_sec\": 0.45}\n",
                "}\n",
                "print(\"[*] Diagnosing QualityInspector audit scorecard...\")\n",
                "remediation = supervisor.diagnose_audit_scorecard(sample_scorecard)\n",
                "print(\"🛡️ [Gemini Supervisor] Autonomous Diagnostic Remediation Plan:\")\n",
                "print(json.dumps(remediation, indent=2))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-9-md"},
            "source": [
                "---\n",
                "## 9. Autonomous Quality Audit & Inline Deliverable Verification\n",
                "The engine automatically executes an objective quality inspection before you review the video:\n",
                "- **Audio Compliance:** Speech SNR, peak/RMS loudness, digital clipping %, speaking pace (WPM).\n",
                "- **Visual Quality:** Face detection rate %, facial Laplacian sharpness, motion stability, freeze detection.\n",
                "- **AV Sync:** Audio-video duration delta & lip-sync alignment.\n",
                "- **Remediation Advisories:** Actionable parameter fixes if thresholds fall below standard."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-9-code"},
            "outputs": [],
            "source": [
                "from IPython.display import HTML, display\n",
                "from base64 import b64encode\n",
                "import os\n",
                "import json\n",
                "from src.verification.inspector import QualityInspector\n",
                "\n",
                "video_to_play = final_video if 'final_video' in locals() and os.path.exists(final_video) else None\n",
                "if not video_to_play:\n",
                "    fallback = '/content/drive/MyDrive/video_gen_platform/outputs/divyansh_studio_master.mp4'\n",
                "    if os.path.exists(fallback):\n",
                "        video_to_play = fallback\n",
                "\n",
                "if video_to_play and os.path.exists(video_to_play):\n",
                "    # 1. Load or run Autonomous Verification Audit\n",
                "    audit_file = os.path.splitext(video_to_play)[0] + '_quality_audit.json'\n",
                "    if os.path.exists(audit_file):\n",
                "        with open(audit_file, 'r') as f:\n",
                "            audit_data = json.load(f)\n",
                "    else:\n",
                "        inspector = QualityInspector()\n",
                "        report = inspector.audit_deliverable(video_to_play, script_text=SCRIPT if 'SCRIPT' in locals() else '')\n",
                "        audit_data = report.to_dict()\n",
                "\n",
                "    # 2. Render Verification Scorecard HTML\n",
                "    status_color = '#10B981' if audit_data['status'] == 'PASSED' else '#F59E0B' if audit_data['status'] == 'NEEDS_REVIEW' else '#EF4444'\n",
                "    audio_m = audit_data.get('audio_metrics', {})\n",
                "    video_m = audit_data.get('video_metrics', {})\n",
                "    sync_m = audit_data.get('sync_metrics', {})\n",
                "\n",
                "    scorecard_html = f'''\n",
                "    <div style=\"font-family: system-ui, sans-serif; max-width: 680px; margin: 0 auto 20px auto; padding: 18px; border-radius: 12px; background: #0F172A; color: #F8FAFC; border: 1px solid #334155; box-shadow: 0 4px 14px rgba(0,0,0,0.4);\">\n",
                "      <div style=\"display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px;\">\n",
                "        <span style=\"font-size: 16px; font-weight: 700;\">🛡️ Autonomous Quality Verification Report</span>\n",
                "        <span style=\"background: {status_color}; color: #FFFFFF; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px;\">{audit_data['status']} ({audit_data['overall_score']:.1f}/100)</span>\n",
                "      </div>\n",
                "      <div style=\"display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 14px; text-align: center;\">\n",
                "        <div style=\"background: #1E293B; padding: 10px; border-radius: 8px;\">\n",
                "          <div style=\"font-size: 11px; color: #94A3B8;\">AUDIO SNR</div>\n",
                "          <div style=\"font-size: 16px; font-weight: 700; color: #38BDF8;\">{audio_m.get('snr_db', 0):.1f} dB</div>\n",
                "        </div>\n",
                "        <div style=\"background: #1E293B; padding: 10px; border-radius: 8px;\">\n",
                "          <div style=\"font-size: 11px; color: #94A3B8;\">FACE PRESENCE</div>\n",
                "          <div style=\"font-size: 16px; font-weight: 700; color: #34D399;\">{video_m.get('face_detection_rate', 100):.1f}%</div>\n",
                "        </div>\n",
                "        <div style=\"background: #1E293B; padding: 10px; border-radius: 8px;\">\n",
                "          <div style=\"font-size: 11px; color: #94A3B8;\">AV SYNC DELTA</div>\n",
                "          <div style=\"font-size: 16px; font-weight: 700; color: #A78BFA;\">{sync_m.get('av_duration_delta', 0):.2f}s</div>\n",
                "        </div>\n",
                "      </div>\n",
                "      <div style=\"margin-top: 12px; font-size: 12px; color: #94A3B8;\">\n",
                "        Clip: {audio_m.get('clipping_pct', 0):.3f}% | Sharpness: {video_m.get('sharpness', 0):.1f} | Pace: {audio_m.get('wpm', 0):.0f} WPM | Peak: {audio_m.get('peak_db', 0):.1f} dBFS\n",
                "      </div>\n",
                "    </div>\n",
                "    '''\n",
                "    display(HTML(scorecard_html))\n",
                "\n",
                "    # 3. Render HTML5 Video Player\n",
                "    mp4_bytes = open(video_to_play, 'rb').read()\n",
                "    data_url = 'data:video/mp4;base64,' + b64encode(mp4_bytes).decode()\n",
                "    display(HTML(f'''\n",
                "    <div style=\"text-align: center;\">\n",
                "      <video width=640 controls autoplay loop style=\"border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.4);\">\n",
                "        <source src=\"{data_url}\" type=\"video/mp4\">\n",
                "      </video>\n",
                "    </div>\n",
                "    '''))\n",
                "else:\n",
                "    print('⚠️ No video deliverable found to display. Please run Section 6 first.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "sec-10-md"},
            "source": [
                "---\n",
                "## 10. (Optional) Visual UI: ComfyUI + Geeky LatentSync Wrapper via Cloudflare Tunnel\n",
                "For users who want to experiment with interactive node graphs in browser.\n",
                "Launches ComfyUI and exposes a public HTTPS tunnel."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "sec-10-code"},
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
