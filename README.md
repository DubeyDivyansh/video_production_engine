<div align="center">

# 🎬 Video Production Engine

**Modular End-to-End AI Video Studio & Persistent Avatar Platform for Google Colab T4**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DubeyDivyansh/video_production_engine/blob/main/quick_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware Target](https://img.shields.io/badge/Hardware-NVIDIA_Tesla_T4_(16GB)-76B900?logo=nvidia&logoColor=white)](https://cloud.google.com/)
[![Repository: Public](https://img.shields.io/badge/Visibility-Public-green.svg)](https://github.com/DubeyDivyansh/video_production_engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  A decoupled, production-grade video synthesis pipeline tailored for <b>zero cloud costs</b> on <b>Google Colab's Free Tesla T4 GPU (16GB VRAM)</b> with persistent <b>Google Drive storage</b> and continuous avatar improvement.
</p>

</div>

---

## 🌟 Key Capabilities

* **🧠 Persistent Avatar Knowledge Base (`AvatarManager`):** User assets (images, videos, audio samples) are archived and automatically evaluated for quality (Audio SNR/clipping, image sharpness Laplacian variance, video optical flow stability). The engine picks optimal prompts and updates profile metadata in Google Drive so you **never restart from scratch** when adding new media.
* **⚡ Dual Speech Synthesis Engines:**
  * **Kokoro-82M (`KokoroEngine`):** Zero-VRAM / CPU narration generating 24kHz studio-grade speech in < 1 second with 82M parameters (Apache 2.0). Requires no reference audio and preserves all 16GB GPU VRAM for video diffusion.
  * **F5-TTS (`F5VoiceCloner`):** Flow-matching diffusion transformer providing zero-shot voice cloning from as little as 5 seconds of reference audio (~1.2 GB VRAM).
* **🎬 Multi-Mode Video Generation & Lip-Sync:**
  * **LTX-Video 0.9.5 & Wan2.1 (`GenerativeVideoEngine`):** Diffusion Transformer (DiT) text-to-video / image-to-video scene synthesis with a **Two-Phase Memory Offload Strategy** (Phase 1 text encoder embed -> Phase 2 Transformer + VAE with CPU offload & VAE tiling), generating high-fidelity scenes on Tesla T4 within 16GB VRAM.
  * **LatentSync (`LatentSyncEngine`):** Spatio-temporal latent diffusion lip-sync retaining 100% of body language, torso motion, and background at 1280x720.
  * **LivePortrait & SadTalker:** Photorealistic facial performance animation from single portrait photos.
* **✨ Studio Post-Production Suite:**
  * **RIFE v4.x Motion Interpolation (`MotionInterpolator`):** Optical-flow frame interpolation upsampling 24fps AI video to broadcast-standard **60fps**, eliminating diffusion micro-stutter in < 8s on T4.
  * **Background Matting & Replacement (`BackgroundReplacer`):** Deep alpha matting (BiRefNet) & chroma keying for custom backdrops or solid studio canvases.
  * **Face Super-Resolution (`FaceEnhancer`):** GFPGAN v1.4 facial upscaling and boundary blending.
  * **Audio Mastering (`AudioEnhancer`):** Spectral noise gating and EBU R128 (-16 LUFS) broadcast loudness normalization.
  * **Dynamic BGM & Subtitles (`StudioCompositor`):** Sidechain voice ducking and styled subtitle burn-in.
* **🤖 Autonomous QA Supervisor & Self-Annealing Loop (`GeminiAgentSupervisor` & `QualityInspector`):** Automated objective QA inspector that audits Audio (SNR, peak dBFS, RMS, digital clipping %, speaking pace WPM), Video (face presence %, Laplacian sharpness, motion stability, freeze frames), and AV sync. Powered by Gemini (`gemini-3.5-flash`), the supervisor analyzes audit scorecards and provides automated corrective remediation recommendations (e.g. spectral gating adjustments, bounding box retargeting, audio re-encoding).
* **🧹 Deterministic VRAM Memory Fence:** Automatically unloads heavy models, triggers garbage collection, and clears PyTorch CUDA cache between stages, preventing Out-Of-Memory (OOM) crashes on 16GB GPUs.
* **💾 Google Drive Persistence:** Checkpoints, avatar profiles, and rendered outputs are persisted in Google Drive so you never re-download models when ephemeral Colab sessions reset.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 0. Creative Orchestration & Asset Resolution                                 │
│ - AvatarManager: Resolves optimal scored audio, image, and motion from Drive│
│ - GeminiAgentSupervisor: Decomposes prompts into 5-part cinematic formula   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Speech Narration & Audio Mastering                                       │
│ [Path A: Kokoro-82M] 0 MB VRAM (CPU in <1s) - 24kHz Studio Narration        │
│ [Path B: F5-TTS]     ~1.2 GB VRAM (DiT)     - 5s Zero-Shot Voice Cloning    │
│ Studio Audio Mastering: noisereduce spectral gating + EBU R128 (-16 LUFS)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. VRAM Memory Fence (flush_vram)                                           │
│ Unloads audio models, executes gc.collect() & torch.cuda.empty_cache()       │
│ Memory: Resets GPU allocation to < 500 MB                                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ 3A. Avatar Animation Pathway             │   │ 3B. Generative Scene Pathway (DiT)       │
│ - LatentSync: Whole-body gesture sync    │   │ - LTX-Video 0.9.5 / Wan2.1 T2V & I2V     │
│ - LivePortrait: Expressive face reenact  │   │ - Two-Phase Memory Offload:              │
│ - SadTalker: 3DMM portrait harness       │   │   Phase 1: T5-XXL prompt embed -> PURGE  │
│ Memory: ~12.2 GB / 15.3 GB VRAM          │   │   Phase 2: Transformer + VAE CPU offload │
│                                          │   │ Memory: ~10.5 GB VRAM peak               │
└─────────────────────┬────────────────────┘   └────────────────────┬─────────────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │ [Optional Hybrid Composite]
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Studio Compositor & Post-Processing Suite                                │
│ - Face Restoration: GFPGAN v1.4 Super-Resolution (< 1.8 GB VRAM)            │
│ - Background Replacement: BiRefNet Alpha Matting / Chroma Key (< 1.1 GB VRAM│
│ - Motion Interpolation: RIFE v4.x (24fps -> 60fps Broadcast Smoothness)      │
│ - Audio-Visual Compositing: Sidechain BGM ducking + styled subtitle burn-in │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Autonomous Quality Verification & Diagnostic Feedback Loop                │
│ - QualityInspector: Objective Audio (SNR, clipping, WPM), Video (face %,    │
│   Laplacian sharpness), and AV sync delta audit (< 1.0s)                    │
│ - GeminiAgentSupervisor: Self-Annealing Diagnostic Loop                      │
│   Analyzes scorecard JSON -> Recommends actionable parameter remediations   │
│   (e.g., retarget bbox, adjust spectral threshold, audio retiming)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
      🎉 Mastered 60fps MP4 Deliverable & QA Audit Scorecard to Google Drive!
```

---

## 📊 Comprehensive Tools & Models Evaluation for Colab T4

| Pipeline Stage | Model / Tool | Status | Colab T4 Compatibility | Evaluation & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **TTS (Voice)** | **Kokoro-82M** | **KEEP (Primary Narration)** | ✅ 100% Compatible (0 MB VRAM) | 82M parameter lightweight TTS. Synthesizes 24kHz studio audio in <1s on CPU. Preserves all GPU memory for video models. Apache 2.0. |
| **TTS (Voice)** | **F5-TTS** | **KEEP (Primary Voice Cloning)** | ✅ 100% Compatible (~1.2 GB VRAM) | Flow-matching DiT with highest prosody fidelity. Requires only 5–8s reference audio. RTF ~0.15. |
| **TTS (Voice)** | **CosyVoice-2** | **ADD (Advanced)** | ✅ Compatible (~2.5 GB VRAM) | Alibaba SOTA multi-lingual diffusion speech with emotion control. Good future upgrade. |
| **TTS (Voice)** | **XTTS-v2** | **DROP as Primary** | ⚠️ Fits (~2.4 GB VRAM) | Slower, prone to word repetition & trailing silence hallucinations on long scripts. Kept only as multilingual fallback. |
| **TTS (Voice)** | **Edge-TTS / Piper** | **ADD (Drafts)** | ✅ 100% Compatible (0 MB VRAM) | Rapid zero-VRAM speech generation for instant previewing without GPU overhead. |
| **Scene Gen (DiT)** | **LTX-Video 0.9.5** | **KEEP (Primary DiT)** | ✅ Compatible with Two-Phase Offload (~10.5 GB VRAM) | Fast DiT video generation (Lightricks). Two-phase memory offload (Phase 1 text encoder -> Phase 2 Transformer + VAE) fits within 16GB T4. 8n+1 frames. |
| **Scene Gen (DiT)** | **Wan2.1 (1.3B)** | **ADD (Compact DiT)** | ✅ Compatible (~8.2 GB VRAM) | High motion dynamics at 480p/720p. 4n+1 frames. Excellent open-weights alternative. |
| **Lip-Sync (Video)** | **LatentSync** | **KEEP (Primary)** | ✅ Compatible with Patch (~12.2 GB VRAM) | Best-in-class for full-body video and natural torso gestures. Requires Turing FP16 patch + DeepCache to fit in 16GB. |
| **Lip-Sync (Video)** | **LivePortrait** | **KEEP / ADD** | ✅ 100% Compatible (~3.5 GB VRAM) | Photorealistic single-portrait reenactment (25–30 FPS). Vastly superior to SadTalker. |
| **Lip-Sync (Video)** | **SadTalker** | **DROP as Primary** | ✅ Compatible (~2.8 GB VRAM) | Fast, but rigid head motion and synthetic 3DMM distortion. Kept as lightweight test harness. |
| **Lip-Sync (Video)** | **MuseTalk** | **DROP from Pro** | ✅ Compatible (~4.0 GB VRAM) | 256x256 latent inpainting is fast (30 FPS), but lower mouth resolution and visible boundary seam. |
| **Lip-Sync (Video)** | **Wav2Lip** | **DROP Completely** | ❌ Outdated | 96x96 blurry mouth, obsolete GAN artifacts. |
| **Motion Interpolation** | **RIFE v4.x** | **KEEP (Primary 60fps)** | ✅ Compatible (< 1.5 GB VRAM) | Real-Time Intermediate Flow Estimation. Upsamples 24fps AI video to 60fps broadcast video in < 8s, eliminating micro-stutter. |
| **Matting** | **BiRefNet / RMBG-1.4** | **KEEP (Primary)** | ✅ Compatible (~1.1 GB VRAM) | Hair-strand level alpha matting at ~18 FPS in FP16 on T4. Eliminates need for green screen. |
| **Restoration** | **GFPGAN v1.4** | **KEEP (Primary)** | ✅ Compatible (~1.8 GB VRAM) | Rapid facial super-resolution (~18s for 300 frames). Restores eye, teeth, and skin clarity. |
| **Audio Mastering**| **noisereduce + pyloudnorm** | **KEEP (Primary)** | ✅ 100% Compatible (CPU only) | Eliminates mic hiss and normalizes loudness to broadcast EBU R128 (-16 LUFS) in 0.5s. |
| **Compositing** | **MoneyPrinterTurbo / FFmpeg** | **INTEGRATED** | ✅ 100% Compatible (CPU only) | Battle-tested BGM mixing with sidechain audio ducking and styled subtitle burn-in. |
| **Quality QA** | **QualityInspector** | **INTEGRATED** | ✅ 100% Compatible (CPU only) | Multi-modal heuristic QA layer (Audio SNR/clipping, Face presence, Sharpness, AV Sync). |
| **QA Supervisor** | **Gemini Managed Agents (`GeminiAgentSupervisor`)** | **INTEGRATED** | ✅ 100% Compatible (API / CPU) | Autonomous QA supervisor powered by `gemini-3.5-flash`. Decomposes briefs into 5-part cinematic prompts and provides self-annealing remediation plans from inspection scorecards. |

---

## 📁 Repository Structure

```
video_production_engine/
├── GEMINI.md                          # 🤖 Machine-readable agent directive & VRAM rules
├── README.md                          # 📖 Project documentation & API guide
├── requirements.txt                   # 📦 Production pip dependencies
├── quick_demo.ipynb                   # 🚀 All-in-One Interactive Colab Runner
├── 01_avatar_knowledge_base.ipynb     # 🧠 Avatar KB & Drive Asset Ingestion (0 GPU Quota)
├── 02_talking_head_studio.ipynb       # 👤 Fast 2D Talking-Head Studio (< 5 mins)
├── 03_fullbody_latentsync_studio.ipynb# 🕴️ Whole-Body LatentSync Studio (#FFFFFF White Backdrop)
├── notebooks/                         # Version-controlled notebook backups
│   ├── 01_avatar_knowledge_base.ipynb
│   ├── 02_talking_head_studio.ipynb
│   ├── 03_fullbody_latentsync_studio.ipynb
│   └── quick_demo.ipynb
├── scripts/
│   ├── generate_notebook.py           # All-in-one notebook generator
│   └── generate_modular_notebooks.py  # Modular suite generator
├── src/
│   ├── __init__.py
│   ├── audio/                         # Voice synthesis engines
│   │   ├── __init__.py
│   │   ├── f5_engine.py               # F5VoiceCloner wrapper (zero-shot voice cloning)
│   │   └── kokoro_engine.py           # Kokoro-82M zero-VRAM narration engine
│   ├── video/                         # Video & animation engines
│   │   ├── __init__.py
│   │   ├── fullbody_animator.py       # 🕴️ Whole-body white canvas & breathing motion
│   │   ├── generative_engine.py       # 🎬 LTX-Video & Wan2.1 two-phase DiT engine
│   │   ├── latentsync_engine.py       # LatentSync whole-body engine (sm_75 patched)
│   │   ├── liveportrait_engine.py     # LivePortrait photorealistic portrait engine
│   │   ├── sadtalker_engine.py        # SadTalker 3DMM + GFPGAN engine
│   │   └── musetalk_engine.py         # MuseTalk real-time engine
│   ├── avatar/                        # Persistent avatar knowledge base
│   │   ├── __init__.py
│   │   ├── profile.py                 # MediaAsset & AvatarProfile data models
│   │   └── manager.py                 # AvatarManager (continuous scoring & slicing)
│   ├── enhancers/                     # Studio post-production suite
│   │   ├── __init__.py
│   │   ├── background.py              # BiRefNet matting & chroma replacement
│   │   ├── face_enhancer.py           # GFPGAN facial super-resolution
│   │   ├── audio_enhancer.py          # Spectral denoiser & -16 LUFS loudness
│   │   ├── motion_interpolator.py     # ⚡ RIFE v4.x 60fps motion interpolator
│   │   └── studio_compositor.py       # Dynamic BGM ducking & subtitle burn-in
│   ├── orchestration/                 # 🤖 Agent supervisor & LLM director
│   │   ├── __init__.py
│   │   └── gemini_agent_supervisor.py # Gemini Managed Agent supervisor & self-annealing QA
│   ├── verification/                  # 🛡️ Autonomous Quality Inspection Layer
│   │   ├── __init__.py
│   │   └── inspector.py               # QualityInspector & VerificationReport
│   ├── utils/                         # Infrastructure & guards
│   │   ├── __init__.py
│   │   ├── colab_quota.py             # ⚡ Multi-account quota monitor & switchover alert
│   │   ├── memory.py                  # VRAM fence & allocator config
│   │   ├── storage.py                 # Google Drive persistence manager
│   │   └── media.py                   # FFmpeg prompt slicing & duration inspectors
│   └── pipeline.py                    # ⚡ VideoProductionPipeline (Top-level orchestrator)
└── tests/
    ├── test_agent_supervisor.py       # Unit tests for prompt expansion & QA diagnosis
    ├── test_audio_engines.py          # Unit tests for F5-TTS & Kokoro-82M
    ├── test_avatar_system.py          # Unit tests for avatar, enhancers, & inspector
    ├── test_fullbody_animator.py      # Unit tests for mod-16 padding & motion curves
    ├── test_generative_engine.py      # Unit tests for LTX-Video & Wan2.1 two-phase DiT
    ├── test_pipeline_modes.py         # Unit tests for Avatar, Generative Scene, & Hybrid modes
    └── test_post_enhancers.py         # Unit tests for RIFE 60fps & post enhancers
```

---

## 💻 1-Click Usage in Google Colab

Open [`quick_demo.ipynb`](https://colab.research.google.com/github/DubeyDivyansh/video_production_engine/blob/main/quick_demo.ipynb) or run the following in Colab:

```python
# 1. Clone repository & install dependencies
!git clone https://github.com/DubeyDivyansh/video_production_engine.git /content/app
%cd /content/app
!pip install -q -r requirements.txt

# 2. Add to Python path
import sys
sys.path.append("/content/app")
from src.pipeline import VideoProductionPipeline

# 3. Initialize pipeline with Google Drive persistence
pipeline = VideoProductionPipeline(use_drive=True)

# 4. Generate your video in 1 call using your persistent Avatar ID!
output_video = pipeline.produce_video(
    avatar_id="divyansh",                                  # Pulls optimal assets from Drive
    script_text="Hello! Welcome to our modular video production platform running on Colab!",
    output_name="divyansh_studio_master.mp4",
    engine="latentsync",                                   # 'latentsync' for body video, 'sadtalker' for photo, 'generative' for scene
    tts_engine="kokoro",                                   # 'kokoro' (0 MB VRAM) or 'f5' (voice cloning)
    denoise_audio=True,                                    # Studio spectral gating + -16 LUFS
    enhance_face=True,                                     # GFPGAN facial super-resolution
    color_backdrop="#1A1A2E",                              # Modern dark studio backdrop
    interpolate_60fps=True,                                # 60fps broadcast motion upsampling
    add_subtitles=True,                                    # Auto-burn styled subtitles
)
```

---

## ⚙️ Tesla T4 (sm_75) Engineering Patches

| Challenge | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **LatentSync T4 OOM** | Official code checks `capability[0] > 7`. Turing T4 is `(7, 5)`, which evaluates to `False`, forcing FP32 (>20GB). | Auto-patched to `>= 7`, enforcing `torch.float16` (~12.2 GB VRAM). |
| **FlashAttention Error** | FlashAttention-2 requires Ampere (`sm_80+`) and crashes on Turing. | Enforced PyTorch native Scaled Dot-Product Attention (`enable_mem_efficient_sdp=True`). |
| **Diffusion DiT OOM** | Running text encoder and Transformer + VAE simultaneously exceeds 16GB VRAM. | Implemented **Two-Phase Memory Offload Strategy**: Purge text encoder after Phase 1, then load Transformer + VAE with CPU offload and tiling. |
| **Slow Diffusion Steps** | U-Net evaluates all 20-30 diffusion steps sequentially. | Integrated **DeepCache** (`cache_interval=2`) to accelerate inference by ~40%. |
| **Drive Caching** | Downloading 6GB weights on every Colab session wastes bandwidth. | Checkpoints automatically persist to `/content/drive/MyDrive/video_gen_platform/checkpoints/`. |

---

## 🖥️ ComfyUI Colab Migration Evaluation (Report)

We evaluated running GeekyGhost's `ComfyUI-Geeky-LatentSyncWrapper` on Colab Free T4:
* **VRAM Overhead:** ComfyUI server consumes ~1.2 GB VRAM. LatentSync with gradient checkpointing takes ~12.8 GB VRAM. Total peak: **~14.0 GB / 15.3 GB**.
* **Tunneling:** Cloudflare tunnel (`cloudflared`) successfully exposes the web UI with HTTPS without requiring paid accounts or tokens.
* **Trade-Off Analysis:**
  * **Headless Python Pipeline (Recommended):** Fast startup (< 1 min), deterministic, automated, zero-OOM memory fenced.
  * **ComfyUI WebUI (Optional):** Slower startup (6–8 minutes on fresh Colab), higher risk of OOM if extra nodes are attached, but gives visual drag-and-drop node graph.
  * **Verdict:** We provide the **Headless Python Pipeline as the primary production engine**, and include a 1-click ComfyUI launch cell in Section 10 of `quick_demo.ipynb` for visual exploration.

## ⚡ Colab Resource Optimization & Limits Guide

For complete details on navigating Google Colab's **90-minute idle timeouts**, **12-hour hard VM caps**, **3.5–5 hour daily GPU allocations**, **multi-account Drive sharing rotation**, and zero-waste memory fencing:

👉 **Read the full blueprint:** [docs/COLAB_RESOURCE_OPTIMIZATION_GUIDE.md](docs/COLAB_RESOURCE_OPTIMIZATION_GUIDE.md)

---

## 🚀 Scalability & Future Roadmap

For the detailed multi-cloud compute architecture, Kaggle backend integration (30h/wk dual T4 GPUs, headless REST API, and zero-download dataset cold starts), and multi-cloud failover dispatcher:

👉 **Read the full roadmap:** [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md)

---

## 🤖 Agentic Directive

For AI assistants and autonomous coding agents working on this repository, please review [`GEMINI.md`](GEMINI.md) for architectural rules, VRAM budget thresholds, and execution protocols.
