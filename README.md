<div align="center">

# 🎬 Video Production Engine

**Modular End-to-End AI Video Studio & Persistent Avatar Platform for Google Colab T4**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DubeyDivyansh/video_production_engine/blob/main/quick_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware Target](https://img.shields.io/badge/Hardware-NVIDIA_Tesla_T4_(16GB)-76B900?logo=nvidia&logoColor=white)](https://cloud.google.com/)
[![Repository: Private](https://img.shields.io/badge/Visibility-Private-red.svg)](https://github.com/DubeyDivyansh/video_production_engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  A decoupled, production-grade video synthesis pipeline tailored for <b>zero cloud costs</b> on <b>Google Colab's Free Tesla T4 GPU (16GB VRAM)</b> with persistent <b>Google Drive storage</b> and continuous avatar improvement.
</p>

</div>

---

## 🌟 Key Capabilities

* **🧠 Persistent Avatar Knowledge Base (`AvatarManager`):** User assets (images, videos, audio samples) are archived and automatically evaluated for quality (Audio SNR/clipping, image sharpness Laplacian variance, video optical flow stability). The engine picks the optimal prompts and updates profile metadata in Google Drive so you **never restart from scratch** when adding new media.
* **🎙️ Zero-Shot Voice Cloning (F5-TTS):** Flow-matching diffusion transformer that synthesizes natural speech in your voice tone using as little as 5 seconds of reference audio.
* **🕺 Whole-Body Gesture Lip-Sync (LatentSync):** Spatio-temporal latent diffusion lip-sync model that retains 100% of body language, torso motion, and background at 1280x720.
* **🎭 Photorealistic Facial Animation (LivePortrait & SadTalker):** Animates single portrait photos with realistic expressions, head movement, and gaze dynamics.
* **✨ Studio Post-Production Enhancers:**
  * **Background Matting & Replacement (`BackgroundReplacer`):** Deep alpha matting (BiRefNet) & chroma keying for custom backdrops or transparent video.
  * **Face Super-Resolution (`FaceEnhancer`):** GFPGAN v1.4 facial upscaling and boundary blending.
  * **Audio Mastering (`AudioEnhancer`):** Spectral noise gating and EBU R128 (-16 LUFS) broadcast loudness normalization.
  * **Dynamic BGM & Subtitles (`StudioCompositor`):** Sidechain voice ducking (lowers background music during speech) and styled subtitle burn-in.
* **🧹 Deterministic VRAM Memory Fence:** Automatically unloads heavy models, triggers garbage collection, and clears PyTorch CUDA cache between stages, preventing Out-Of-Memory (OOM) crashes on 16GB GPUs.
* **💾 Google Drive Persistence:** Checkpoints (~5.5 GB), avatar profiles, and rendered outputs are persisted in Google Drive so you never re-download models when ephemeral Colab sessions reset.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 0. Avatar Profile Resolution (AvatarManager)                │
│ Google Drive: /content/drive/MyDrive/video_gen_platform/    │
│ Automatically scores & selects cleanest audio, image, video │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Voice Synthesis (F5-TTS)                                 │
│ Inputs: optimal_audio.wav + custom_script_text              │
│ Memory: ~1.2 GB VRAM (Peak)                                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Studio Audio Mastering (noisereduce + pyloudnorm)        │
│ Spectral noise gating + EBU R128 (-16 LUFS) broadcast audio │
│ Memory: 0 MB VRAM (CPU Execution in 0.5s)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. VRAM Memory Fence (flush_vram)                           │
│ Unloads F5-TTS, runs gc.collect() & torch.cuda.empty_cache()│
│ Memory: Resets GPU to < 500 MB                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Video Animation Engine (LatentSync / LivePortrait)       │
│ Inputs: base_video.mp4 + mastered_audio.wav                 │
│ Optimizations: Turing sm_75 FP16 Patch + DeepCache          │
│ Memory: ~12.2 GB / 15.3 GB VRAM                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Studio Compositor & Post-Processing                      │
│ - GFPGAN Face Super-Resolution & Smoothing                  │
│ - BiRefNet / Chroma Background Replacement                  │
│ - Sidechain BGM Mixing with Automatic Voice Ducking         │
│ - Styled Subtitle Generation & Burn-In                      │
│ Memory: < 2.0 GB VRAM                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
      🎉 Final Mastered MP4 Persisted to Google Drive!
```

---

## 📊 Comprehensive Tools & Models Evaluation for Colab T4

| Pipeline Stage | Model / Tool | Status | Colab T4 Compatibility | Evaluation & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **TTS (Voice)** | **F5-TTS** | **KEEP (Primary)** | ✅ 100% Compatible (~1.2 GB VRAM) | Flow-matching DiT with highest prosody fidelity. Requires only 5–8s reference audio. RTF ~0.15. |
| **TTS (Voice)** | **CosyVoice-2** | **ADD (Advanced)** | ✅ Compatible (~2.5 GB VRAM) | Alibaba SOTA multi-lingual diffusion speech with emotion control. Good future upgrade. |
| **TTS (Voice)** | **XTTS-v2** | **DROP as Primary** | ⚠️ Fits (~2.4 GB VRAM) | Slower, prone to word repetition & trailing silence hallucinations on long scripts. Kept only as multilingual fallback. |
| **TTS (Voice)** | **Edge-TTS / Piper** | **ADD (Drafts)** | ✅ 100% Compatible (0 MB VRAM) | Rapid zero-VRAM speech generation for instant previewing without GPU overhead. |
| **Lip-Sync (Video)** | **LatentSync** | **KEEP (Primary)** | ✅ Compatible with Patch (~12.2 GB VRAM) | Best-in-class for full-body video and natural torso gestures. Requires Turing FP16 patch + DeepCache to fit in 16GB. |
| **Lip-Sync (Video)** | **LivePortrait** | **KEEP / ADD** | ✅ 100% Compatible (~3.5 GB VRAM) | Photorealistic single-portrait reenactment (25–30 FPS). Vastly superior to SadTalker. |
| **Lip-Sync (Video)** | **SadTalker** | **DROP as Primary** | ✅ Compatible (~2.8 GB VRAM) | Fast, but rigid head motion and synthetic 3DMM distortion. Kept as lightweight test harness. |
| **Lip-Sync (Video)** | **MuseTalk** | **DROP from Pro** | ✅ Compatible (~4.0 GB VRAM) | 256x256 latent inpainting is fast (30 FPS), but lower mouth resolution and visible boundary seam. |
| **Lip-Sync (Video)** | **Wav2Lip** | **DROP Completely** | ❌ Outdated | 96x96 blurry mouth, obsolete GAN artifacts. |
| **Matting** | **BiRefNet / RMBG-1.4** | **KEEP (Primary)** | ✅ Compatible (~1.1 GB VRAM) | Hair-strand level alpha matting at ~18 FPS in FP16 on T4. Eliminates need for green screen. |
| **Restoration** | **GFPGAN v1.4** | **KEEP (Primary)** | ✅ Compatible (~1.8 GB VRAM) | Rapid facial super-resolution (~18s for 300 frames). Restores eye, teeth, and skin clarity. |
| **Audio Mastering**| **noisereduce + pyloudnorm** | **KEEP (Primary)** | ✅ 100% Compatible (CPU only) | Eliminates mic hiss and normalizes loudness to broadcast EBU R128 (-16 LUFS) in 0.5s. |
| **Compositing** | **MoneyPrinterTurbo / FFmpeg** | **INTEGRATED** | ✅ 100% Compatible (CPU only) | Battle-tested BGM mixing with sidechain audio ducking and styled subtitle burn-in. |

---

## 📁 Repository Structure

```
video_production_engine/
├── GEMINI.md                          # 🤖 Machine-readable agent directive & VRAM rules
├── README.md                          # 📖 Project documentation & API guide
├── requirements.txt                   # 📦 Production pip dependencies
├── quick_demo.ipynb                   # 🚀 1-Click Interactive Colab Runner
├── notebooks/
│   └── quick_demo.ipynb               # Colab notebook backup
├── scripts/
│   └── generate_notebook.py           # Notebook generator & validator
├── src/
│   ├── __init__.py
│   ├── audio/                         # Voice synthesis engines
│   │   ├── __init__.py
│   │   └── f5_engine.py               # F5VoiceCloner wrapper
│   ├── video/                         # Video & animation engines
│   │   ├── __init__.py
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
│   │   └── studio_compositor.py       # Dynamic BGM ducking & subtitle burn-in
│   ├── utils/                         # Infrastructure & guards
│   │   ├── __init__.py
│   │   ├── memory.py                  # VRAM fence & allocator config
│   │   ├── storage.py                 # Google Drive persistence manager
│   │   └── media.py                   # FFmpeg prompt slicing & duration inspectors
│   └── pipeline.py                    # ⚡ VideoProductionPipeline (Top-level orchestrator)
└── tests/
    └── test_avatar_system.py          # Unit tests for avatar manager & studio suite
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
    engine="latentsync",                                   # 'latentsync' for body video, 'sadtalker' for photo
    denoise_audio=True,                                    # Studio spectral gating + -16 LUFS
    enhance_face=True,                                     # GFPGAN facial super-resolution
    color_backdrop="#1A1A2E",                              # Modern dark studio backdrop
    add_subtitles=True,                                    # Auto-burn styled subtitles
)
```

---

## ⚙️ Tesla T4 (sm_75) Engineering Patches

| Challenge | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **LatentSync T4 OOM** | Official code checks `capability[0] > 7`. Turing T4 is `(7, 5)`, which evaluates to `False`, forcing FP32 (>20GB). | Auto-patched to `>= 7`, enforcing `torch.float16` (~12.2 GB VRAM). |
| **FlashAttention Error** | FlashAttention-2 requires Ampere (`sm_80+`) and crashes on Turing. | Enforced PyTorch native Scaled Dot-Product Attention (`enable_mem_efficient_sdp=True`). |
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
  * **Verdict:** We provide the **Headless Python Pipeline as the primary production engine**, and include a 1-click ComfyUI launch cell in Section 6 of `quick_demo.ipynb` for visual exploration.

---

## 🤖 Agentic Directive

For AI assistants and autonomous coding agents working on this repository, please review [`GEMINI.md`](GEMINI.md) for architectural rules, VRAM budget thresholds, and execution protocols.
