<div align="center">

# 🎬 Video Production Engine

**Modular End-to-End AI Video & Voice Generation Platform for Google Colab T4**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DubeyDivyansh/video_production_engine/blob/main/quick_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware Target](https://img.shields.io/badge/Hardware-NVIDIA_Tesla_T4_(16GB)-76B900?logo=nvidia&logoColor=white)](https://cloud.google.com/)
[![Repository: Private](https://img.shields.io/badge/Visibility-Private-red.svg)](https://github.com/DubeyDivyansh/video_production_engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  A decoupled, production-grade video synthesis pipeline tailored for <b>zero cloud costs</b> on <b>Google Colab's Free Tesla T4 GPU</b> with persistent <b>Google Drive storage</b>.
</p>

</div>

---

## 🌟 Key Capabilities

* **🎙️ Zero-Shot Voice Cloning (F5-TTS):** Flow-matching diffusion transformer that synthesizes natural speech in your voice tone using as little as 5 seconds of reference audio.
* **🎞️ 3D Facial Pose Animation (SadTalker):** Generates realistic 3D talking head animations from single portrait photos with GFPGAN face restoration.
* **🕺 Whole-Body Gesture Lip-Sync (LatentSync):** Latent diffusion spatio-temporal lip-sync model that retains 100% of body language, torso motion, and background.
* **🧹 Deterministic VRAM Memory Fence:** Automatically flushes PyTorch CUDA cache between stages, preventing Out-Of-Memory (OOM) crashes on 16GB GPUs.
* **💾 Google Drive Persistence:** Heavy model weights (~5.5 GB) and output videos are automatically persisted in Google Drive so you never re-download models when Colab resets.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Voice Synthesis (F5-TTS)                                 │
│ Inputs: reference_audio.wav + custom_script_text            │
│ Memory: ~2.8 GB VRAM                                        │
└──────────────────────────────┬──────────────────────────────┘
                               │ (WAV generated & saved)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. VRAM Memory Fence (flush_vram)                           │
│ Unloads F5-TTS, runs gc.collect() & torch.cuda.empty_cache()│
│ Memory: Resets GPU to < 500 MB                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Animation Engine (SadTalker / LatentSync)                │
│ Inputs: face_photo.jpg (or body_video.mp4) + synthesized.wav│
│ Optimizations: Turing sm_75 FP16 Patch + DeepCache          │
│ Memory: ~12.2 GB / 15.3 GB VRAM                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            Final Synchronized MP4 Saved to Drive!
```

---

## 📁 Repository Structure

```
video_production_engine/
├── GEMINI.md                          # 🤖 Machine-readable agent directive
├── README.md                          # 📖 Project documentation & API guide
├── requirements.txt                   # 📦 Production pip dependencies
├── quick_demo.ipynb                   # 🚀 1-Click Interactive Colab Runner
├── notebooks/
│   └── quick_demo.ipynb               # Colab notebook backup
└── src/
    ├── __init__.py
    ├── audio/                         # Voice synthesis engine
    │   ├── __init__.py
    │   └── f5_engine.py               # F5VoiceCloner wrapper
    ├── video/                         # Video & animation engines
    │   ├── __init__.py
    │   ├── sadtalker_engine.py        # SadTalker 3DMM + GFPGAN engine
    │   └── latentsync_engine.py       # LatentSync whole-body engine
    ├── utils/                         # Infrastructure & guards
    │   ├── __init__.py
    │   ├── memory.py                  # VRAM fence & allocator config
    │   ├── storage.py                 # Google Drive persistence manager
    │   └── media.py                   # FFmpeg prompt slicing & audio extractors
    └── pipeline.py                    # ⚡ VideoProductionPipeline (Top-level orchestrator)
```

---

## 💻 1-Click Usage in Google Colab

Open the notebook directly on Google Colab or run this 4-line snippet in any fresh Colab GPU cell:

```python
# 1. Clone repository & install dependencies
!git clone https://github.com/DubeyDivyansh/video_production_engine.git /content/app
%cd /content/app
!pip install -q -r requirements.txt

# 2. Add to Python path and import unified pipeline
import sys
sys.path.append("/content/app")
from src.pipeline import VideoProductionPipeline

# 3. Initialize pipeline with Google Drive persistence
pipeline = VideoProductionPipeline(use_drive=True)

# 4. Generate your video in 1 call!
output_video = pipeline.produce_video(
    face_image_path="/content/custom_face.jpeg",
    ref_audio_path="/content/custom_audio.wav",
    script_text="Hello! Welcome to our modular video production platform running on Colab!",
    output_name="my_first_avatar.mp4",
    engine="sadtalker"  # or 'latentsync' for full-body video
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

## 🤖 Agentic Directive

For AI assistants and autonomous coding agents working on this repository, please review [`GEMINI.md`](GEMINI.md) for architectural rules, VRAM budget thresholds, and execution protocols.
