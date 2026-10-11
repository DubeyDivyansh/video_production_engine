# GEMINI.md - Video Production Engine Operating Directive

> Machine-Readable Architecture & Agent Operational Guidelines for Google Colab Execution.

## 🎯 Engine Mission
The `video_production_engine` is a modular, production-ready framework for generating personalized AI videos (voice cloning, talking head, and whole-body gesture-preserving lip-sync) using **Google Colab's Free NVIDIA Tesla T4 GPU (16GB VRAM)**.

## 🏗️ 3-Layer Architecture Mapping

### Layer 1: Directive (What to do)
- Defined in this `GEMINI.md` and `README.md`.
- Workflow:
  1. Mount Google Drive (`/content/drive/MyDrive/video_gen_platform`) for persistence.
  2. Generate synthetic speech using **F5-TTS** (`src.audio.f5_engine`).
  3. Purge GPU memory via Memory Fence (`src.utils.memory.flush_vram`).
  4. Generate animation / lip-sync via **LatentSync** (`src.video.latentsync_engine`) or **SadTalker** (`src.video.sadtalker_engine`).
  5. Export artifacts to Google Drive and render inline HTML5 preview.

### Layer 2: Orchestration (Decision-making)
- Handled by `src.pipeline.VideoProductionPipeline` and the Colab notebook (`quick_demo.ipynb`).
- Enforces strict VRAM isolation between audio and video models to avoid Out-Of-Memory (OOM) errors.

### Layer 3: Deterministic Execution (Doing the work)
- `src/audio/f5_engine.py`: F5-TTS zero-shot flow-matching voice cloning.
- `src/video/latentsync_engine.py`: Latent diffusion lip-sync retaining full body movement.
- `src/video/sadtalker_engine.py`: 3D facial motion generation with GFPGAN enhancement.
- `src/utils/memory.py`: VRAM garbage collection and PyTorch cache management.
- `src/utils/storage.py`: Google Drive workspace and model cache mounting.
- `src/utils/media.py`: Audio cutting, prompt trimming, and video format handling.

## ⚡ Hardware Constraints & Critical Patches (NVIDIA Tesla T4)

1. **Compute Capability 7.5 Bug:**
   - In LatentSync, `is_fp16_supported = torch.cuda.get_device_capability()[0] > 7` fails on T4 (`capability = (7, 5)`).
   - **Mandate:** Always force `torch.float16` and apply the `>= 7` patch in `latentsync_engine.py`.
2. **FlashAttention vs SDPA:**
   - Turing (T4) does NOT support FlashAttention-2.
   - **Mandate:** Always use PyTorch native `enable_mem_efficient_sdp(True)`.
3. **DeepCache:**
   - Always initialize DeepCache with `cache_interval=2` or `3` to save ~40% inference time and dampen VRAM spikes.
4. **VRAM Budget:**
   - Audio Phase: Maximum budget = **3.5 GB VRAM**.
   - Video Phase: Maximum budget = **13.5 GB VRAM**.
   - Always run `flush_vram()` at phase boundaries.
