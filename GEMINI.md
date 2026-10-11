# GEMINI.md - Video Production Engine Operating Directive

> Machine-Readable Architecture & Agent Operational Guidelines for Google Colab Execution.

## 🎯 Engine Mission
The `video_production_engine` is a modular, production-ready framework for generating personalized AI videos (voice cloning, talking head, whole-body gesture-preserving lip-sync, text-to-video diffusion scene generation, and studio post-processing) using **Google Colab's Free NVIDIA Tesla T4 GPU (16GB VRAM)** with zero cloud costs.

## 🏗️ 3-Layer Architecture Mapping

### Layer 1: Directive (What to do)
- Defined in this `GEMINI.md` and `README.md`.
- Workflow:
  1. Mount Google Drive (`/content/drive/MyDrive/video_gen_platform`) for persistence.
  2. Resolve or register avatar assets via **AvatarManager** (`src.avatar.manager`).
  3. Synthesize speech via **Kokoro-82M** (`src.audio.kokoro_engine`) or **F5-TTS** (`src.audio.f5_engine`) and master audio to EBU R128 (-16 LUFS) via **AudioEnhancer**.
  4. Purge GPU memory via Memory Fence (`src.utils.memory.flush_vram`).
  5. Generate video scene or animation via **LTX-Video / Wan2.1** (`src.video.generative_engine`), **LatentSync** (`src.video.latentsync_engine`), **LivePortrait**, or **SadTalker**.
  6. Apply studio enhancers: **FaceEnhancer** (GFPGAN), **BackgroundReplacer** (BiRefNet/Chroma), **MotionInterpolator** (RIFE 60fps), and **StudioCompositor** (sidechain BGM ducking + subtitles).
  7. Audit quality via **QualityInspector** and autonomous diagnosis via **GeminiAgentSupervisor**.
  8. Export deliverables to Google Drive and render inline HTML5 preview.

### Layer 2: Orchestration (Decision-making)
- Handled by `src.pipeline.VideoProductionPipeline`, `src.orchestration.gemini_agent_supervisor.GeminiAgentSupervisor`, and the Colab notebook (`quick_demo.ipynb`).
- Enforces strict VRAM isolation between audio, generative video, and post-processing stages to prevent Out-Of-Memory (OOM) errors.
- Handles multi-mode dispatching:
  - **Avatar Mode:** Lip-sync avatar onto base video/photo using LatentSync / LivePortrait / SadTalker.
  - **Generative Scene Mode:** Two-phase DiT scene synthesis using LTX-Video 0.9.5 or Wan2.1.
  - **Hybrid Mode:** Combines generative scene background with avatar overlay.
- Autonomous Self-Annealing QA loop: Diagnoses objective scorecard heuristics (SNR, clipping, face detection %, sharpness, AV delta) and issues corrective adjustments.
- Model routing adheres to Gemini API model endpoints:
  - `gemini-3.5-flash`: Complex reasoning, prompt decomposition, and architectural supervision.
  - `gemini-3.5-flash-lite`: Quick lookups, formatting, and fast JSON scoring.

### Layer 3: Deterministic Execution (Doing the work)
- `src/avatar/manager.py`: Ingestion, scoring (SNR, Laplacian variance, optical flow), and optimal slice extraction.
- `src/avatar/profile.py`: JSON-serializable `AvatarProfile` dataclass persisted in Drive.
- `src/audio/kokoro_engine.py`: Kokoro-82M high-speed zero-VRAM narration engine (0 MB VRAM, CPU in < 1s, Apache 2.0).
- `src/audio/f5_engine.py`: F5-TTS zero-shot flow-matching voice cloning (~1.2 GB VRAM).
- `src/enhancers/audio_enhancer.py`: Spectral noise reduction and EBU R128 loudness normalization.
- `src/video/generative_engine.py`: LTX-Video 0.9.5 and Wan2.1 Two-Phase DiT generative video engine (~10.5 GB VRAM peak).
- `src/video/latentsync_engine.py`: Latent diffusion lip-sync retaining full body movement (~12.2 GB VRAM).
- `src/video/liveportrait_engine.py`: Photorealistic single-image portrait reenactment (~3.5 GB VRAM).
- `src/video/sadtalker_engine.py`: 3D facial motion generation with GFPGAN enhancement (~2.8 GB VRAM).
- `src/enhancers/background.py`: Deep alpha matting (BiRefNet) and chroma key background replacement (~1.1 GB VRAM).
- `src/enhancers/face_enhancer.py`: GFPGAN facial super-resolution and border softening (~1.8 GB VRAM).
- `src/enhancers/motion_interpolator.py`: RIFE v4.x 60fps optical-flow frame interpolation (< 1.5 GB VRAM).
- `src/enhancers/studio_compositor.py`: Sidechain BGM ducking and styled subtitle burn-in (CPU / FFmpeg).
- `src/orchestration/gemini_agent_supervisor.py`: Gemini-driven 5-part prompt expansion and autonomous scorecard remediation.
- `src/verification/inspector.py`: Heuristic audio, video, and AV sync metric analysis.
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
3. **Diffusion DiT Two-Phase Offload (LTX-Video / Wan2.1):**
   - Running T5-XXL text encoder simultaneously with DiT Transformer and VAE exceeds 16GB VRAM.
   - **Mandate:** Apply Two-Phase Memory Offload:
     - Phase 1: Compute prompt embeddings using T5-XXL -> immediately delete and flush VRAM (`del text_encoder; flush_vram()`).
     - Phase 2: Load Transformer + VAE with `pipe.enable_model_cpu_offload()` and `pipe.vae.enable_tiling()` (Peak VRAM: ~10.5 GB).
     - Temporal compression constraint: LTX-Video requires **8n + 1 frames** (e.g. 41 frames) and dimensions divisible by 32. Wan2.1 requires **4n + 1 frames**.
4. **DeepCache:**
   - Always initialize DeepCache with `cache_interval=2` or `3` to save ~40% inference time and dampen VRAM spikes.
5. **VRAM Budget:**
   - Audio Phase: Kokoro-82M (**0 MB VRAM**) / F5-TTS (Peak = **1.2 GB VRAM**).
   - Video Diffusion / Animation Phase: Maximum budget = **13.5 GB VRAM** (LTX-Video ~10.5 GB, LatentSync ~12.2 GB).
   - Post-Processing Phase: Maximum budget = **2.5 GB VRAM** (RIFE < 1.5 GB, GFPGAN ~1.8 GB).
   - Always run `flush_vram()` at phase boundaries.
6. **Gemini API Model Usage Guidelines:**
   - Only utilize the following two Gemini endpoints:
     1. `gemini-3.5-flash`: Complex reasoning, creative direction, prompt expansion, and QA diagnosis.
     2. `gemini-3.5-flash-lite`: Small tasks, quick edits, lookups, and fast JSON scoring.
