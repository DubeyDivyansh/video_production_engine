# GEMINI.md - Video Production Engine Operating Directive

> Machine-Readable Architecture & Agent Operational Guidelines for Google Colab Execution.

## 🎯 Engine Mission
The `video_production_engine` is a modular, production-ready framework for generating personalized AI videos (voice cloning, talking head, whole-body gesture-preserving lip-sync, and studio post-processing) using **Google Colab's Free NVIDIA Tesla T4 GPU (16GB VRAM)** with zero cloud costs.

## 🏗️ 3-Layer Architecture Mapping

### Layer 1: Directive (What to do)
- Defined in this `GEMINI.md` and `README.md`.
- Workflow:
  1. Mount Google Drive (`/content/drive/MyDrive/video_gen_platform`) for persistence.
  2. Resolve or register avatar assets via **AvatarManager** (`src.avatar.manager`).
  3. Synthesize speech via **F5-TTS** (`src.audio.f5_engine`) and master audio to EBU R128 (-16 LUFS) via **AudioEnhancer**.
  4. Purge GPU memory via Memory Fence (`src.utils.memory.flush_vram`).
  5. Generate animation via **LatentSync** (`src.video.latentsync_engine`) or **LivePortrait** / **SadTalker**.
  6. Apply studio enhancers: **FaceEnhancer** (GFPGAN), **BackgroundReplacer** (BiRefNet/Chroma), and **StudioCompositor** (sidechain BGM ducking + subtitles).
  7. Export deliverables to Google Drive and render inline HTML5 preview.

### Layer 2: Orchestration (Decision-making)
- Handled by `src.pipeline.VideoProductionPipeline` and the Colab notebook (`quick_demo.ipynb`).
- Enforces strict VRAM isolation between audio and video models to prevent Out-Of-Memory (OOM) errors.
- Handles automatic asset resolution: if user provides `avatar_id`, the pipeline pulls the optimal scored assets without manual path wiring.

### Layer 3: Deterministic Execution (Doing the work)
- `src/avatar/manager.py`: Ingestion, scoring (SNR, Laplacian variance, optical flow), and optimal slice extraction.
- `src/avatar/profile.py`: JSON-serializable `AvatarProfile` dataclass persisted in Drive.
- `src/audio/f5_engine.py`: F5-TTS zero-shot flow-matching voice cloning.
- `src/enhancers/audio_enhancer.py`: Spectral noise reduction and EBU R128 loudness normalization.
- `src/video/latentsync_engine.py`: Latent diffusion lip-sync retaining full body movement.
- `src/video/liveportrait_engine.py`: Photorealistic single-image portrait reenactment.
- `src/video/sadtalker_engine.py`: 3D facial motion generation with GFPGAN enhancement.
- `src/enhancers/background.py`: Deep alpha matting (BiRefNet) and chroma key background replacement.
- `src/enhancers/face_enhancer.py`: GFPGAN facial super-resolution and border softening.
- `src/enhancers/studio_compositor.py`: Sidechain BGM ducking and styled subtitle burn-in.
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
   - Post-Processing Phase: Maximum budget = **2.5 GB VRAM**.
   - Always run `flush_vram()` at phase boundaries.
