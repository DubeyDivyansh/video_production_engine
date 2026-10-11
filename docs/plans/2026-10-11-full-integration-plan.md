# Full Text-to-Avatar Platform Integration Plan (Colab T4 Architecture)

> **Repository:** [`DubeyDivyansh/video_production_engine`](https://github.com/DubeyDivyansh/video_production_engine) (Private)  
> **Target Hardware:** Free Google Colab NVIDIA Tesla T4 GPU (15.3 GB VRAM)  
> **Persistence:** Google Drive (`/content/drive/MyDrive/video_gen_platform`)

---

## 🔍 Exploration Audit: What Was Analyzed & Built vs. What Is Pending

| Component / Directive | Target Repository | Exploration & Implementation Status | Notes & Findings |
| :--- | :--- | :--- | :--- |
| **1. Voice Cloning (Priority)** | [`SWivid/F5-TTS`](https://github.com/SWivid/F5-TTS) | ✅ **Fully Tested & Integrated** | Tested live on Colab T4. Synthesized `f5_tts_test.wav` (545 KB). Peak VRAM ~2.8 GB. Wrapped in `src.audio.f5_engine.py`. |
| **2. Voice Cloning (Fallback)** | [`coqui-ai/TTS`](https://github.com/coqui-ai/TTS) (XTTS-v2) | 🔍 **Architecture Analyzed** | Slower than F5-TTS, higher VRAM (~4.2 GB). Retained as secondary fallback if multilingual voice cloning is needed. |
| **3. Video Lip-Sync (Priority)** | [`bytedance/LatentSync`](https://github.com/bytedance/LatentSync) | ⚠️ **Analyzed & Patched (Ready for Step 3 Run)** | Discovered critical Turing bug (`capability > 7` forces FP32 OOM). Patched to `>= 7` for FP16 (~12.2 GB VRAM). Wrapped in `src.video.latentsync_engine.py`. **Live inference run pending in Step 3.** |
| **4. Lip-Sync (Alternative)** | [`TMElyralab/MuseTalk`](https://github.com/TMElyralab/MuseTalk) | 🔍 **Architecture Analyzed** | Latent inpainting model. Very lightweight on T4 (~5.2 GB VRAM). To be integrated as low-latency fallback. |
| **5. Avatar Motion (Fallback)** | [`KwaiVGI/LivePortrait`](https://github.com/KwaiVGI/LivePortrait) | 🔍 **Architecture Analyzed** | Best-in-class for animating still portraits with head motion, eye gaze, and blinking (~3.8 GB VRAM). |
| **6. Visual UI (Orchestration)** | [`comfyanonymous/ComfyUI`](https://github.com/comfyanonymous/ComfyUI) | 📋 **Evaluated for Step 4** | Web server baseline is ~900 MB - 1.2 GB VRAM. Safe on T4 if models are loaded sequentially. |
| **7. Geeky LatentSync Node** | [`GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper`](https://github.com/GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper) | 🔍 **Architecture Analyzed** | Uses gradient checkpointing + isolated paths to keep VRAM < 15GB. To be deployed in Step 4. |
| **8. VRAM Fence & Memory Guard** | Custom | ✅ **Fully Tested & Integrated** | `flush_vram()` flushes cache and drops memory to < 500 MB between stages. Tested in `src.utils.memory.py`. |
| **9. Google Drive Persistence** | Google Colab | ✅ **Fully Tested & Integrated** | Mounts `/content/drive/MyDrive/video_gen_platform/` for zero re-downloads of checkpoints. |
| **10. Dedicated Private Git Repo** | GitHub | ✅ **Created & Initialized** | Standalone private repo created: `DubeyDivyansh/video_production_engine`. |

---

## 🚀 Step-by-Step Integration & Execution Plan

### Phase 1: Step 3 Execution — LatentSync Live Video Run on Colab T4
**Goal:** Take the generated cloned voice (`f5_tts_test.wav`) and the user's base video (`VID20240712164337.mp4`), and execute LatentSync diffusion inference with whole-body gesture retention on Google Colab T4.

- [ ] **Task 1.1: Prepare Base Video in Colab Session**
  - Download `VID20240712164337.mp4` to `/content/base_video.mp4` and copy to Drive inputs.
  - Trim a 10–15s test segment to verify pipeline speed before rendering the full 58s clip.
- [ ] **Task 1.2: Install LatentSync & DeepCache on Colab**
  - Clone LatentSync, apply Turing FP16 patch (`capability >= 7`), and enable `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
  - Symlink / download checkpoints into `/content/drive/MyDrive/video_gen_platform/checkpoints/latentsync/`.
- [ ] **Task 1.3: Execute LatentSync Inference on T4**
  - Run `LatentSyncEngine.sync_lips()` with `f5_tts_test.wav` and base video.
  - Monitor `!nvidia-smi` to ensure peak VRAM stays within **12.5 GB / 15.3 GB**.
- [ ] **Task 1.4: Verify Artifact & Deliver**
  - Save output video to `/content/drive/MyDrive/video_gen_platform/outputs/latentsync_output.mp4`.
  - Download locally to `C:\Users\Acer\Downloads\latentsync_output.mp4`.
  - **[STOP Checkpoint 3]** Present Code Diff, VRAM metrics, and video preview for user review.

---

### Phase 2: Multi-Engine Fallback Expansion (LivePortrait & MuseTalk)
**Goal:** Expand `src/video/` so the engine can generate expressive head/body animation from still images (LivePortrait) and ultra-fast lightweight lip-sync (MuseTalk).

- [ ] **Task 2.1: Implement LivePortrait Engine (`src/video/liveportrait_engine.py`)**
  - Add wrapper class for KwaiVGI/LivePortrait.
  - Support animating still images (like `screenshot_face.jpg` or `custom_face.jpeg`) with realistic eye blinks, smiles, and head nods.
  - VRAM Footprint: ~3.8 GB VRAM.
- [ ] **Task 2.2: Implement MuseTalk Engine (`src/video/musetalk_engine.py`)**
  - Add wrapper class for TMElyralab/MuseTalk for near-real-time lip-sync.
  - VRAM Footprint: ~5.2 GB VRAM.
- [ ] **Task 2.3: Wire Multi-Engine Dispatcher in `src/pipeline.py`**
  - Add `engine="liveportrait"`, `engine="musetalk"`, `engine="latentsync"`, `engine="sadtalker"`.
  - Commit all engine additions to private GitHub repo.

---

### Phase 3: Step 4 Execution — ComfyUI & Geeky LatentSync Wrapper Evaluation
**Goal:** Provide an interactive visual node-based UI on Google Colab for users who prefer visual workflow tuning over Python scripts.

- [ ] **Task 3.1: ComfyUI Server Setup on Colab**
  - Write Colab deployment script for ComfyUI.
  - Expose UI using `cloudflared` (Cloudflare Tunnel) or `localtunnel` without open port exposure.
- [ ] **Task 3.2: Install Geeky LatentSync Wrapper Node**
  - Clone `GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper` into `ComfyUI/custom_nodes/`.
  - Configure gradient checkpointing and Low-VRAM mode.
- [ ] **Task 3.3: Benchmark T4 Memory Overhead**
  - Measure ComfyUI idle footprint (~900 MB) + LatentSync node execution.
  - Document performance trade-offs vs Headless Python Script.
- [ ] **Task 3.4: Provide ComfyUI Workflow JSON**
  - Export a pre-wired `.json` workflow connecting F5-TTS audio to LatentSync lip-sync node.
  - Commit workflow to `video_production_engine/workflows/`.

---

## 🔒 Verification & Quality Gates

1. **VRAM Ceiling:** Under no circumstances should any stage exceed 14.5 GB VRAM on Tesla T4.
2. **Drive Sync:** Every model download must be cached to Google Drive so restart time is < 60 seconds.
3. **Commit Discipline:** Every completed task must be committed and pushed to `DubeyDivyansh/video_production_engine`.
