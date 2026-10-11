# Full-Body LatentSync & Modular Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated full-body video generation subsystem driven by LatentSync and BiRefNet on white studio backdrops, and decompose the Colab architecture into 3 modular, sustainable notebooks.

**Architecture:**
1. `src/video/fullbody_animator.py` provides high-resolution whole-body image dimension padding (mod-16), white studio backdrop compositing, and subtle breathing motion loop synthesis.
2. `src/pipeline.py` and `src/avatar/profile.py` are extended to support whole-body pose assets and automated base video creation for LatentSync.
3. Three focused Colab notebooks (`01_avatar_knowledge_base.ipynb`, `02_talking_head_studio.ipynb`, `03_fullbody_latentsync_studio.ipynb`) are created under `notebooks/` and copied to root for seamless Google Colab opening.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV / Pillow, NumPy, F5-TTS, LatentSync, BiRefNet, GFPGAN v1.4, pytest.

## Global Constraints
- NVIDIA Tesla T4 (16GB VRAM, Turing sm_75) compatibility with FP16 precision.
- Mod-16 spatial dimensions required for UNet latent diffusion (e.g. 864×960).
- Pure white studio background hex `#FFFFFF`.
- Clean error handling with fallback when heavy GPU libraries are absent.
- Full test coverage with zero regressions across existing unit tests.

---

### Task 1: Extend Avatar Profile with Whole-Body Pose Data Models

**Files:**
- Modify: `src/avatar/profile.py`
- Modify: `src/avatar/manager.py`
- Test: `tests/test_avatar_system.py`

**Interfaces:**
- `AvatarProfile.optimal_body_pose_path: Optional[str] = None`
- `AvatarProfile.optimal_body_pose_score: float = -1.0`
- `AvatarManager.register_media(avatar_id, file_path, media_type='body_pose')`

- [ ] **Step 1: Write test for body pose asset registration in AvatarProfile**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement body_pose support in profile.py and manager.py**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 2: Build `FullBodyAnimator` Module for White Backdrop & Motion Synthesis

**Files:**
- Create: `src/video/fullbody_animator.py`
- Modify: `src/video/__init__.py`
- Create: `tests/test_fullbody_animator.py`

**Interfaces:**
- `FullBodyAnimator.pad_to_mod16(image: np.ndarray, bg_color=(255,255,255)) -> np.ndarray`
- `FullBodyAnimator.composite_white_studio(image_path: str, output_path: str) -> str`
- `FullBodyAnimator.synthesize_base_video(image_path: str, duration_s: float, output_video_path: str, fps: int = 25, motion_mode: str = 'subtle_breathing') -> str`

- [ ] **Step 1: Write tests for FullBodyAnimator (mod-16 padding, duration matching, breathing motion calculation)**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement `src/video/fullbody_animator.py`**
- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

---

### Task 3: Integrate `FullBodyAnimator` with `LatentSyncEngine` & `VideoProductionPipeline`

**Files:**
- Modify: `src/video/latentsync_engine.py`
- Modify: `src/pipeline.py`
- Modify: `tests/test_avatar_system.py`

**Interfaces:**
- `pipeline.produce_video(..., engine='latentsync', body_image_path=..., motion_mode='subtle_breathing', studio_backdrop='#FFFFFF')`
- When `engine='latentsync'` and no `base_video_path` is passed, automatically resolves `body_image_path` (or profile's `optimal_body_pose_path`) and synthesizes base video using `FullBodyAnimator`.

- [ ] **Step 1: Write integration test for pipeline LatentSync fallback resolution**
- [ ] **Step 2: Implement pipeline and LatentSync engine integration**
- [ ] **Step 3: Run all unit tests to verify zero regressions**
- [ ] **Step 4: Commit**

---

### Task 4: Modular Colab Notebook Suite Generation

**Files:**
- Create: `scripts/generate_modular_notebooks.py`
- Generate: `notebooks/01_avatar_knowledge_base.ipynb`
- Generate: `notebooks/02_talking_head_studio.ipynb`
- Generate: `notebooks/03_fullbody_latentsync_studio.ipynb`

- [ ] **Step 1: Implement generator script for the 3 specialized notebooks**
- [ ] **Step 2: Execute script and validate notebook JSON schemas**
- [ ] **Step 3: Copy generated notebooks to root and DubeyDivyansh mirror**
- [ ] **Step 4: Commit**

---

### Task 5: End-to-End Verification, Documentation Update & GitHub Push

**Files:**
- Modify: `README.md`
- Test: Run full `pytest` suite across all tests
- Git: Push to `DubeyDivyansh/video_production_engine` and `DubeyDivyansh/DubeyDivyansh`

- [ ] **Step 1: Run comprehensive pytest suite**
- [ ] **Step 2: Update README with Full-Body LatentSync & 3-Notebook Architecture**
- [ ] **Step 3: Commit and push both GitHub repositories**
