# Persistent Avatar Knowledge Base & Studio Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent Google Drive-backed Avatar Knowledge Base (`AvatarManager` + `AvatarProfile`) for continuous multi-modal learning from uploaded assets, coupled with a full Studio Compositor (`BackgroundReplacer`, `FaceEnhancer`, `AudioEnhancer`), integrated into the modular `video_production_engine` pipeline and runnable on Colab T4.

**Architecture:** File-backed persistent avatar profiles storing multi-modal metadata and embeddings on Google Drive, routing into an automated quality scoring pipeline, followed by video synthesis and pluggable post-production enhancers (RMBG matting, GFPGAN upscaling, spectral audio denoising).

**Tech Stack:** Python 3.10+, PyTorch (FP16 on Turing sm_75), OpenCV, Librosa, Noisereduce, BiRefNet / RMBG, GFPGAN, F5-TTS, LatentSync, Google Drive.

## Global Constraints
- Target GPU: Tesla T4 (16 GB VRAM, Turing compute capability 7.5).
- Peak VRAM footprint across any pipeline stage must remain $\le 12.5\text{ GB}$.
- Cloud persistence root: `/content/drive/MyDrive/video_gen_platform/`.
- Local repository: `c:\Users\Acer\Antigravity Projects\projects\video_production_engine`.
- Remote repository: `https://github.com/DubeyDivyansh/video_production_engine` (private).

---

### Task 1: Avatar Data Models & Serialization (`src/avatar/profile.py`)

**Files:**
- Create: `src/avatar/profile.py`
- Create: `src/avatar/__init__.py`
- Test: `tests/test_avatar_profile.py`

**Interfaces:**
- Produces: `MediaAsset` dataclass, `AvatarProfile` dataclass with `to_json()`, `from_json()`, `save()`, `load()`.

- [ ] **Step 1: Write the failing test**
```python
import os
import json
import pytest
from src.avatar.profile import MediaAsset, AvatarProfile

def test_avatar_profile_serialization(tmp_path):
    profile = AvatarProfile(
        avatar_id="test_user",
        created_at="2026-10-11T12:00:00",
        updated_at="2026-10-11T12:00:00",
        optimal_audio_path="/path/to/audio.wav",
        optimal_audio_score=24.5,
        optimal_video_path="/path/to/video.mp4",
        optimal_video_score=0.85,
        optimal_image_path="/path/to/image.jpg",
        optimal_image_score=150.2,
        embedding_paths={"f5": "/path/to/latent.pt"},
        raw_assets=[
            MediaAsset(file_path="/raw/a.wav", media_type="audio", quality_score=24.5, timestamp="2026-10-11T12:00:00")
        ]
    )
    save_path = tmp_path / "avatar_profile.json"
    profile.save(str(save_path))
    assert save_path.exists()
    
    loaded = AvatarProfile.load(str(save_path))
    assert loaded.avatar_id == "test_user"
    assert loaded.optimal_audio_score == 24.5
    assert len(loaded.raw_assets) == 1
    assert loaded.raw_assets[0].file_path == "/raw/a.wav"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_avatar_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.avatar'`

- [ ] **Step 3: Implement `src/avatar/profile.py`**
Implement data classes and JSON serialization methods.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_avatar_profile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/avatar/ tests/test_avatar_profile.py
git commit -m "feat(avatar): implement AvatarProfile and MediaAsset serialization"
```

---

### Task 2: Avatar Management & Automatic Quality Scoring (`src/avatar/manager.py`)

**Files:**
- Create: `src/avatar/manager.py`
- Test: `tests/test_avatar_manager.py`

**Interfaces:**
- Consumes: `AvatarProfile`, `MediaAsset` from `src.avatar.profile`.
- Produces: `AvatarManager` with `register_media()`, `get_avatar()`, `list_avatars()`, and scoring methods `score_audio()`, `score_image()`, `score_video()`.

- [ ] **Step 1: Write the failing test**
```python
import os
import pytest
import numpy as np
import soundfile as sf
import cv2
from src.avatar.manager import AvatarManager

def test_avatar_manager_ingest_and_score(tmp_path):
    manager = AvatarManager(base_dir=str(tmp_path))
    
    # 1. Create dummy test audio
    audio_path = tmp_path / "test_audio.wav"
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    waveform = np.sin(2 * np.pi * 440 * t) * 0.5
    sf.write(str(audio_path), waveform, sr)
    
    # 2. Create dummy test image
    img_path = tmp_path / "test_img.jpg"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(img_path), img)
    
    # 3. Register media
    res_audio = manager.register_media("user_demo", str(audio_path), media_type="audio")
    res_img = manager.register_media("user_demo", str(img_path), media_type="image")
    
    profile = manager.get_avatar("user_demo")
    assert profile.avatar_id == "user_demo"
    assert profile.optimal_audio_path is not None
    assert profile.optimal_image_path is not None
    assert len(profile.raw_assets) == 2
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_avatar_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.avatar.manager'`

- [ ] **Step 3: Implement `src/avatar/manager.py`**
Implement audio SNR/clipping scoring, Laplacian blur variance calculation, and directory management.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_avatar_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/avatar/manager.py tests/test_avatar_manager.py
git commit -m "feat(avatar): implement AvatarManager with automated quality scoring"
```

---

### Task 3: Studio Background Removal & Compositing (`src/enhancers/background.py`)

**Files:**
- Create: `src/enhancers/background.py`
- Create: `src/enhancers/__init__.py`
- Test: `tests/test_background_replacer.py`

**Interfaces:**
- Produces: `BackgroundReplacer` with `remove_background()`, `replace_background()`, `apply_color_backdrop()`.

- [ ] **Step 1: Write the failing test**
```python
import os
import cv2
import numpy as np
import pytest
from src.enhancers.background.py import BackgroundReplacer

def test_chroma_and_composite(tmp_path):
    replacer = BackgroundReplacer(use_gpu=False)
    # Test chroma mask generation on pure green frame
    green_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    green_frame[:, :] = [0, 255, 0] # BGR pure green
    # Add a white box in center
    green_frame[25:75, 25:75] = [255, 255, 255]
    
    alpha_mask = replacer._compute_chroma_mask(green_frame)
    assert alpha_mask[50, 50] == 255  # Center white box kept
    assert alpha_mask[10, 10] == 0    # Background removed
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_background_replacer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/enhancers/background.py`**
Implement RMBG / BiRefNet matting inference wrapper, chroma-key fallback, alpha video writer (VP9), and backdrop image/video blending.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_background_replacer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/enhancers/background.py tests/test_background_replacer.py
git commit -m "feat(enhancers): implement BackgroundReplacer for studio matting"
```

---

### Task 4: Face Restoration & Audio Mastering Enhancers (`src/enhancers/face_enhancer.py`, `src/enhancers/audio_enhancer.py`)

**Files:**
- Create: `src/enhancers/face_enhancer.py`
- Create: `src/enhancers/audio_enhancer.py`
- Test: `tests/test_enhancers.py`

**Interfaces:**
- Produces:
  - `FaceEnhancer` with `enhance_image()`, `enhance_video()`.
  - `AudioEnhancer` with `denoise_audio()`, `normalize_loudness()`.

- [ ] **Step 1: Write the failing test**
```python
import numpy as np
import soundfile as sf
import pytest
from src.enhancers.audio_enhancer import AudioEnhancer

def test_audio_enhancer_denoise_and_lufs(tmp_path):
    enhancer = AudioEnhancer()
    raw_wav = tmp_path / "raw.wav"
    sr = 24000
    noise = np.random.normal(0, 0.05, sr * 2)
    signal = np.sin(2 * np.pi * 300 * np.linspace(0, 2, sr * 2)) * 0.4
    sf.write(str(raw_wav), signal + noise, sr)
    
    out_wav = tmp_path / "denoised.wav"
    enhancer.denoise_audio(str(raw_wav), str(out_wav))
    assert out_wav.exists()
    data, _ = sf.read(str(out_wav))
    assert len(data) > 0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_enhancers.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `FaceEnhancer` and `AudioEnhancer`**
Implement GFPGAN integration with soft lip-region blending, spectral gating noise reduction, and LUFS normalization.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_enhancers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/enhancers/face_enhancer.py src/enhancers/audio_enhancer.py tests/test_enhancers.py
git commit -m "feat(enhancers): implement FaceEnhancer and AudioEnhancer"
```

---

### Task 5: Pipeline Router Integration & Colab Notebook Update

**Files:**
- Modify: `src/pipeline.py`
- Modify: `quick_demo.ipynb`
- Modify: `requirements.txt`
- Modify: `README.md`
- Test: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Updates `VideoProductionPipeline.produce_video()` to support `avatar_id`, `replace_background`, `enhance_face`, `denoise_audio`.

- [ ] **Step 1: Write the integration test**
- [ ] **Step 2: Update `src/pipeline.py`**
- [ ] **Step 3: Update `requirements.txt` & `quick_demo.ipynb`**
- [ ] **Step 4: Verify test suite passes**
- [ ] **Step 5: Commit and push to origin/main**
