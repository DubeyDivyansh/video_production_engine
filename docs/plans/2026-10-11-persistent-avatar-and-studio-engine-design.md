# Design Specification: Persistent Avatar Knowledge Base & Full Studio Production Engine

- **Date**: 2026-10-11
- **Status**: Approved
- **Target Platform**: Google Colab (Tesla T4 GPU, 16GB VRAM, Google Drive Persistence) & Private Repository (`DubeyDivyansh/video_production_engine`)

---

## 1. Executive Summary
This design specification defines the architecture for:
1. **Persistent Avatar Knowledge Base & Asset Hub (`AvatarManager` & `AvatarProfile`)**: A continuous improvement system residing in Google Drive that ingests user media (images, videos, audio), scores asset quality using automated best-fit heuristics, extracts multi-modal representations (F5 voice embeddings, 3DMM landmarks, face recognition vectors), and enables zero-restart inference.
2. **Full Studio Compositor & Enhancers**: Post-production modules for high-accuracy background removal/replacement (RMBG / BiRefNet / Chroma Keying), face restoration & super-resolution (GFPGAN / CodeFormer), and broadcast audio mastering (spectral gating noise reduction & -16 LUFS normalization).

---

## 2. Directory Hierarchy (Google Drive & Workspace)

The platform mounts Google Drive at `/content/drive/MyDrive/video_gen_platform/` with the following persistent tree:

```text
/content/drive/MyDrive/video_gen_platform/
├── avatars/
│   └── {avatar_id}/                        # E.g., 'divyansh'
│       ├── avatar_profile.json             # Canonical metadata, quality scores, asset pointers
│       ├── raw/                            # Immutable historical archive of all user uploads
│       │   ├── images/                     # [YYYYMMDD_HHMMSS]_[filename].jpg
│       │   ├── audios/                     # [YYYYMMDD_HHMMSS]_[filename].wav
│       │   └── videos/                     # [YYYYMMDD_HHMMSS]_[filename].mp4
│       ├── embeddings/                     # Precomputed neural tensors (cached for instant runs)
│       │   ├── f5_speaker_latent.pt        # F5-TTS zero-shot voice embedding
│       │   ├── face_identity_emb.npy       # Facial identity embedding
│       │   └── landmark_coefficients.pkl   # 3DMM facial motion geometry
│       └── processed/                      # Pre-trimmed, optimal reference slices
│           ├── optimal_voice_prompt.wav    # Best 7-8s audio slice (highest SNR)
│           ├── optimal_base_video.mp4      # Best gesture/motion video loop
│           └── reference_face_crop.png     # Highest-sharpness aligned portrait
├── checkpoints/                            # Heavy model weights (cached across Colab sessions)
│   ├── latentsync/
│   ├── sadtalker/
│   ├── liveportrait/
│   ├── gfpgan/
│   └── birefnet/
└── outputs/                                # Rendered production outputs
```

---

## 3. Subsystem Architecture

### 3.1 Persistent Avatar Knowledge Base (`src/avatar/`)

#### Module: `src/avatar/profile.py`
Data classes modeling the avatar's state:
* `MediaAsset`: Metadata record for an uploaded file (path, timestamp, type, quality_score, metrics_dict).
* `AvatarProfile`:
  * `avatar_id: str`
  * `created_at: str`, `updated_at: str`
  * `optimal_audio_path: str`, `optimal_audio_score: float`
  * `optimal_video_path: str`, `optimal_video_score: float`
  * `optimal_image_path: str`, `optimal_image_score: float`
  * `embedding_paths: Dict[str, str]`
  * `raw_assets: List[MediaAsset]`
  * `custom_configs: Dict[str, Any]` (preferred engine, default backdrop, etc.)

#### Module: `src/avatar/manager.py`
The orchestration engine for avatar ingestion, calibration, and retrieval:
* **`register_media(media_path: str, media_type: str = "auto") -> Dict[str, Any]`**:
  1. Detects file type (`audio`, `image`, `video`).
  2. Archives original file to `raw/{type}/` with timestamp.
  3. Computes automated quality heuristics:
     * **Audio Metrics**:
       * Signal-to-Noise Ratio (SNR in dB).
       * Clipping ratio (samples $> 0.99$).
       * Silence ratio (fraction of audio with RMS $< 20\text{ dB}$ below peak).
       * If audio is cleaner than current `optimal_audio_score`, trims optimal 7.5s speech segment, extracts F5 voice prompt, and caches to `processed/optimal_voice_prompt.wav`.
     * **Image Metrics**:
       * Laplacian variance (sharpness score).
       * Face detection bounding box and roll/pitch/yaw angles.
       * If sharper and more frontal than current `optimal_image_score`, updates `processed/reference_face_crop.png`.
     * **Video Metrics**:
       * Resolution and aspect ratio.
       * Optical flow variance (identifies smooth, jitter-free body posture periods).
       * Face track continuity.
       * Trims best 10–15s loop to `processed/optimal_base_video.mp4`.
  4. Saves updated state to `avatar_profile.json` atomically.
* **`get_avatar(avatar_id: str) -> AvatarProfile`**:
  Loads profile and validates that all referenced assets exist.
* **`list_avatars() -> List[str]`**:
  Enumerates all avatars stored in Google Drive.

---

### 3.2 Studio Compositor & Enhancers (`src/enhancers/`)

#### Module: `src/enhancers/background.py` (`BackgroundReplacer`)
* **Engine**: BiRefNet / RMBG-1.4 (PyTorch FP16 on T4 GPU) with fallback to OpenCV Chroma Keying.
* **Methods**:
  * `remove_background(video_path: str, output_path: str, alpha: bool = True) -> str`:
    Processes video frames in batches (batch size 8 on T4). Outputs RGBA video encoded with VP9 codec (`.webm`) or ProRes 4444 (`.mov`).
  * `replace_background(video_path: str, backdrop_path: str, output_path: str, feather_radius: int = 2) -> str`:
    Extracts alpha matte for foreground avatar, resizes backdrop image or loops backdrop video, applies Gaussian edge feathering (radius 2px) to prevent halos, and composites foreground onto backdrop.
  * `apply_color_backdrop(video_path: str, hex_color: str, output_path: str) -> str`:
    Fills background with solid studio color (e.g. green screen `#00FF00`, slate grey `#1E1E1E`, white `#FFFFFF`).

#### Module: `src/enhancers/face_enhancer.py` (`FaceEnhancer`)
* **Engine**: GFPGAN v1.4 / CodeFormer.
* **Methods**:
  * `enhance_video(input_video_path: str, output_video_path: str, blend_weight: float = 0.8) -> str`:
    Runs frame-by-frame face alignment, upsamples facial textures by 2x, restores teeth/eye fidelity, and uses soft spatial blending around lip boundaries to prevent boundary lines.

#### Module: `src/enhancers/audio_enhancer.py` (`AudioEnhancer`)
* **Methods**:
  * `denoise_audio(input_wav: str, output_wav: str) -> str`:
    Applies non-stationary spectral gating via `noisereduce` to eliminate background mic hum and ambient room noise.
  * `normalize_loudness(input_wav: str, output_wav: str, target_lufs: float = -16.0) -> str`:
    Applies EBU R128 loudness normalization to standard -16 LUFS with true peak limiting at -1.0 dBFS.

---

### 3.3 Unified Pipeline Integration (`src/pipeline.py`)

The `VideoProductionPipeline` router is updated to accept an `avatar_id`:
```python
class VideoProductionPipeline:
    def __init__(self, use_drive: bool = True, avatar_id: Optional[str] = None):
        self.avatar_manager = AvatarManager(use_drive=use_drive)
        self.avatar = self.avatar_manager.get_avatar(avatar_id) if avatar_id else None
        self.face_enhancer = FaceEnhancer()
        self.bg_replacer = BackgroundReplacer()
        self.audio_enhancer = AudioEnhancer()

    def produce_video(
        self,
        script_text: str,
        avatar_id: Optional[str] = None,
        engine: str = "latentsync",      # 'latentsync' | 'sadtalker' | 'liveportrait' | 'musetalk'
        replace_background: Optional[str] = None,  # image/video path or hex color
        alpha_video: bool = False,                 # transparent .webm
        enhance_face: bool = True,
        denoise_audio: bool = True,
        output_name: str = "production_output.mp4"
    ) -> str:
        # 1. Resolve avatar assets from persistent profile
        # 2. Synthesize audio with F5-TTS
        # 3. Post-process audio (denoise + -16 LUFS)
        # 4. Flush VRAM fence
        # 5. Route to requested video engine with Turing T4 FP16 configs
        # 6. Apply Face Restoration (GFPGAN)
        # 7. Apply Studio Background Matting / Compositing
        # 8. Mux mastered audio and video
        # 9. Return persistent output path
```

---

## 4. Colab T4 Memory & Performance Budget

| Stage | Model / Library | Precision | Peak VRAM | Time (12s video) |
|---|---|---|---|---|
| **Voice Cloning** | F5-TTS (Flow Matching DiT) | FP16 | ~1.1 GB | ~4.5 s |
| **Audio Mastering** | noisereduce + pyloudnorm | CPU | 0 MB VRAM | ~0.8 s |
| **VRAM Fence Flush** | `torch.cuda.empty_cache()` | - | ~0 MB | < 0.1 s |
| **Lip-Sync Video** | LatentSync (DeepCache sm_75) | FP16 | ~12.2 GB | ~4.5 min |
| **Lip-Sync Video (Alt)** | SadTalker (3DMM + GFPGAN) | FP16 | ~4.2 GB | ~1.5 min |
| **Face Restoration** | GFPGAN v1.4 | FP16 | ~2.4 GB | ~18 s |
| **Background Matting**| BiRefNet / RMBG-1.4 | FP16 | ~1.1 GB | ~22 s |

**Total Peak VRAM at any stage:** $\le 12.2\text{ GB}$ (Comfortably fits inside Colab T4 16GB VRAM limit).

---

## 5. Verification & Testing Criteria

1. **Avatar Registration Test**:
   - Register `custom_face.jpeg`, `screenshot_face.jpg`, `custom_audio.wav`, and `base_video.mp4`.
   - Verify `avatar_profile.json` is generated with valid scores and timestamps.
2. **Avatar Loading Test**:
   - Load `avatar_id="divyansh"` from cold start; verify all pointers resolve to valid files.
3. **Background Matting Test**:
   - Run `BackgroundReplacer` on a 5-second test video with solid color (`#00FF00`) and custom backdrop image; verify output MP4 visually and check alpha channel.
4. **Face Restoration Test**:
   - Run `FaceEnhancer` on generated clip; verify resolution increase and natural facial contours.
5. **Colab 1-Click Execution**:
   - Run the updated `DubeyDivyansh/quick_demo.ipynb` notebook cell end-to-end without OOM or manual configuration.
