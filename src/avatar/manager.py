"""
Avatar Manager.
Handles multi-modal asset ingestion, quality scoring, optimal reference selection,
and persistent Google Drive synchronization.
"""

import os
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import cv2
except ImportError:
    cv2 = None

from src.avatar.profile import MediaAsset, AvatarProfile


class AvatarManager:
    """Manages persistent avatars and their multimodal assets."""

    def __init__(self, base_dir: Optional[str] = None, use_drive: bool = True):
        if base_dir:
            self.base_dir = base_dir
        elif use_drive and os.path.exists("/content/drive/MyDrive"):
            self.base_dir = "/content/drive/MyDrive/video_gen_platform/avatars"
        else:
            self.base_dir = os.path.join(os.getcwd(), "data", "avatars")

        os.makedirs(self.base_dir, exist_ok=True)

    def get_avatar_dir(self, avatar_id: str) -> str:
        path = os.path.join(self.base_dir, avatar_id)
        os.makedirs(os.path.join(path, "raw", "images"), exist_ok=True)
        os.makedirs(os.path.join(path, "raw", "audios"), exist_ok=True)
        os.makedirs(os.path.join(path, "raw", "videos"), exist_ok=True)
        os.makedirs(os.path.join(path, "processed"), exist_ok=True)
        os.makedirs(os.path.join(path, "embeddings"), exist_ok=True)
        return path

    def get_profile_path(self, avatar_id: str) -> str:
        return os.path.join(self.get_avatar_dir(avatar_id), "avatar_profile.json")

    def get_avatar(self, avatar_id: str) -> AvatarProfile:
        profile_path = self.get_profile_path(avatar_id)
        if os.path.exists(profile_path):
            return AvatarProfile.load(profile_path)
        # Create brand new avatar profile
        profile = AvatarProfile(avatar_id=avatar_id)
        profile.save(profile_path)
        return profile

    def list_avatars(self) -> List[str]:
        if not os.path.exists(self.base_dir):
            return []
        return [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]

    # --- Quality Scoring Heuristics ---

    def score_audio(self, audio_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Scores audio quality based on SNR (Signal-to-Noise Ratio), clipping, and vocal energy.
        Score range: typically 0.0 to 100.0 (higher is cleaner).
        """
        metrics = {"path": audio_path, "snr_db": 0.0, "clipping_ratio": 0.0, "duration_s": 0.0}
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            duration = len(data) / float(sr)
            metrics["duration_s"] = duration

            # Clipping detection
            clip_count = np.sum(np.abs(data) >= 0.99)
            clipping_ratio = clip_count / float(len(data))
            metrics["clipping_ratio"] = float(clipping_ratio)

            # Signal energy vs noise floor estimation
            energy = data ** 2
            mean_energy = np.mean(energy)
            # Estimate noise from lowest 10% energy frames
            frame_len = int(sr * 0.05)
            if len(data) > frame_len:
                num_frames = len(data) // frame_len
                frames = data[:num_frames * frame_len].reshape(num_frames, frame_len)
                frame_energies = np.mean(frames ** 2, axis=1)
                sorted_energies = np.sort(frame_energies)
                noise_floor = np.mean(sorted_energies[:max(1, int(num_frames * 0.1))]) + 1e-9
                snr_db = 10 * np.log10(mean_energy / noise_floor + 1e-9)
            else:
                snr_db = 15.0

            metrics["snr_db"] = float(snr_db)

            # Composite score: penalize clipping and reward good SNR (capped at 40 dB)
            clipping_penalty = clipping_ratio * 100.0
            score = max(0.0, min(100.0, (snr_db * 2.5) - clipping_penalty))
            return float(score), metrics

        except Exception as e:
            metrics["error"] = str(e)
            return 10.0, metrics

    def score_image(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Scores image quality based on Laplacian variance (sharpness) and resolution.
        """
        metrics = {"path": image_path, "sharpness": 0.0, "width": 0, "height": 0}
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0, metrics

            h, w = img.shape[:2]
            metrics["width"] = w
            metrics["height"] = h

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics["sharpness"] = float(sharpness)

            # Resolution factor
            res_factor = min(1.0, (w * h) / (1024 * 1024))
            score = float(sharpness * res_factor)
            return score, metrics

        except Exception as e:
            metrics["error"] = str(e)
            return 0.0, metrics

    def score_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Scores video quality based on resolution, frame rate, and motion stability.
        """
        metrics = {"path": video_path, "width": 0, "height": 0, "fps": 0, "frames": 0, "stability": 0.0}
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, metrics

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            metrics["width"] = w
            metrics["height"] = h
            metrics["fps"] = fps
            metrics["frames"] = total_frames

            # Sample 20 frames to measure optical flow stability
            step = max(1, total_frames // 20)
            prev_gray = None
            flow_magnitudes = []

            for i in range(0, min(total_frames, 20 * step), step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_magnitudes.append(np.mean(mag))
                prev_gray = gray
            cap.release()

            avg_motion = float(np.mean(flow_magnitudes)) if flow_magnitudes else 1.0
            # Stability rewards gentle movement (0.5 to 3.0), penalizes jitter (> 8.0)
            stability = max(0.0, 10.0 - abs(avg_motion - 1.5))
            metrics["stability"] = stability

            res_score = min(100.0, (w * h) / (1280 * 720) * 50.0)
            score = float(res_score + stability * 5.0)
            return score, metrics

        except Exception as e:
            metrics["error"] = str(e)
            return 0.0, metrics

    # --- Media Registration & Continuous Calibration ---

    def register_media(
        self,
        avatar_id: str,
        file_path: str,
        media_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Ingests user media, computes quality score, archives file to Drive,
        and auto-updates optimal reference slices if the new asset is higher quality.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if media_type == "auto":
            if ext in [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]:
                media_type = "audio"
            elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                media_type = "image"
            elif ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                media_type = "video"
            else:
                raise ValueError(f"Unsupported media extension: {ext}")

        avatar_dir = self.get_avatar_dir(avatar_id)
        profile = self.get_avatar(avatar_id)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        orig_name = os.path.basename(file_path)
        dest_filename = f"{timestamp}_{orig_name}"

        # 1. Archive raw media
        raw_dest_dir = os.path.join(avatar_dir, "raw", f"{media_type}s")
        raw_dest_path = os.path.join(raw_dest_dir, dest_filename)
        shutil.copy2(file_path, raw_dest_path)

        # 2. Score asset quality & update optimal slices
        score = 0.0
        metrics = {}

        if media_type == "audio":
            score, metrics = self.score_audio(file_path)
            # If cleaner, set as optimal audio and trim a crisp 7.5s reference slice
            if score > profile.optimal_audio_score or profile.optimal_audio_path is None:
                optimal_wav = os.path.join(avatar_dir, "processed", "optimal_voice_prompt.wav")
                self._slice_optimal_audio(file_path, optimal_wav, target_duration=7.5)
                profile.optimal_audio_path = optimal_wav
                profile.optimal_audio_score = score

        elif media_type == "image":
            score, metrics = self.score_image(file_path)
            if score > profile.optimal_image_score or profile.optimal_image_path is None:
                optimal_img = os.path.join(avatar_dir, "processed", "reference_face_crop.png")
                shutil.copy2(file_path, optimal_img)
                profile.optimal_image_path = optimal_img
                profile.optimal_image_score = score

        elif media_type == "video":
            score, metrics = self.score_video(file_path)
            if score > profile.optimal_video_score or profile.optimal_video_path is None:
                optimal_vid = os.path.join(avatar_dir, "processed", "optimal_base_video.mp4")
                shutil.copy2(file_path, optimal_vid)
                profile.optimal_video_path = optimal_vid
                profile.optimal_video_score = score

        # 3. Append to raw assets log and persist profile
        asset_record = MediaAsset(
            file_path=raw_dest_path,
            media_type=media_type,
            quality_score=score,
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
            original_filename=orig_name
        )
        profile.raw_assets.append(asset_record)
        profile.save(self.get_profile_path(avatar_id))

        return {
            "avatar_id": avatar_id,
            "media_type": media_type,
            "archived_path": raw_dest_path,
            "quality_score": score,
            "metrics": metrics,
            "updated_optimal": (
                profile.optimal_audio_path if media_type == "audio" else
                profile.optimal_image_path if media_type == "image" else
                profile.optimal_video_path
            )
        }

    def _slice_optimal_audio(self, in_wav: str, out_wav: str, target_duration: float = 7.5) -> None:
        """Trims the cleanest speech segment matching target_duration."""
        try:
            import soundfile as sf
            data, sr = sf.read(in_wav)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            target_samples = int(target_duration * sr)
            if len(data) <= target_samples:
                sf.write(out_wav, data, sr)
                return

            # Find segment with highest speech energy
            hop = int(sr * 0.5)
            best_idx = 0
            best_energy = -1.0

            for i in range(0, len(data) - target_samples, hop):
                chunk = data[i:i + target_samples]
                energy = np.sum(chunk ** 2)
                if energy > best_energy:
                    best_energy = energy
                    best_idx = i

            sf.write(out_wav, data[best_idx:best_idx + target_samples], sr)
        except Exception:
            shutil.copy2(in_wav, out_wav)
