"""
Unit Tests for Avatar Management, Scoring, and Studio Enhancers.
Supports local execution (with standard mocks when heavy ML libraries are absent)
and full integration testing on Google Colab / GPU runtimes.
"""

import os
import sys
import json
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.avatar.profile import MediaAsset, AvatarProfile

try:
    import numpy as np
    import soundfile as sf
    import cv2
    FULL_DEPS_AVAILABLE = True
except ImportError:
    FULL_DEPS_AVAILABLE = False


class TestAvatarProfile(unittest.TestCase):
    def test_avatar_profile_serialization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = AvatarProfile(
                avatar_id="divyansh_test",
                optimal_audio_path="/content/optimal.wav",
                optimal_audio_score=45.2,
                optimal_body_pose_path="/content/suit.jpg",
                optimal_body_pose_score=85.0,
                raw_assets=[
                    MediaAsset(
                        file_path="/content/a.wav",
                        media_type="audio",
                        quality_score=45.2,
                        timestamp="2026-10-11T12:00:00",
                    )
                ],
            )
            save_file = os.path.join(tmp_dir, "avatar_profile.json")
            profile.save(save_file)
            self.assertTrue(os.path.exists(save_file))

            loaded = AvatarProfile.load(save_file)
            self.assertEqual(loaded.avatar_id, "divyansh_test")
            self.assertEqual(loaded.optimal_audio_score, 45.2)
            self.assertEqual(loaded.optimal_body_pose_path, "/content/suit.jpg")
            self.assertEqual(loaded.optimal_body_pose_score, 85.0)
            self.assertEqual(len(loaded.raw_assets), 1)
            self.assertEqual(loaded.raw_assets[0].file_path, "/content/a.wav")

    def test_srt_generation(self):
        from src.enhancers.studio_compositor import StudioCompositor
        compositor = StudioCompositor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = os.path.join(tmp_dir, "test.srt")
            script = "Hello world! This is a test of our video production engine. Let us verify timing."
            compositor.generate_srt_from_script(script, total_duration_sec=12.0, srt_output_path=srt_path)
            self.assertTrue(os.path.exists(srt_path))
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("1\n00:00:00,000 -->", content)
            self.assertIn("Hello world", content)

    def test_drive_id_extraction(self):
        url = "https://drive.google.com/file/d/1yCoOYz-Q-pzAaDYEWJUV1RihuiYJGQX2/view?usp=sharing"
        import re
        match = re.search(r"[-_\w]{25,}", url)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(0), "1yCoOYz-Q-pzAaDYEWJUV1RihuiYJGQX2")


@unittest.skipUnless(FULL_DEPS_AVAILABLE, "Requires numpy, soundfile, and cv2")
class TestAvatarIntegration(unittest.TestCase):
    def setUp(self):
        from src.avatar.manager import AvatarManager
        from src.enhancers.background import BackgroundReplacer
        from src.enhancers.audio_enhancer import AudioEnhancer

        self.AvatarManager = AvatarManager
        self.BackgroundReplacer = BackgroundReplacer
        self.AudioEnhancer = AudioEnhancer

    def test_avatar_manager_ingestion_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = self.AvatarManager(base_dir=tmp_dir, use_drive=False)

            # 1. Create dummy audio
            audio_path = os.path.join(tmp_dir, "sample.wav")
            sr = 16000
            t = np.linspace(0, 2, sr * 2)
            sig = np.sin(2 * np.pi * 440 * t) * 0.4
            sf.write(audio_path, sig, sr)

            # 2. Create dummy image
            img_path = os.path.join(tmp_dir, "portrait.jpg")
            img = np.zeros((150, 150, 3), dtype=np.uint8)
            cv2.circle(img, (75, 75), 40, (255, 255, 255), -1)
            cv2.imwrite(img_path, img)

            # 3. Register media
            res_audio = manager.register_media("divyansh", audio_path, media_type="audio")
            res_img = manager.register_media("divyansh", img_path, media_type="image")

            self.assertGreater(res_audio["quality_score"], 0)
            self.assertGreater(res_img["quality_score"], 0)

            profile = manager.get_avatar("divyansh")
            self.assertIsNotNone(profile.optimal_audio_path)
            self.assertIsNotNone(profile.optimal_image_path)
            self.assertEqual(len(profile.raw_assets), 2)

    def test_background_chroma_mask(self):
        replacer = self.BackgroundReplacer(method="chroma", use_gpu=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]  # BGR green
        frame[30:70, 30:70] = [255, 255, 255]  # Center subject

        mask = replacer._compute_chroma_mask(frame)
        self.assertGreater(mask[50, 50], 200)
        self.assertLess(mask[5, 5], 50)


class TestQualityInspector(unittest.TestCase):
    def setUp(self):
        from src.verification.inspector import QualityInspector, VerificationReport
        self.QualityInspector = QualityInspector
        self.VerificationReport = VerificationReport

    def _create_synthetic_wav(self, file_path: str, duration_sec: float = 2.0, sr: int = 16000, clip: bool = False):
        import wave
        import struct
        import math
        n_samples = int(duration_sec * sr)
        with wave.open(file_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            frames = bytearray()
            for i in range(n_samples):
                # Simulate speech with an initial pause (0.3s)
                if i < int(0.3 * sr):
                    sample = 0
                else:
                    val = math.sin(2 * math.pi * 440 * (i / sr))
                    if clip and i > n_samples // 2:
                        sample = 32767  # Max clipping
                    else:
                        sample = int(val * 16000)
                frames.extend(struct.pack("<h", sample))
            w.writeframes(frames)

    def test_audio_inspection_and_metrics(self):
        inspector = self.QualityInspector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "test_audio.wav")
            self._create_synthetic_wav(wav_path, duration_sec=2.0)
            
            metrics, warnings, errors = inspector.inspect_audio_file(wav_path, script_text="Hello world speech test")
            self.assertEqual(len(errors), 0)
            self.assertAlmostEqual(metrics["duration"], 2.0, places=1)
            self.assertGreater(metrics["snr_db"], 10.0)
            self.assertLess(metrics["clipping_pct"], 0.1)

    def test_audio_clipping_detection(self):
        inspector = self.QualityInspector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            clipped_wav = os.path.join(tmp_dir, "clipped.wav")
            self._create_synthetic_wav(clipped_wav, duration_sec=1.0, clip=True)
            
            metrics, warnings, errors = inspector.inspect_audio_file(clipped_wav)
            self.assertGreater(metrics["clipping_pct"], 10.0)
            self.assertTrue(any("clipping" in w for w in warnings))

    def test_av_sync_detection(self):
        inspector = self.QualityInspector()
        audio_m = {"duration": 10.0}
        video_m = {"duration": 8.0}
        sync_m, warnings, errors = inspector.inspect_av_sync("dummy.mp4", audio_m, video_m)
        self.assertAlmostEqual(sync_m["av_duration_delta"], 2.0)
        self.assertTrue(any("mismatch" in w for w in warnings))

    def test_verification_report_save_and_remediations(self):
        inspector = self.QualityInspector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "clean.wav")
            self._create_synthetic_wav(wav_path, duration_sec=1.5)
            
            report = inspector.audit_deliverable(
                video_path=os.path.join(tmp_dir, "mock_video.mp4"),
                speech_wav_path=wav_path,
                script_text="Hello Divyansh here",
                save_report=True,
            )
            self.assertIsInstance(report, self.VerificationReport)
            report_file = os.path.join(tmp_dir, "mock_video_quality_audit.json")
            self.assertTrue(os.path.exists(report_file))


if __name__ == "__main__":
    unittest.main()

