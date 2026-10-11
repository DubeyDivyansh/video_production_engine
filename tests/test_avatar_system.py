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


if __name__ == "__main__":
    unittest.main()
