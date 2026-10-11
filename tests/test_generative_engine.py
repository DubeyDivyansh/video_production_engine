"""
Unit Tests for GenerativeVideoEngine
Validates two-phase DiT video generation (LTX-Video and Wan2.1 on T4),
memory offloading, dimension/frame validation, and fallback handling.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestGenerativeVideoEngine(unittest.TestCase):
    def setUp(self):
        from src.video.generative_engine import GenerativeVideoEngine
        self.engine = GenerativeVideoEngine()

    def test_init_and_supported_models(self):
        self.assertEqual(self.engine.supported_models, ["ltx-video", "wan-1.3b"])
        self.assertIn("ltx-video", self.engine.SUPPORTED_MODELS)
        self.assertIn("wan-1.3b", self.engine.SUPPORTED_MODELS)
        self.assertTrue(hasattr(self.engine, "checkpoints_dir"))

        # Test custom checkpoints directory
        from src.video.generative_engine import GenerativeVideoEngine
        custom_engine = GenerativeVideoEngine(checkpoints_dir="/custom/path")
        self.assertEqual(custom_engine.checkpoints_dir, "/custom/path")

    def test_validate_frame_count(self):
        # LTX-Video requires (num_frames - 1) % 8 == 0 (8n + 1)
        self.assertTrue(self.engine.validate_frame_count(41, "ltx-video"))
        self.assertTrue(self.engine.validate_frame_count(49, "ltx-video"))
        self.assertTrue(self.engine.validate_frame_count(97, "ltx-video"))
        self.assertTrue(self.engine.validate_frame_count(1, "ltx-video"))
        self.assertFalse(self.engine.validate_frame_count(50, "ltx-video"))
        self.assertFalse(self.engine.validate_frame_count(42, "ltx-video"))
        self.assertFalse(self.engine.validate_frame_count(0, "ltx-video"))

        # Wan2.1 requires (num_frames - 1) % 4 == 0 (4n + 1)
        self.assertTrue(self.engine.validate_frame_count(41, "wan-1.3b"))
        self.assertTrue(self.engine.validate_frame_count(45, "wan-1.3b"))
        self.assertTrue(self.engine.validate_frame_count(1, "wan-1.3b"))
        self.assertFalse(self.engine.validate_frame_count(42, "wan-1.3b"))
        self.assertFalse(self.engine.validate_frame_count(50, "wan-1.3b"))

        # Other models default to True if frame count > 0
        self.assertTrue(self.engine.validate_frame_count(50, "other-model"))

    def test_validate_dimensions(self):
        # Must be divisible by 32
        self.assertTrue(self.engine.validate_dimensions(512, 320))
        self.assertTrue(self.engine.validate_dimensions(768, 512))
        self.assertTrue(self.engine.validate_dimensions(64, 64))
        self.assertFalse(self.engine.validate_dimensions(500, 300))
        self.assertFalse(self.engine.validate_dimensions(512, 300))
        self.assertFalse(self.engine.validate_dimensions(500, 320))
        self.assertFalse(self.engine.validate_dimensions(0, 320))

    def test_generate_scene_dimension_validation_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            with self.assertRaises(ValueError) as ctx:
                self.engine.generate_scene(
                    prompt="A serene waterfall",
                    output_path=out_path,
                    width=500,
                    height=300,
                    num_frames=41,
                )
            self.assertIn("divisible by 32", str(ctx.exception).lower())

    def test_generate_scene_frame_count_validation_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            with self.assertRaises(ValueError) as ctx:
                self.engine.generate_scene(
                    prompt="A serene waterfall",
                    output_path=out_path,
                    width=512,
                    height=320,
                    num_frames=50,  # Invalid for ltx-video
                    model_name="ltx-video",
                )
            self.assertIn("frame", str(ctx.exception).lower())

    def test_generate_scene_unsupported_model_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            with self.assertRaises(ValueError) as ctx:
                self.engine.generate_scene(
                    prompt="A serene waterfall",
                    output_path=out_path,
                    model_name="unsupported-model-v9",
                )
            self.assertIn("unsupported", str(ctx.exception).lower())

    def test_generate_scene_mocked_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "nested", "scene.mp4")
            dummy_frames = ["frame_1", "frame_2", "frame_3"]

            with patch.object(self.engine, "_execute_two_phase_ltx", return_value=dummy_frames) as mock_ltx:
                with patch("src.video.generative_engine.export_to_video") as mock_export:
                    result = self.engine.generate_scene(
                        prompt="Cinematic shot of cyber city",
                        output_path=out_path,
                        model_name="ltx-video",
                        width=512,
                        height=320,
                        num_frames=41,
                        fps=24,
                    )
                    self.assertEqual(result, out_path)
                    self.assertTrue(os.path.exists(os.path.dirname(out_path)))
                    mock_ltx.assert_called_once()
                    mock_export.assert_called_once_with(dummy_frames, out_path, fps=24)

    def test_generate_image_to_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.png")
            with open(img_path, "wb") as f:
                f.write(b"fake image data")
            out_path = os.path.join(tmpdir, "animated.mp4")

            with patch.object(self.engine, "generate_scene", return_value=out_path) as mock_gen_scene:
                result = self.engine.generate_image_to_video(
                    image_path=img_path,
                    prompt="Gentle wind blowing through trees",
                    output_path=out_path,
                    num_frames=41,
                    fps=24,
                )
                self.assertEqual(result, out_path)
                mock_gen_scene.assert_called_once()

    def test_two_phase_ltx_cpu_fallback(self):
        # Verifies offline CPU fallback returns num_frames frames without throwing
        frames = self.engine._execute_two_phase_ltx(
            prompt="Sunny day in the park",
            negative_prompt="blurry",
            width=512,
            height=320,
            num_frames=41,
        )
        self.assertEqual(len(frames), 41)

    def test_two_phase_wan_cpu_fallback(self):
        frames = self.engine._execute_two_phase_wan(
            prompt="Sunny day in the park",
            negative_prompt="blurry",
            width=512,
            height=320,
            num_frames=41,
        )
        self.assertEqual(len(frames), 41)

    def test_generate_scene_wan_mocked_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "wan_scene.mp4")
            dummy_frames = ["frame_1", "frame_2"]

            with patch.object(self.engine, "_execute_two_phase_wan", return_value=dummy_frames) as mock_wan:
                with patch("src.video.generative_engine.export_to_video") as mock_export:
                    result = self.engine.generate_scene(
                        prompt="Cosmic nebula floating in space",
                        output_path=out_path,
                        model_name="wan-1.3b",
                        width=512,
                        height=320,
                        num_frames=41,
                        fps=24,
                    )
                    self.assertEqual(result, out_path)
                    mock_wan.assert_called_once()
                    mock_export.assert_called_once_with(dummy_frames, out_path, fps=24)

    def test_export_to_video_fallback(self):
        from src.video.generative_engine import export_to_video
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_export.mp4")
            frames = [Image.new("RGB", (64, 64), color="blue") for _ in range(5)]
            exported = export_to_video(frames, out_path, fps=24)
            self.assertEqual(exported, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
    def test_package_exports_generative_engine(self):
        from src.video import GenerativeVideoEngine
        self.assertIsNotNone(GenerativeVideoEngine)


if __name__ == "__main__":
    unittest.main()

