"""
Unit tests for post-production enhancers (Motion Interpolator with RIFE 60fps & FFmpeg fallback).
Ensures zero-crash execution, fallback resilience, and precise command construction.
"""

import os
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestMotionInterpolator(unittest.TestCase):
    def test_interpolator_initialization_default(self):
        from src.enhancers.motion_interpolator import MotionInterpolator

        interpolator = MotionInterpolator()
        self.assertEqual(interpolator.target_fps, 60)
        expected_rife_bin = os.environ.get("RIFE_BIN", "/usr/local/bin/rife-ncnn-vulkan")
        self.assertEqual(interpolator.rife_binary_path, expected_rife_bin)

    def test_interpolator_initialization_custom(self):
        from src.enhancers.motion_interpolator import MotionInterpolator

        custom_bin = "/custom/tools/rife-vulkan"
        interpolator = MotionInterpolator(target_fps=120, rife_binary_path=custom_bin)
        self.assertEqual(interpolator.target_fps, 120)
        self.assertEqual(interpolator.rife_binary_path, custom_bin)

    def test_interpolate_raises_if_video_missing(self):
        from src.enhancers.motion_interpolator import MotionInterpolator

        interpolator = MotionInterpolator()
        with self.assertRaises(FileNotFoundError):
            interpolator.interpolate_60fps(
                video_path="/non_existent_directory_or_file/fake_input.mp4",
                output_path="/tmp/output.mp4",
            )

    @patch("src.enhancers.motion_interpolator.subprocess.run")
    def test_interpolate_with_rife_binary_present(self, mock_subprocess_run):
        from src.enhancers.motion_interpolator import MotionInterpolator

        rife_bin = "/usr/local/bin/rife-ncnn-vulkan"
        interpolator = MotionInterpolator(target_fps=60, rife_binary_path=rife_bin)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_video = os.path.join(tmp_dir, "input_24fps.mp4")
            output_video = os.path.join(tmp_dir, "sub", "output_60fps.mp4")
            with open(input_video, "w") as f:
                f.write("mock video data")

            # Mock os.path.exists to confirm rife_bin exists
            real_exists = os.path.exists

            def mock_exists(path):
                if path == rife_bin or path == input_video:
                    return True
                return real_exists(path)

            with patch("os.path.exists", side_effect=mock_exists):
                result = interpolator.interpolate_60fps(
                    video_path=input_video,
                    output_path=output_video,
                    multiplier=2,
                )

                self.assertEqual(result, output_video)
                mock_subprocess_run.assert_called_once()
                args, kwargs = mock_subprocess_run.call_args
                cmd = args[0]
                expected_cmd = [
                    rife_bin,
                    "-i",
                    input_video,
                    "-o",
                    output_video,
                    "-m",
                    "rife-v4",
                ]
                self.assertEqual(cmd, expected_cmd)
                self.assertTrue(kwargs.get("check"))

    @patch("src.enhancers.motion_interpolator.subprocess.run")
    def test_interpolate_with_ffmpeg_fallback_command(self, mock_subprocess_run):
        from src.enhancers.motion_interpolator import MotionInterpolator

        interpolator = MotionInterpolator(target_fps=60, rife_binary_path="/nonexistent/rife")

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_video = os.path.join(tmp_dir, "input_24fps.mp4")
            output_video = os.path.join(tmp_dir, "sub", "output_60fps.mp4")
            with open(input_video, "w") as f:
                f.write("mock video data")

            result = interpolator.interpolate_60fps(
                video_path=input_video,
                output_path=output_video,
            )

            self.assertEqual(result, output_video)
            mock_subprocess_run.assert_called_once()
            args, kwargs = mock_subprocess_run.call_args
            cmd = args[0]
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_video,
                "-filter:v",
                "minterpolate='fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "copy",
                output_video,
            ]
            self.assertEqual(cmd, expected_cmd)
            self.assertTrue(kwargs.get("check"))

    @patch("src.enhancers.motion_interpolator.subprocess.run")
    def test_interpolate_fallback_when_ffmpeg_missing(self, mock_subprocess_run):
        from src.enhancers.motion_interpolator import MotionInterpolator

        # Simulate system where neither rife nor ffmpeg executable exists
        mock_subprocess_run.side_effect = FileNotFoundError("ffmpeg not found on PATH")

        interpolator = MotionInterpolator(target_fps=60, rife_binary_path="/nonexistent/rife")

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_video = os.path.join(tmp_dir, "input_24fps.mp4")
            output_video = os.path.join(tmp_dir, "output_simulated.mp4")
            with open(input_video, "wb") as f:
                f.write(b"mock raw video bytes")

            # Should not crash, but fallback gracefully and create output container
            with self.assertLogs("src.enhancers.motion_interpolator", level="WARNING"):
                result = interpolator.interpolate_60fps(
                    video_path=input_video,
                    output_path=output_video,
                )

            self.assertEqual(result, output_video)
            self.assertTrue(os.path.exists(output_video))
            with open(output_video, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"mock raw video bytes")

    @patch("src.enhancers.motion_interpolator.subprocess.run")
    def test_interpolate_fallback_on_subprocess_error(self, mock_subprocess_run):
        from src.enhancers.motion_interpolator import MotionInterpolator

        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["ffmpeg", "-y"]
        )

        interpolator = MotionInterpolator(target_fps=60, rife_binary_path="/nonexistent/rife")

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_video = os.path.join(tmp_dir, "input_24fps.mp4")
            output_video = os.path.join(tmp_dir, "output_simulated_error.mp4")
            with open(input_video, "wb") as f:
                f.write(b"fallback stream bytes")

            with self.assertLogs("src.enhancers.motion_interpolator", level="WARNING"):
                result = interpolator.interpolate_60fps(
                    video_path=input_video,
                    output_path=output_video,
                )

            self.assertEqual(result, output_video)
            self.assertTrue(os.path.exists(output_video))

    def test_package_exports_motion_interpolator(self):
        import src.enhancers as enhancers

        self.assertTrue(hasattr(enhancers, "MotionInterpolator"))
        self.assertIn("MotionInterpolator", enhancers.__all__)


if __name__ == "__main__":
    unittest.main()
