"""
Unit Tests for VisualWatcher (Video-Watching & Visual QA Engine).
Tests scene-change detection, hook microscope, and visual report generation.
"""

import os
import sys
import tempfile
import unittest
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verification import VisualWatcher, VisualInspectionResult


class TestVisualWatcher(unittest.TestCase):
    def setUp(self):
        self.watcher = VisualWatcher()

    def _create_synthetic_video(self, file_path: str, duration_sec: float = 3.0, cuts: bool = True):
        """Creates a synthetic MP4 video with or without shot cuts."""
        fps = 10
        w, h = 160, 160
        num_frames = int(duration_sec * fps)
        frames = []

        for i in range(num_frames):
            frame = np.ones((h, w), dtype=np.uint8) * 100
            if cuts:
                # Change scene color every 1 second (every 10 frames)
                scene_idx = i // 10
                frame[:, :] = (scene_idx * 70) % 255
            frames.append(frame)

        raw = b"".join([f.tobytes() for f in frames])
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-pix_fmt", "yuv420p", file_path
        ]
        subprocess.run(cmd, input=raw, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_watch_video_with_cuts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            v_path = os.path.join(tmp_dir, "test_cuts.mp4")
            self._create_synthetic_video(v_path, duration_sec=3.0, cuts=True)

            res = self.watcher.watch_video(v_path, out_dir=os.path.join(tmp_dir, "output"))
            self.assertIsInstance(res, VisualInspectionResult)
            self.assertAlmostEqual(res.duration_seconds, 3.0, places=1)
            self.assertGreater(len(res.hero_frames), 0)
            self.assertTrue(os.path.exists(res.report_markdown_path))

    def test_watch_video_static_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            v_path = os.path.join(tmp_dir, "test_static.mp4")
            self._create_synthetic_video(v_path, duration_sec=2.0, cuts=False)

            res = self.watcher.watch_video(v_path, out_dir=os.path.join(tmp_dir, "output_static"))
            self.assertTrue(res.is_static_footage)
            self.assertTrue(os.path.exists(res.report_markdown_path))
            with open(res.report_markdown_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Static", content)


if __name__ == "__main__":
    unittest.main()
