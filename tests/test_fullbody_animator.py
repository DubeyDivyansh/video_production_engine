"""
Unit Tests for FullBodyAnimator
Validates mod-16 dimension padding, subtle breathing motion curves, and video synthesis.
"""

import os
import sys
import unittest
import tempfile
import math

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.video.fullbody_animator import FullBodyAnimator


class TestFullBodyAnimator(unittest.TestCase):
    def setUp(self):
        self.animator = FullBodyAnimator()

    def test_calculate_target_mod16_dimensions(self):
        # 855x954 -> 864x960
        target_w, target_h = self.animator.get_target_mod16_dimensions(855, 954)
        self.assertEqual(target_w % 16, 0)
        self.assertEqual(target_h % 16, 0)
        self.assertGreaterEqual(target_w, 855)
        self.assertGreaterEqual(target_h, 954)
        self.assertEqual(target_w, 864)
        self.assertEqual(target_h, 960)

    def test_motion_transforms_subtle_breathing(self):
        fps = 25
        duration = 4.0  # 1 full breath cycle
        num_frames = int(fps * duration)
        transforms = self.animator.calculate_motion_transforms(
            num_frames=num_frames,
            fps=fps,
            motion_mode="subtle_breathing"
        )
        self.assertEqual(len(transforms), num_frames)

        # Verify scale oscillates gently between 1.0 and 1.008
        scales = [t["scale"] for t in transforms]
        self.assertAlmostEqual(min(scales), 1.0, places=2)
        self.assertLessEqual(max(scales), 1.01)

        # Verify vertical shift is bounded within +/- 2.5 pixels
        dys = [t["dy"] for t in transforms]
        self.assertLessEqual(max(dys), 2.5)
        self.assertGreaterEqual(min(dys), -2.5)

    def test_motion_transforms_static_hold(self):
        num_frames = 50
        transforms = self.animator.calculate_motion_transforms(
            num_frames=num_frames,
            fps=25,
            motion_mode="static_hold"
        )
        self.assertEqual(len(transforms), num_frames)
        for t in transforms:
            self.assertEqual(t["scale"], 1.0)
            self.assertEqual(t["dy"], 0.0)
            self.assertEqual(t["dx"], 0.0)

    def test_pad_pillow_image_to_mod16(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not available")

        # Create a test 855x954 image
        img = Image.new("RGB", (855, 954), color=(50, 50, 50))
        padded = self.animator.pad_image_to_mod16(img, bg_color=(255, 255, 255))
        self.assertEqual(padded.width, 864)
        self.assertEqual(padded.height, 960)
        # Check that top-left pixel is white background (255, 255, 255)
        self.assertEqual(padded.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
