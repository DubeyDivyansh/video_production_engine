"""
Full-Body Animator & Studio Canvas Compositor
Prepares whole-body still portraits for diffusion lip-sync (LatentSync)
by applying pure white studio backdrops (#FFFFFF), mod-16 dimension padding,
and generating base MP4 videos with natural subtle breathing dynamics.
"""

import os
import math
import shutil
import subprocess
from typing import Tuple, List, Dict, Optional, Any

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FullBodyAnimator:
    """
    Handles image-to-video base synthesis for whole-body avatars,
    background matting onto pure white studio backdrops,
    and dimension compliance for diffusion models.
    """

    def __init__(self, bg_color_hex: str = "#FFFFFF"):
        self.bg_color_hex = bg_color_hex
        self.bg_rgb = self._hex_to_rgb(bg_color_hex)

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def get_target_mod16_dimensions(width: int, height: int) -> Tuple[int, int]:
        """
        Calculates the smallest dimensions >= (width, height) that are divisible by 16.
        Required by UNet latent representations (downsampled 8x/16x).
        """
        target_w = int(math.ceil(width / 16.0) * 16)
        target_h = int(math.ceil(height / 16.0) * 16)
        return target_w, target_h

    def pad_image_to_mod16(self, img_pil: Any, bg_color: Optional[Tuple[int, int, int]] = None) -> Any:
        """
        Centers the PIL image within mod-16 dimensions, padding borders with bg_color.
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for image padding.")

        color = bg_color or self.bg_rgb
        w, h = img_pil.size
        target_w, target_h = self.get_target_mod16_dimensions(w, h)

        if target_w == w and target_h == h:
            return img_pil

        padded = Image.new("RGB", (target_w, target_h), color=color)
        offset_x = (target_w - w) // 2
        offset_y = (target_h - h) // 2
        padded.paste(img_pil, (offset_x, offset_y))
        return padded

    def composite_white_studio(
        self,
        image_path: str,
        output_path: str,
        matting_enhancer: Optional[Any] = None,
    ) -> str:
        """
        Isolates the subject using BiRefNet background matting (if available)
        and composites onto a pure white studio canvas, padded to mod-16 dimensions.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not PIL_AVAILABLE:
            shutil.copy2(image_path, output_path)
            return output_path

        # 1. Background removal if enhancer provided
        composite_done = False
        if matting_enhancer is not None:
            try:
                # Use project's BackgroundMattingEnhancer
                temp_matted = output_path + ".matted.png"
                matting_enhancer.remove_background(image_path, temp_matted)
                if os.path.exists(temp_matted):
                    fg = Image.open(temp_matted).convert("RGBA")
                    bg = Image.new("RGBA", fg.size, self.bg_rgb + (255,))
                    bg.paste(fg, (0, 0), fg)
                    final_rgb = bg.convert("RGB")
                    final_mod16 = self.pad_image_to_mod16(final_rgb, self.bg_rgb)
                    final_mod16.save(output_path, quality=95)
                    composite_done = True
                    if os.path.exists(temp_matted):
                        os.remove(temp_matted)
            except Exception as e:
                print(f"[FullBodyAnimator] Note: BiRefNet matting fallback ({e}), using padded source.")

        if not composite_done:
            # Fallback: load source and pad to mod-16 with clean studio background
            img = Image.open(image_path).convert("RGB")
            padded = self.pad_image_to_mod16(img, self.bg_rgb)
            padded.save(output_path, quality=95)

        return output_path

    def calculate_motion_transforms(
        self,
        num_frames: int,
        fps: int = 25,
        motion_mode: str = "subtle_breathing"
    ) -> List[Dict[str, float]]:
        """
        Generates frame-by-frame subtle affine motion parameters.
        - subtle_breathing: Sinusoidal expansion (1.000 to 1.007) and vertical displacement (+/- 2px).
        - static_hold: 1.0 scale and 0.0 displacement across all frames.
        """
        transforms = []
        if motion_mode == "static_hold" or num_frames <= 0:
            return [{"scale": 1.0, "dx": 0.0, "dy": 0.0} for _ in range(num_frames)]

        # Breathing cycle frequency: ~0.22 Hz (one natural breath every ~4.5 seconds)
        freq = 0.22
        for i in range(num_frames):
            t = i / float(fps)
            sin_val = math.sin(2.0 * math.pi * freq * t)
            # Scale expands gently on inhale (up to +0.7%) and relaxes on exhale
            scale = 1.0 + 0.0035 * (sin_val + 1.0)
            # Torso rises ~2px on inhale, lowers on exhale
            dy = -1.8 * sin_val
            transforms.append({"scale": scale, "dx": 0.0, "dy": dy})

        return transforms

    def synthesize_base_video(
        self,
        image_path: str,
        duration_s: float,
        output_video_path: str,
        fps: int = 25,
        motion_mode: str = "subtle_breathing",
        matting_enhancer: Optional[Any] = None,
    ) -> str:
        """
        Synthesizes a continuous MP4 base video of specified duration
        from a whole-body portrait, applying white studio compositing and subtle breathing motion.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        num_frames = max(1, int(math.ceil(duration_s * fps)))

        # 1. Prepare studio white composite image
        prepared_img_path = output_video_path + ".prepared.jpg"
        self.composite_white_studio(image_path, prepared_img_path, matting_enhancer=matting_enhancer)

        # 2. Check for motion requirement
        if motion_mode == "static_hold" or not PIL_AVAILABLE:
            # Fast-path via ffmpeg single image loop
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", prepared_img_path,
                "-t", str(duration_s),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(prepared_img_path):
                    os.remove(prepared_img_path)
                return output_video_path
            except Exception as e:
                print(f"[FullBodyAnimator] ffmpeg loop note: {e}")

        # 3. Apply sinusoidal breathing motion via frame transforms
        try:
            transforms = self.calculate_motion_transforms(num_frames, fps, motion_mode)
            base_img = Image.open(prepared_img_path).convert("RGB")
            w, h = base_img.size

            # Create temp directory for motion frames
            temp_frames_dir = output_video_path + "_frames"
            os.makedirs(temp_frames_dir, exist_ok=True)

            for i, trans in enumerate(transforms):
                s = trans["scale"]
                dy = trans["dy"]
                if abs(s - 1.0) > 1e-4 or abs(dy) > 1e-4:
                    new_w = int(round(w * s))
                    new_h = int(round(h * s))
                    resized = base_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                    frame = Image.new("RGB", (w, h), color=self.bg_rgb)
                    paste_x = (w - new_w) // 2
                    paste_y = int(round((h - new_h) // 2 + dy))
                    frame.paste(resized, (paste_x, paste_y))
                else:
                    frame = base_img

                frame_path = os.path.join(temp_frames_dir, f"frame_{i:05d}.jpg")
                frame.save(frame_path, quality=95)

            # Assemble into MP4 via ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-r", str(fps),
                "-i", os.path.join(temp_frames_dir, "frame_%05d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Cleanup temp frames
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            if os.path.exists(prepared_img_path):
                os.remove(prepared_img_path)

            return output_video_path

        except Exception as e:
            print(f"[FullBodyAnimator] Motion video assembly note: {e}, falling back to static loop.")
            # Ultimate fallback: loop image directly
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", prepared_img_path,
                "-t", str(duration_s),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(prepared_img_path):
                os.remove(prepared_img_path)
            return output_video_path
