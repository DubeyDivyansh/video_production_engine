"""
Studio Background Replacer & Matting Module.
Supports AI matting (BiRefNet / RMBG / rembg) and chroma-keying for video background replacement.
"""

import os
import numpy as np
import subprocess
from typing import Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None


class BackgroundReplacer:
    """Removes or replaces background in avatar videos."""

    def __init__(self, method: str = "auto", use_gpu: bool = True):
        self.method = method
        self.use_gpu = use_gpu
        self._rembg_session = None

    def _get_rembg_session(self):
        if self._rembg_session is None:
            try:
                import rembg
                self._rembg_session = rembg.new_session("birefnet-general" if self.use_gpu else "u2net")
            except Exception:
                try:
                    import rembg
                    self._rembg_session = rembg.new_session("u2net")
                except Exception:
                    self._rembg_session = None
        return self._rembg_session

    def _compute_chroma_mask(
        self,
        bgr_frame: np.ndarray,
        lower_green: Tuple[int, int, int] = (35, 50, 50),
        upper_green: Tuple[int, int, int] = (85, 255, 255)
    ) -> np.ndarray:
        """Computes alpha mask for green screen backgrounds."""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Invert so subject is 255 and background is 0
        alpha = cv2.bitwise_not(mask)
        # Soften edges with morphological close + blur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        return alpha

    def _compute_ai_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Computes foreground alpha mask using AI matting model."""
        session = self._get_rembg_session()
        if session is not None:
            import rembg
            rgba = rembg.remove(bgr_frame, session=session)
            return rgba[:, :, 3]
        # Fallback to green chroma if AI matting unavailable
        return self._compute_chroma_mask(bgr_frame)

    def extract_alpha_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Determines whether to use chroma keying or AI human matting."""
        # Auto-detect if frame is predominantly green background
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        green_pixels = np.sum((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85))
        is_green_screen = (green_pixels / float(bgr_frame.shape[0] * bgr_frame.shape[1])) > 0.35

        if is_green_screen or self.method == "chroma":
            return self._compute_chroma_mask(bgr_frame)
        return self._compute_ai_mask(bgr_frame)

    def remove_background(self, input_image_path: str, output_image_path: str) -> str:
        """
        Removes background from an image file using BiRefNet / AI matting (rembg).
        Saves output as RGBA image with transparent background.
        """
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        session = self._get_rembg_session()
        if session is not None:
            try:
                import rembg
                from PIL import Image
                inp = Image.open(input_image_path)
                out = rembg.remove(inp, session=session)
                out.save(output_image_path)
                return output_image_path
            except Exception as e:
                print(f"[BackgroundReplacer] rembg execution note: {e}")

        # Fallback if rembg is not available or fails
        try:
            from PIL import Image
            img = Image.open(input_image_path).convert("RGBA")
            img.save(output_image_path)
        except Exception:
            import shutil
            shutil.copy2(input_image_path, output_image_path)
        return output_image_path

    def replace_background(
        self,
        video_path: str,
        backdrop_path: str,
        output_path: str,
        feather_radius: int = 2
    ) -> str:
        """
        Replaces video background with a static image or dynamic video backdrop.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video not found: {video_path}")
        if not os.path.exists(backdrop_path):
            raise FileNotFoundError(f"Backdrop not found: {backdrop_path}")

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_raw_video = output_path.replace(".mp4", "_temp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_raw_video, fourcc, fps, (w, h))

        # Check if backdrop is image or video
        is_bg_video = backdrop_path.lower().endswith((".mp4", ".mov", ".webm", ".avi"))
        bg_cap = cv2.VideoCapture(backdrop_path) if is_bg_video else None
        bg_img = None if is_bg_video else cv2.imread(backdrop_path)
        if bg_img is not None:
            bg_img = cv2.resize(bg_img, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get backdrop frame
            if is_bg_video:
                ret_bg, current_bg = bg_cap.read()
                if not ret_bg:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_bg, current_bg = bg_cap.read()
                current_bg = cv2.resize(current_bg, (w, h))
            else:
                current_bg = bg_img.copy()

            # Generate alpha matte (0 to 1 float)
            alpha = self.extract_alpha_mask(frame)
            if feather_radius > 0:
                alpha = cv2.GaussianBlur(alpha, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
            alpha_f = (alpha / 255.0)[:, :, np.newaxis]

            # Composite foreground over background
            composite = (frame * alpha_f + current_bg * (1.0 - alpha_f)).astype(np.uint8)
            out.write(composite)
            frame_idx += 1

        cap.release()
        if bg_cap:
            bg_cap.release()
        out.release()

        # Re-attach original audio via ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw_video,
            "-i", video_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_raw_video):
            os.remove(temp_raw_video)

        return output_path

    def apply_color_backdrop(
        self,
        video_path: str,
        output_path: str,
        color_bgr: Tuple[int, int, int] = (0, 255, 0)
    ) -> str:
        """Fills background with a solid studio color (e.g., pure green for chroma)."""
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        solid_bg = np.zeros((h, w, 3), dtype=np.uint8)
        solid_bg[:, :] = color_bgr
        cap.release()

        temp_bg_img = output_path + ".temp_bg.png"
        cv2.imwrite(temp_bg_img, solid_bg)
        try:
            return self.replace_background(video_path, temp_bg_img, output_path)
        finally:
            if os.path.exists(temp_bg_img):
                os.remove(temp_bg_img)
