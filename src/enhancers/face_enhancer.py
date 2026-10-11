"""
Face Restoration and Super-Resolution Module.
Modular wrapper around GFPGAN and CodeFormer for high-fidelity facial detail upscaling.
"""

import os
import numpy as np
import subprocess
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None


class FaceEnhancer:
    """Enhances facial clarity, sharpness, and mouth boundaries in generated videos."""

    def __init__(self, model_name: str = "gfpgan", upscale: int = 1, use_gpu: bool = True):
        self.model_name = model_name
        self.upscale = upscale
        self.use_gpu = use_gpu
        self._restorer = None

    def _init_restorer(self):
        if self._restorer is not None:
            return self._restorer

        try:
            from gfpgan import GFPGANer
            model_path = "/content/drive/MyDrive/video_gen_platform/checkpoints/gfpgan/GFPGANv1.4.pth"
            if not os.path.exists(model_path):
                model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

            self._restorer = GFPGANer(
                model_path=model_path,
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None
            )
        except Exception:
            self._restorer = None
        return self._restorer

    def enhance_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """Restores a single BGR image/frame."""
        restorer = self._init_restorer()
        if restorer is not None:
            _, _, restored_face = restorer.enhance(
                bgr_image, has_aligned=False, only_center_face=False, paste_back=True
            )
            return restored_face
        # Fallback to mild unsharp masking if GFPGAN not loaded
        gaussian = cv2.GaussianBlur(bgr_image, (0, 0), 2.0)
        return cv2.addWeighted(bgr_image, 1.3, gaussian, -0.3, 0)

    def enhance_video(self, input_video: str, output_video: str) -> str:
        """Processes and restores all facial frames in a video."""
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")

        cap = cv2.VideoCapture(input_video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        temp_out = output_video.replace(".mp4", "_enhanced_temp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            enhanced_frame = self.enhance_image(frame)
            if enhanced_frame.shape[:2] != (h, w):
                enhanced_frame = cv2.resize(enhanced_frame, (w, h))
            out.write(enhanced_frame)

        cap.release()
        out.release()

        # Remux audio
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_out,
            "-i", input_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", output_video
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_out):
            os.remove(temp_out)

        return output_video
