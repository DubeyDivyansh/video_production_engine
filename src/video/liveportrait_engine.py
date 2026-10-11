"""
LivePortrait Video Engine
High-fidelity, expressive portrait animation from a single image or video driver.
Provides natural eye blinks, gaze shifts, smiles, and head nods.
Ideal as a single-image avatar fallback.
"""

import os
import subprocess


class LivePortraitEngine:
    """
    Wrapper for KwaiVGI/LivePortrait inference on Google Colab T4.
    Runs comfortably within ~3.8 GB VRAM.
    """

    def __init__(self, repo_dir: str = "/content/LivePortrait"):
        self.repo_dir = repo_dir

    def ensure_installed(self):
        """
        Clones LivePortrait repository if not present.
        """
        if not os.path.exists(self.repo_dir):
            print(f"[LivePortrait] Cloning repository to {self.repo_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/KwaiVGI/LivePortrait.git", self.repo_dir],
                check=True,
            )

    def generate(
        self,
        source_image: str,
        driving_video: str,
        output_dir: str = "./animations",
        flag_eye_retargeting: bool = True,
        flag_lip_retargeting: bool = True,
    ) -> str:
        """
        Drives a still portrait using facial motion extracted from a driving video.

        Args:
            source_image: Path to the target face portrait (JPEG/PNG).
            driving_video: Path to motion driving MP4 video.
            output_dir: Destination directory for rendered outputs.
            flag_eye_retargeting: Retargets natural eye movements and blinks.
            flag_lip_retargeting: Retargets lip movements.

        Returns:
            Path to the generated MP4 animation.
        """
        self.ensure_installed()
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", "inference.py",
            "-s", os.path.abspath(source_image),
            "-d", os.path.abspath(driving_video),
            "-o", os.path.abspath(output_dir),
        ]
        if flag_eye_retargeting:
            cmd.append("--flag_eye_retargeting")
        if flag_lip_retargeting:
            cmd.append("--flag_lip_retargeting")

        print(f"[LivePortrait] Generating expressive motion for: {source_image}...")
        subprocess.run(cmd, cwd=self.repo_dir, check=True)

        output_path = os.path.join(output_dir, "liveportrait_output.mp4")
        return output_path
