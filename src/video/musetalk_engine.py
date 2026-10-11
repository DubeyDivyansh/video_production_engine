"""
MuseTalk Video Engine
Real-Time High Quality Audio-Driven Lip-Syncing using Latent Inpainting.
Lightweight footprint on Tesla T4 (~5.2 GB VRAM).
"""

import os
import subprocess


class MuseTalkEngine:
    """
    Wrapper for TMElyralab/MuseTalk lip-sync inference.
    """

    def __init__(self, repo_dir: str = "/content/MuseTalk"):
        self.repo_dir = repo_dir

    def ensure_installed(self):
        """
        Clones MuseTalk repository if not present.
        """
        if not os.path.exists(self.repo_dir):
            print(f"[MuseTalk] Cloning repository to {self.repo_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/TMElyralab/MuseTalk.git", self.repo_dir],
                check=True,
            )

    def sync_lips(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str = "./results",
        bbox_shift: int = 0,
    ) -> str:
        """
        Applies audio-driven lip synchronization to a video or face bounding box.

        Args:
            video_path: Path to target video file.
            audio_path: Path to driving speech audio file.
            output_dir: Directory where the result will be saved.
            bbox_shift: Bounding box shift parameter for face framing adjustments.

        Returns:
            Path to rendered lip-synced video.
        """
        self.ensure_installed()
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", "-m", "scripts.inference",
            "--video_path", os.path.abspath(video_path),
            "--audio_path", os.path.abspath(audio_path),
            "--result_dir", os.path.abspath(output_dir),
            "--bbox_shift", str(bbox_shift),
        ]

        print(f"[MuseTalk] Running latent inpainting lip-sync...")
        subprocess.run(cmd, cwd=self.repo_dir, check=True)

        output_path = os.path.join(output_dir, "musetalk_output.mp4")
        return output_path
