"""
SadTalker Video Engine
3D Morphable Model (3DMM) Coefficients + GFPGAN Face Restoration
Ideal for animating single portraits and still photos.
"""

import glob
import os
import shutil
import subprocess


class SadTalkerEngine:
    """
    Wrapper for executing SadTalker animation inference on Google Colab GPU.
    """

    def __init__(self, repo_dir: str = "/content/SadTalker"):
        self.repo_dir = repo_dir

    def ensure_installed(self):
        """
        Ensures the repository is cloned and checkpoints exist.
        """
        if not os.path.exists(self.repo_dir):
            print(f"[SadTalker] Cloning repository to {self.repo_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/Winfredy/SadTalker.git", self.repo_dir],
                check=True,
            )
        
        checkpoints_dir = os.path.join(self.repo_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
            print("[SadTalker] Downloading model checkpoints...")
            subprocess.run(
                ["bash", "scripts/download_models.sh"],
                cwd=self.repo_dir,
                check=True,
            )

    def generate(
        self,
        source_image: str,
        driven_audio: str,
        output_dir: str = "./results",
        still_mode: bool = True,
        use_enhancer: bool = True,
    ) -> str:
        """
        Animates a single face portrait driven by an audio file.

        Args:
            source_image: Path to JPEG/PNG portrait photo.
            driven_audio: Path to speech audio WAV.
            output_dir: Destination directory for rendered video.
            still_mode: If True, reduces head motion wobble for cleaner portraits.
            use_enhancer: If True, applies GFPGAN face restoration.

        Returns:
            Path to the generated MP4 file.
        """
        self.ensure_installed()

        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python3.8", "inference.py",
            "--driven_audio", os.path.abspath(driven_audio),
            "--source_image", os.path.abspath(source_image),
            "--result_dir", os.path.abspath(output_dir),
            "--preprocess", "full",
        ]
        if still_mode:
            cmd.append("--still")
        if use_enhancer:
            cmd.extend(["--enhancer", "gfpgan"])

        print(f"[SadTalker] Running inference for: {source_image}...")
        subprocess.run(cmd, cwd=self.repo_dir, check=True)

        # Locate latest generated MP4 in the output directory
        pattern = os.path.join(output_dir, "*", "*.mp4")
        generated_videos = sorted(glob.glob(pattern))
        if not generated_videos:
            pattern_flat = os.path.join(output_dir, "*.mp4")
            generated_videos = sorted(glob.glob(pattern_flat))

        if not generated_videos:
            raise RuntimeError("[SadTalker] Inference completed but output MP4 was not found.")

        latest_video = generated_videos[-1]
        print(f"[SadTalker] Output video rendered: {latest_video}")
        return latest_video
