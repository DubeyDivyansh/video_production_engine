"""
LatentSync Video Engine
Audio-conditioned Latent Diffusion Lip-Sync.
Preserves full-body motion, gestures, and natural posture without cropping.
Optimized for Google Colab Tesla T4 (Turing, sm_75) in FP16 precision.
"""

import os
import re
import subprocess
from ..utils.memory import configure_t4_allocator


class LatentSyncEngine:
    """
    Wrapper for LatentSync 1.5 inference with T4 VRAM optimizations.
    """

    def __init__(self, repo_dir: str = "/content/LatentSync"):
        self.repo_dir = repo_dir

    def ensure_installed_and_patched(self):
        """
        Clones LatentSync and patches `scripts/inference.py` to enable FP16 on Tesla T4 (sm_75).
        """
        if not os.path.exists(self.repo_dir):
            print(f"[LatentSync] Cloning repository to {self.repo_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/bytedance/LatentSync.git", self.repo_dir],
                check=True,
            )

        # Apply T4 Turing compute capability patch (replace > 7 with >= 7)
        inference_file = os.path.join(self.repo_dir, "scripts", "inference.py")
        if os.path.exists(inference_file):
            with open(inference_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            old_pattern = r"torch\.cuda\.get_device_capability\(\)\[0\]\s*>\s*7"
            if re.search(old_pattern, content):
                new_content = re.sub(old_pattern, "torch.cuda.get_device_capability()[0] >= 7", content)
                with open(inference_file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print("[LatentSync] Applied Turing T4 FP16 patch successfully.")

    def sync_lips(
        self,
        video_path: str,
        audio_path: str,
        output_video_path: str,
        unet_config: str = "configs/unet/second_stage.yaml",
        ckpt_path: str = "checkpoints/latentsync_unet.pt",
    ) -> str:
        """
        Synchronizes lips on an existing body video or portrait base using LatentSync.

        Args:
            video_path: Path to source MP4 video with full body posture / movement.
            audio_path: Path to driving speech audio (e.g. from F5-TTS).
            output_video_path: Destination path for the synchronized MP4 video.
            unet_config: Path to UNet configuration yaml.
            ckpt_path: Path to LatentSync model weights checkpoint.

        Returns:
            Path to the output MP4 video.
        """
        self.ensure_installed_and_patched()
        configure_t4_allocator()

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

        cmd = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", unet_config,
            "--inference_ckpt_path", ckpt_path,
            "--video_path", os.path.abspath(video_path),
            "--audio_path", os.path.abspath(audio_path),
            "--video_out_path", os.path.abspath(output_video_path),
        ]

        print(f"[LatentSync] Running diffusion lip-sync inference...")
        print(f"    - Input Video: {video_path}")
        print(f"    - Driven Audio: {audio_path}")
        print(f"    - Output Destination: {output_video_path}")

        subprocess.run(cmd, cwd=self.repo_dir, check=True)
        print(f"[LatentSync] Successfully rendered: {output_video_path}")
        return output_video_path

    def sync_lips_from_image(
        self,
        image_path: str,
        audio_path: str,
        output_video_path: str,
        motion_mode: str = "subtle_breathing",
        studio_backdrop: str = "#FFFFFF",
        unet_config: str = "configs/unet/second_stage.yaml",
        ckpt_path: str = "checkpoints/latentsync_unet.pt",
    ) -> str:
        """
        Synthesizes a full-body base video from a portrait/pose image on a white studio canvas,
        then performs diffusion lip-sync using LatentSync.
        """
        from .fullbody_animator import FullBodyAnimator
        try:
            import soundfile as sf
            duration_s = float(sf.info(audio_path).duration)
        except Exception:
            import wave
            with wave.open(audio_path, 'rb') as f:
                duration_s = float(f.getnframes()) / float(f.getframerate())

        temp_dir = os.path.dirname(os.path.abspath(output_video_path))
        base_video_path = os.path.join(temp_dir, "temp_base_body_video.mp4")

        animator = FullBodyAnimator(bg_color_hex=studio_backdrop)
        animator.synthesize_base_video(
            image_path=image_path,
            duration_s=duration_s,
            output_video_path=base_video_path,
            motion_mode=motion_mode,
        )

        try:
            return self.sync_lips(
                video_path=base_video_path,
                audio_path=audio_path,
                output_video_path=output_video_path,
                unet_config=unet_config,
                ckpt_path=ckpt_path,
            )
        finally:
            if os.path.exists(base_video_path):
                os.remove(base_video_path)
