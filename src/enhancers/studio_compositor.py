"""
Studio Compositor Engine.
Handles professional post-production finishing:
1. Dynamic Background Music (BGM) mixing with sidechain audio ducking.
2. Animated / Styled Subtitle generation and burn-in.
Utilizes battle-tested, zero-VRAM FFmpeg filter chains adapted from MoneyPrinterTurbo & OpenMontage.
"""

import os
import subprocess
from typing import Optional, List, Tuple
from ..utils.media import check_ffmpeg


class StudioCompositor:
    """
    Compositor responsible for BGM mixing, audio ducking, and subtitle rendering.
    Runs entirely through FFmpeg, requiring 0 MB of GPU VRAM.
    """

    def __init__(self):
        check_ffmpeg()

    def mix_bgm(
        self,
        video_path: str,
        bgm_path: str,
        output_path: str,
        bgm_volume: float = 0.15,
        enable_ducking: bool = True,
    ) -> str:
        """
        Mixes background music into a video with automatic voice ducking.
        When the avatar speaks, the BGM volume is ducked to prevent masking speech.

        Args:
            video_path: Path to the input video containing avatar speech.
            bgm_path: Path to the background audio/music track (.mp3, .wav).
            output_path: Path where the final video with mixed audio will be saved.
            bgm_volume: Baseline relative volume of BGM (0.0 to 1.0, default 0.15).
            enable_ducking: If True, uses sidechaincompress to dynamically lower BGM
                           whenever avatar voice exceeds threshold.

        Returns:
            Path to the output video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(bgm_path):
            raise FileNotFoundError(f"BGM file not found: {bgm_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Loop BGM to match video length if necessary, and mix audio streams
        if enable_ducking:
            # Sidechain ducking filter:
            # 1. Loop BGM infinitely
            # 2. Scale baseline BGM volume
            # 3. Duck BGM against avatar speech stream [0:a]
            # 4. Mix speech + ducked BGM
            filter_complex = (
                f"[1:a]aloop=loop=-1:size=2e+09,volume={bgm_volume}[bgm];"
                f"[bgm][0:a]sidechaincompress=threshold=0.05:ratio=4:attack=50:release=350[ducked_bgm];"
                f"[0:a][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        else:
            filter_complex = (
                f"[1:a]aloop=loop=-1:size=2e+09,volume={bgm_volume}[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        print(f"[*] Mixing BGM ('{os.path.basename(bgm_path)}') with ducking={enable_ducking}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"[-] BGM mix warning: {res.stderr}")
            # Fallback simple amix without sidechain
            fallback_filter = f"[1:a]aloop=loop=-1:size=2e+09,volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first[aout]"
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", bgm_path,
                "-filter_complex", fallback_filter,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ]
            subprocess.run(fallback_cmd, check=True)

        return output_path

    def generate_srt_from_script(
        self,
        script_text: str,
        total_duration_sec: float,
        srt_output_path: str,
    ) -> str:
        """
        Creates a time-synced SRT subtitle file distributed evenly across sentence clauses.
        Ideal when transcript timing from ASR is not pre-computed.

        Args:
            script_text: Full script spoken by avatar.
            total_duration_sec: Exact duration of the video/audio in seconds.
            srt_output_path: Path to write the .srt file.

        Returns:
            Path to the saved .srt file.
        """
        import re

        # Split into clauses by punctuation
        clauses = [c.strip() for c in re.split(r"[.!?,\n]+", script_text) if c.strip()]
        if not clauses:
            clauses = [script_text.strip()]

        total_words = sum(len(c.split()) for c in clauses)
        if total_words == 0:
            total_words = 1

        sec_per_word = total_duration_sec / total_words

        def format_timestamp(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        os.makedirs(os.path.dirname(srt_output_path), exist_ok=True)
        current_time = 0.0

        with open(srt_output_path, "w", encoding="utf-8") as f:
            for idx, clause in enumerate(clauses, start=1):
                word_count = len(clause.split())
                clause_duration = max(1.0, word_count * sec_per_word)
                end_time = min(total_duration_sec, current_time + clause_duration)

                f.write(f"{idx}\n")
                f.write(f"{format_timestamp(current_time)} --> {format_timestamp(end_time)}\n")
                f.write(f"{clause}\n\n")

                current_time = end_time

        return srt_output_path

    def burn_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font_size: int = 24,
        font_color: str = "&H00FFFFFF",  # ASS format: BGR hex (White = &H00FFFFFF)
        outline_color: str = "&H00000000",  # Black border
        border_width: int = 3,
    ) -> str:
        """
        Burns styled subtitles permanently into video frames using FFmpeg's libass filter.

        Args:
            video_path: Input video file.
            srt_path: Subtitle SRT file.
            output_path: Output burned video file.
            font_size: Font size for subtitles (default 24).
            font_color: Font color in ASS hex format.
            outline_color: Outline border color in ASS hex format.
            border_width: Outline stroke width.

        Returns:
            Path to the output video with burned subtitles.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Normalize path for FFmpeg filter argument
        escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        style = f"FontSize={font_size},PrimaryColour={font_color},OutlineColour={outline_color},BorderStyle=1,Outline={border_width},MarginV=30"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{escaped_srt}':force_style='{style}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]

        print(f"[*] Burning styled subtitles from '{os.path.basename(srt_path)}'...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
