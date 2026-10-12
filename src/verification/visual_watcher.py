"""
Visual Watcher & Inspection Engine (Inspired by claude-watch).
Provides visual video-watching capabilities for autonomous agents:
- Scene-change frame extraction (one frame per shot).
- 0-10s Hook microscope (dense sampling of opening moments).
- Editorial pacing metrics (cuts per minute, shot length).
- Detection of static/frozen footage.
- Ingest-ready visual audit report generation with embedded frame snapshots.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class VisualInspectionResult:
    video_path: str
    duration_seconds: float
    width: int
    height: int
    hero_frames: List[str] = field(default_factory=list)
    all_frames: List[Dict[str, Any]] = field(default_factory=list)
    hook_frames: List[str] = field(default_factory=list)
    cuts_count: int = 0
    cuts_per_minute: float = 0.0
    is_static_footage: bool = False
    report_markdown_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualWatcher:
    """
    Autonomous Video Watcher that extracts scene-change frames,
    editorial pacing metrics, and hook snapshots for thorough agent QA.
    """

    def __init__(self, output_base_dir: Optional[str] = None):
        self.output_base_dir = output_base_dir

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Probes video duration and dimensions via ffprobe."""
        meta = {"duration": 0.0, "width": 1280, "height": 720, "fps": 25.0}
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration:stream=width,height,r_frame_rate",
                "-of", "json",
                video_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
            data = json.loads(res.stdout)
            meta["duration"] = float(data.get("format", {}).get("duration", 0.0))
            for s in data.get("streams", []):
                if "width" in s and s.get("width"):
                    meta["width"] = int(s["width"])
                    meta["height"] = int(s["height"])
                    break
        except Exception:
            pass
        return meta

    def watch_video(
        self,
        video_path: str,
        out_dir: Optional[str] = None,
        max_frames: int = 40,
        scene_threshold: float = 0.25,
        extract_hook: bool = True,
    ) -> VisualInspectionResult:
        """
        Watches a video by extracting scene changes, hook microscope, and pacing.

        Args:
            video_path: Path to MP4/WebM video file.
            out_dir: Optional destination directory for extracted frames.
            max_frames: Cap on total scene frames.
            scene_threshold: FFmpeg scene change sensitivity (default 0.25).
            extract_hook: Whether to extract dense frames for the first 10 seconds.

        Returns:
            VisualInspectionResult with all frame paths and pacing metrics.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        meta = self._get_video_metadata(video_path)
        duration = meta["duration"]

        if out_dir:
            work_dir = os.path.abspath(out_dir)
        elif self.output_base_dir:
            work_dir = os.path.join(self.output_base_dir, "visual_watch")
        else:
            work_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "visual_watch")

        frames_dir = os.path.join(work_dir, "frames")
        hook_dir = os.path.join(work_dir, "hook")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(hook_dir, exist_ok=True)

        # 1. Scene-Change Frame Extraction via FFmpeg
        scene_pattern = os.path.join(frames_dir, "shot_%03d.jpg")
        cmd_scene = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='gt(scene,{scene_threshold})',scale=512:-1",
            "-vsync", "vfr",
            "-q:v", "3",
            scene_pattern
        ]
        try:
            subprocess.run(cmd_scene, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            pass

        extracted_shots = sorted([
            os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ])

        # If scene-change detected 0 or very few frames, fall back to auto-interval sampling
        is_static = len(extracted_shots) <= 1
        if len(extracted_shots) < 4 and duration > 2.0:
            sample_rate = max(0.5, float(max_frames) / max(1.0, duration))
            fallback_pattern = os.path.join(frames_dir, "frame_%03d.jpg")
            cmd_uniform = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={sample_rate:.3f},scale=512:-1",
                "-q:v", "4",
                fallback_pattern
            ]
            try:
                subprocess.run(cmd_uniform, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                extracted_shots = sorted([
                    os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")
                ])
            except Exception:
                pass

        # Limit to max_frames
        if len(extracted_shots) > max_frames:
            step = max(1, len(extracted_shots) // max_frames)
            extracted_shots = extracted_shots[::step][:max_frames]

        # Select Hero Frames (up to 5 key moments)
        hero_frames = []
        if extracted_shots:
            hero_indices = [0, len(extracted_shots) // 4, len(extracted_shots) // 2, (3 * len(extracted_shots)) // 4, len(extracted_shots) - 1]
            hero_frames = [extracted_shots[idx] for idx in sorted(set(hero_indices)) if idx < len(extracted_shots)]

        # 2. Hook Microscope (0-10s dense frames)
        hook_frames = []
        if extract_hook and duration > 1.0:
            hook_duration = min(10.0, duration)
            hook_pattern = os.path.join(hook_dir, "hook_%02d.jpg")
            cmd_hook = [
                "ffmpeg", "-y", "-ss", "0", "-to", str(hook_duration),
                "-i", video_path,
                "-vf", "fps=2,scale=512:-1",
                "-q:v", "3",
                hook_pattern
            ]
            try:
                subprocess.run(cmd_hook, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                hook_frames = sorted([
                    os.path.join(hook_dir, f) for f in os.listdir(hook_dir) if f.endswith(".jpg")
                ])
            except Exception:
                pass

        # 3. Pacing Metrics
        cuts_count = max(0, len(extracted_shots) - 1)
        cuts_per_min = (cuts_count / duration * 60.0) if duration > 0 else 0.0

        # 4. Generate Markdown Visual Audit Report
        report_path = os.path.join(work_dir, "visual_watch_report.md")
        self._generate_report_markdown(
            report_path=report_path,
            video_path=video_path,
            meta=meta,
            hero_frames=hero_frames,
            all_frames=extracted_shots,
            hook_frames=hook_frames,
            cuts_count=cuts_count,
            cuts_per_min=cuts_per_min,
            is_static=is_static
        )

        return VisualInspectionResult(
            video_path=video_path,
            duration_seconds=duration,
            width=meta["width"],
            height=meta["height"],
            hero_frames=hero_frames,
            all_frames=[{"path": p, "name": os.path.basename(p)} for p in extracted_shots],
            hook_frames=hook_frames,
            cuts_count=cuts_count,
            cuts_per_minute=cuts_per_min,
            is_static_footage=is_static,
            report_markdown_path=report_path
        )

    def _generate_report_markdown(
        self,
        report_path: str,
        video_path: str,
        meta: Dict[str, Any],
        hero_frames: List[str],
        all_frames: List[str],
        hook_frames: List[str],
        cuts_count: int,
        cuts_per_min: float,
        is_static: bool
    ) -> None:
        """Writes structured markdown report detailing visual moments and shots."""
        lines = [
            f"# 👁️ Visual Inspection Report: `{os.path.basename(video_path)}`",
            "",
            "## 1. Video Specifications",
            f"- **Resolution:** {meta['width']}x{meta['height']}",
            f"- **Duration:** {meta['duration']:.2f} seconds",
            f"- **Total Shot Cuts:** {cuts_count}",
            f"- **Pacing (Cuts/Min):** {cuts_per_min:.1f}",
            f"- **Footage Classification:** {'⚠️ Static / Screen-captured / Motionless' if is_static else '✅ Dynamic Cinematic Cuts'}",
            "",
            "## 2. Hero Visual Frames (Key Moments)",
            "These keyframes capture the primary scene boundaries:",
            ""
        ]

        for idx, hf in enumerate(hero_frames, 1):
            lines.append(f"- **Keyframe {idx}:** `{hf}`")

        if hook_frames:
            lines.extend([
                "",
                f"## 3. 0-10s Hook Microscope ({len(hook_frames)} frames @ 2fps)",
                "Dense opening review of visual presentation:",
                ""
            ])
            for hf in hook_frames[:6]:
                lines.append(f"- `{hf}`")

        lines.extend([
            "",
            f"## 4. Total Extracted Frames ({len(all_frames)})",
            f"Frames stored in: `{os.path.dirname(report_path)}/frames/`",
            ""
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
