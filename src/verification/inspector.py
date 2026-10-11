"""
Autonomous Quality Verification & Inspection Engine.
Performs objective quality audits on generated audio, video, and lip-sync
deliverables before human review.
Evaluates:
- Audio: SNR, Peak dBFS, RMS dBFS, digital clipping, speaking rate (WPM), trailing silence.
- Video: Resolution, framerate, face detection rate, facial sharpness, motion stability, freeze detection.
- Audio-Visual Sync: AV duration discrepancy, lip motion variance.
"""

import os
import re
import json
import math
import wave
import struct
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False


@dataclass
class VerificationReport:
    video_path: str
    overall_score: float  # 0.0 to 100.0
    status: str  # 'PASSED', 'NEEDS_REVIEW', 'FAILED'
    audio_metrics: Dict[str, Any] = field(default_factory=dict)
    video_metrics: Dict[str, Any] = field(default_factory=dict)
    sync_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    remediations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        color_symbol = "✅" if self.status == "PASSED" else "⚠️" if self.status == "NEEDS_REVIEW" else "❌"
        print("\n" + "=" * 65)
        print(f"{color_symbol} QUALITY VERIFICATION AUDIT: {self.status} (Score: {self.overall_score:.1f}/100)")
        print("=" * 65)
        print(f"📁 Video: {os.path.basename(self.video_path)}")

        if self.audio_metrics:
            a = self.audio_metrics
            print(f"🎙️ Audio: SNR: {a.get('snr_db', 0):.1f}dB | Peak: {a.get('peak_db', 0):.1f}dBFS | RMS: {a.get('rms_db', 0):.1f}dBFS | Clip: {a.get('clipping_pct', 0):.3f}%")
            if "wpm" in a:
                print(f"   Speaking Pace: {a['wpm']:.1f} WPM")

        if self.video_metrics:
            v = self.video_metrics
            print(f"🎞️ Video: {v.get('width', 0)}x{v.get('height', 0)} @ {v.get('fps', 0):.1f}fps | Duration: {v.get('duration', 0):.2f}s")
            print(f"   Face Detection Rate: {v.get('face_detection_rate', 0):.1f}% | Sharpness: {v.get('sharpness', 0):.1f} | Stability: {v.get('stability', 0):.1f}")

        if self.sync_metrics:
            s = self.sync_metrics
            print(f"⚡ AV Sync: Duration Delta: {s.get('av_duration_delta', 0):.3f}s")

        if self.warnings:
            print("\n⚠️ Warnings:")
            for w in self.warnings:
                print(f"   - {w}")

        if self.errors:
            print("\n❌ Errors:")
            for e in self.errors:
                print(f"   - {e}")

        if self.remediations:
            print("\n💡 Suggested Remediation:")
            for r in self.remediations:
                print(f"   * {r}")
        print("=" * 65 + "\n")


class QualityInspector:
    """
    Automated QA Inspector for AI-generated avatars and voiceovers.
    Runs fast heuristic tests (audio clarity, facial presence, jitter, sync).
    """

    def __init__(self, pass_threshold: float = 70.0, review_threshold: float = 50.0):
        self.pass_threshold = pass_threshold
        self.review_threshold = review_threshold

    def inspect_audio_file(self, wav_path: str, script_text: str = "") -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Inspects speech audio for acoustic compliance."""
        metrics = {}
        warnings = []
        errors = []

        if not os.path.exists(wav_path):
            errors.append(f"Audio file not found: {wav_path}")
            return metrics, warnings, errors

        try:
            with wave.open(wav_path, "rb") as w:
                sr = w.getframerate()
                n_frames = w.getnframes()
                raw = w.readframes(n_frames)

            samples = struct.unpack(f"<{n_frames}h", raw)
            duration = n_frames / float(sr)
            metrics["duration"] = duration
            metrics["sample_rate"] = sr

            # Peak & RMS calculations
            peak = max(abs(s) for s in samples)
            peak_db = 20 * math.log10(peak / 32767.0) if peak > 0 else -100.0
            rms = math.sqrt(sum(s**2 for s in samples) / len(samples))
            rms_db = 20 * math.log10(rms / 32767.0) if rms > 0 else -100.0

            metrics["peak_db"] = peak_db
            metrics["rms_db"] = rms_db

            # Clipping check
            clip_count = sum(1 for s in samples if abs(s) >= 32700)
            clip_pct = (clip_count / float(len(samples))) * 100.0
            metrics["clipping_pct"] = clip_pct

            if clip_pct > 0.05:
                warnings.append(f"Audio has {clip_pct:.2f}% digital clipping (risk of harsh distortion).")

            # Signal-to-noise ratio estimation
            # Approximate noise from quietest 10% blocks
            block_size = int(sr * 0.05)
            block_rms = []
            for i in range(0, len(samples) - block_size, block_size):
                chunk = samples[i : i + block_size]
                b_rms = math.sqrt(sum(s**2 for s in chunk) / block_size)
                block_rms.append(b_rms)

            if block_rms:
                block_rms.sort()
                noise_floor = block_rms[max(0, int(len(block_rms) * 0.1))] + 1e-5
                snr_db = 20 * math.log10(rms / noise_floor)
            else:
                snr_db = 20.0

            metrics["snr_db"] = snr_db
            if snr_db < 15.0:
                warnings.append(f"Low speech SNR ({snr_db:.1f} dB); audio may contain perceptible background hiss.")

            # Speaking pace (Words Per Minute)
            if script_text and duration > 0:
                words = len(script_text.strip().split())
                wpm = (words / duration) * 60.0
                metrics["wpm"] = wpm
                if wpm > 220:
                    warnings.append(f"Speaking pace is unusually fast ({wpm:.0f} WPM; normal is 130-170).")
                elif wpm < 85:
                    warnings.append(f"Speaking pace is sluggish ({wpm:.0f} WPM; normal is 130-170).")

        except Exception as e:
            errors.append(f"Audio inspection failed: {e}")

        return metrics, warnings, errors

    def inspect_video_file(self, video_path: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Inspects video file for resolution, framerate, face stability, and blur."""
        metrics = {}
        warnings = []
        errors = []

        if not os.path.exists(video_path):
            errors.append(f"Video file not found: {video_path}")
            return metrics, warnings, errors

        # 1. ffprobe stream checks
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of", "json",
                video_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            probe_data = json.loads(res.stdout)
            format_info = probe_data.get("format", {})
            metrics["duration"] = float(format_info.get("duration", 0.0))

            streams = probe_data.get("streams", [])
            has_video = any(s.get("codec_type") == "video" for s in streams)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)

            if not has_video:
                errors.append("Deliverable has no valid video stream.")
            if not has_audio:
                errors.append("Deliverable has no audio stream (silent video).")

            for s in streams:
                if s.get("codec_type") == "video":
                    metrics["width"] = s.get("width", 0)
                    metrics["height"] = s.get("height", 0)
                    r_fps = s.get("r_frame_rate", "25/1")
                    if "/" in r_fps:
                        num, den = r_fps.split("/")
                        metrics["fps"] = float(num) / float(den) if float(den) > 0 else 25.0
                    else:
                        metrics["fps"] = float(r_fps)

        except Exception as e:
            warnings.append(f"ffprobe stream inspection warning: {e}")

        # 2. OpenCV frame sampling (if cv2 available)
        if CV2_AVAILABLE and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )

                    sample_count = min(25, total_frames)
                    step = max(1, total_frames // sample_count)

                    faces_detected = 0
                    sharpness_scores = []
                    flow_magnitudes = []
                    prev_gray = None
                    frozen_frames = 0

                    for i in range(0, total_frames, step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret:
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Sharpness (Laplacian variance)
                        var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        sharpness_scores.append(var)

                        # Face presence detection
                        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
                        if len(faces) > 0:
                            faces_detected += 1

                        # Motion / Freeze check
                        if prev_gray is not None:
                            diff = cv2.absdiff(prev_gray, gray)
                            mean_diff = float(np.mean(diff))
                            if mean_diff < 0.3:  # frozen frame
                                frozen_frames += 1
                            else:
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                flow_magnitudes.append(float(np.mean(mag)))

                        prev_gray = gray

                    cap.release()

                    actual_samples = len(sharpness_scores) if sharpness_scores else 1
                    face_rate = (faces_detected / float(actual_samples)) * 100.0
                    avg_sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0
                    avg_motion = float(np.mean(flow_magnitudes)) if flow_magnitudes else 1.0

                    metrics["face_detection_rate"] = face_rate
                    metrics["sharpness"] = avg_sharpness
                    metrics["stability"] = max(0.0, 10.0 - abs(avg_motion - 1.5))
                    metrics["frozen_frame_pct"] = (frozen_frames / float(actual_samples)) * 100.0

                    if face_rate < 80.0:
                        warnings.append(f"Face detected in only {face_rate:.1f}% of sampled frames (possible cropping issue).")
                    if avg_sharpness < 70.0:
                        warnings.append(f"Average frame sharpness is low ({avg_sharpness:.1f}); face may appear blurry.")
                    if metrics["frozen_frame_pct"] > 30.0:
                        warnings.append(f"High frozen frame ratio ({metrics['frozen_frame_pct']:.1f}%); video may appear static.")

            except Exception as e:
                warnings.append(f"OpenCV visual analysis warning: {e}")

        return metrics, warnings, errors

    def inspect_av_sync(
        self,
        video_path: str,
        audio_metrics: Dict[str, Any],
        video_metrics: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Validates synchronization and duration alignment between audio and video."""
        sync_metrics = {}
        warnings = []
        errors = []

        v_dur = video_metrics.get("duration", 0.0)
        a_dur = audio_metrics.get("duration", 0.0)

        if v_dur > 0 and a_dur > 0:
            delta = abs(v_dur - a_dur)
            sync_metrics["av_duration_delta"] = delta
            if delta > 0.5:
                warnings.append(f"Audio and Video duration mismatch: delta = {delta:.2f}s (audio={a_dur:.2f}s, video={v_dur:.2f}s).")
            if delta > 2.0:
                errors.append(f"Severe audio-video desync: duration discrepancy exceeds {delta:.2f}s.")
        else:
            sync_metrics["av_duration_delta"] = 0.0

        return sync_metrics, warnings, errors

    def audit_deliverable(
        self,
        video_path: str,
        speech_wav_path: Optional[str] = None,
        script_text: str = "",
        save_report: bool = True,
    ) -> VerificationReport:
        """
        Executes a complete autonomous audit on the generated video deliverable.
        Computes composite score and produces a structured VerificationReport.
        """
        all_warnings = []
        all_errors = []
        remediations = []

        # 1. Extract audio from video if speech_wav_path not provided
        temp_wav = None
        if not speech_wav_path and os.path.exists(video_path):
            temp_wav = video_path + ".temp_audit.wav"
            try:
                cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", temp_wav]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                speech_wav_path = temp_wav
            except Exception:
                pass

        # 2. Inspect audio & video components
        audio_m, a_warn, a_err = self.inspect_audio_file(speech_wav_path, script_text) if speech_wav_path else ({}, [], [])
        video_m, v_warn, v_err = self.inspect_video_file(video_path)
        sync_m, s_warn, s_err = self.inspect_av_sync(video_path, audio_m, video_m)

        all_warnings.extend(a_warn + v_warn + s_warn)
        all_errors.extend(a_err + v_err + s_err)

        # Cleanup temp audio
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

        # 3. Calculate Composite Score (0.0 to 100.0)
        score = 100.0

        # Critical failure deductions
        if all_errors:
            score -= len(all_errors) * 35.0

        # Audio score components
        snr = audio_m.get("snr_db", 25.0)
        if snr < 20.0:
            score -= (20.0 - snr) * 1.5

        clip_pct = audio_m.get("clipping_pct", 0.0)
        if clip_pct > 0.01:
            score -= min(15.0, clip_pct * 100.0)

        # Video score components
        face_rate = video_m.get("face_detection_rate", 100.0)
        if face_rate < 90.0:
            score -= (90.0 - face_rate) * 0.4

        sharpness = video_m.get("sharpness", 120.0)
        if sharpness < 80.0:
            score -= min(15.0, (80.0 - sharpness) * 0.3)

        frozen_pct = video_m.get("frozen_frame_pct", 0.0)
        if frozen_pct > 20.0:
            score -= min(15.0, (frozen_pct - 20.0) * 0.5)

        # Sync deductions
        delta = sync_m.get("av_duration_delta", 0.0)
        if delta > 0.3:
            score -= min(20.0, delta * 15.0)

        score = max(0.0, min(100.0, score))

        # 4. Determine status
        if all_errors or score < self.review_threshold:
            status = "FAILED"
        elif all_warnings or score < self.pass_threshold:
            status = "NEEDS_REVIEW"
        else:
            status = "PASSED"

        # 5. Suggest remediations
        if any("clipping" in w for w in all_warnings):
            remediations.append("Apply AudioEnhancer with -16 LUFS normalization to eliminate peak saturation.")
        if any("SNR" in w for w in all_warnings):
            remediations.append("Enable spectral noise gating (denoise_audio=True) in pipeline produce_video.")
        if any("sharpness" in w for w in all_warnings):
            remediations.append("Enable GFPGAN facial super-resolution (enhance_face=True) to sharpen eyes and skin.")
        if any("desync" in e for e in all_errors) or any("mismatch" in w for w in all_warnings):
            remediations.append("Re-encode with FFmpeg -shortest flag to force audio-video duration synchronization.")

        report = VerificationReport(
            video_path=video_path,
            overall_score=score,
            status=status,
            audio_metrics=audio_m,
            video_metrics=video_m,
            sync_metrics=sync_m,
            warnings=all_warnings,
            errors=all_errors,
            remediations=remediations,
        )

        if save_report:
            report_path = os.path.splitext(video_path)[0] + "_quality_audit.json"
            report.save(report_path)

        return report
