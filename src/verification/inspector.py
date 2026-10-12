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
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False



@dataclass
class VerificationReport:
    video_path: str
    overall_score: float  # 0.0 to 100.0
    status: str  # 'PASSED', 'NEEDS_REVIEW', 'FAILED'
    audio_metrics: Dict[str, Any] = field(default_factory=dict)
    video_metrics: Dict[str, Any] = field(default_factory=dict)
    sync_metrics: Dict[str, Any] = field(default_factory=dict)
    visual_watch_metrics: Dict[str, Any] = field(default_factory=dict)
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

        if self.visual_watch_metrics:
            w = self.visual_watch_metrics
            print(f"👁️ Visual Watch: Cuts: {w.get('cuts_count', 0)} ({w.get('cuts_per_minute', 0):.1f} cuts/min) | Static: {w.get('is_static_footage', False)}")
            if w.get('hero_frames'):
                print(f"   Hero Keyframes: {len(w['hero_frames'])} frames extracted")

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

        # 2. Frame sampling (via OpenCV or FFmpeg rawvideo fallback)
        frames = []
        if CV2_AVAILABLE and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    sample_count = min(25, total_frames)
                    step = max(1, total_frames // sample_count)
                    for i in range(0, total_frames, step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    cap.release()
            except Exception as e:
                warnings.append(f"OpenCV visual analysis warning: {e}")

        # Fallback to FFmpeg frame extraction if cv2 is unavailable or extracted 0 frames
        if not frames and os.path.exists(video_path):
            try:
                cmd = [
                    "ffmpeg", "-v", "error",
                    "-i", video_path,
                    "-vf", "fps=5,scale=256:256",
                    "-f", "rawvideo",
                    "-pix_fmt", "gray",
                    "-",
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
                if res.stdout and np is not None:
                    raw_arr = np.frombuffer(res.stdout, dtype=np.uint8).reshape(-1, 256, 256)
                    frames = [raw_arr[idx] for idx in range(len(raw_arr))]
            except Exception as e:
                warnings.append(f"FFmpeg frame sampling notice: {e}")

        # Process sampled frames
        if frames and np is not None:
            try:
                face_cascade = None
                if CV2_AVAILABLE:
                    try:
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                        )
                    except Exception:
                        face_cascade = None

                sharpness_scores = []
                flow_magnitudes = []
                mouth_deltas = []
                head_deltas = []
                faces_detected = 0
                frozen_frames = 0
                prev_gray = None

                for gray in frames:
                    fh, fw = gray.shape[:2]

                    # Sharpness calculation
                    if CV2_AVAILABLE:
                        var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    else:
                        lap = gray[1:-1, 1:-1].astype(float) * 4 - gray[:-2, 1:-1].astype(float) - gray[2:, 1:-1].astype(float) - gray[1:-1, :-2].astype(float) - gray[1:-1, 2:].astype(float)
                        var = float(np.var(lap))
                    sharpness_scores.append(var)

                    # Face presence
                    face_box = None
                    if face_cascade is not None:
                        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
                        if len(faces) > 0:
                            faces_detected += 1
                            face_box = faces[0]
                    else:
                        # Heuristic center portrait presence
                        center_crop = gray[int(fh * 0.2) : int(fh * 0.6), int(fw * 0.3) : int(fw * 0.7)]
                        if float(np.var(center_crop)) > 15.0:
                            faces_detected += 1
                            face_box = (int(fw * 0.25), int(fh * 0.15), int(fw * 0.5), int(fh * 0.6))

                    # Motion and Lip-motion analysis
                    if prev_gray is not None:
                        diff = np.abs(gray.astype(float) - prev_gray.astype(float))
                        mean_diff = float(np.mean(diff))
                        if mean_diff < 0.25:
                            frozen_frames += 1
                        else:
                            flow_magnitudes.append(mean_diff)

                        # Determine mouth and upper-face regions
                        if face_box is not None:
                            fx, fy, fw_b, fh_b = face_box
                            m_y1, m_y2 = max(0, fy + int(fh_b * 0.60)), min(fh, fy + fh_b)
                            m_x1, m_x2 = max(0, fx + int(fw_b * 0.20)), min(fw, fx + int(fw_b * 0.80))
                            h_y1, h_y2 = max(0, fy + int(fh_b * 0.15)), min(fh, fy + int(fh_b * 0.45))
                            h_x1, h_x2 = max(0, fx + int(fw_b * 0.20)), min(fw, fx + int(fw_b * 0.80))
                        else:
                            m_y1, m_y2 = int(fh * 0.55), int(fh * 0.78)
                            m_x1, m_x2 = int(fw * 0.35), int(fw * 0.65)
                            h_y1, h_y2 = int(fh * 0.20), int(fh * 0.45)
                            h_x1, h_x2 = int(fw * 0.35), int(fw * 0.65)

                        mouth_patch = diff[m_y1:m_y2, m_x1:m_x2]
                        head_patch = diff[h_y1:h_y2, h_x1:h_x2]

                        m_val = float(np.mean(mouth_patch)) if mouth_patch.size > 0 else 0.0
                        h_val = float(np.mean(head_patch)) if head_patch.size > 0 else 0.0
                        mouth_deltas.append(m_val)
                        head_deltas.append(h_val)

                    prev_gray = gray

                actual_samples = len(sharpness_scores) if sharpness_scores else 1
                face_rate = (faces_detected / float(actual_samples)) * 100.0
                avg_sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0
                avg_motion = float(np.mean(flow_magnitudes)) if flow_magnitudes else 1.0
                avg_lip_motion = float(np.mean(mouth_deltas)) if mouth_deltas else 0.0

                metrics["face_detection_rate"] = face_rate
                metrics["sharpness"] = avg_sharpness
                metrics["stability"] = max(0.0, 10.0 - abs(avg_motion - 1.5))
                metrics["frozen_frame_pct"] = (frozen_frames / float(actual_samples)) * 100.0
                metrics["lip_motion_score"] = avg_lip_motion
                metrics["lip_motion_detected"] = bool(avg_lip_motion >= 1.0)

                if face_rate < 80.0:
                    warnings.append(f"Face detected in only {face_rate:.1f}% of sampled frames (possible cropping issue).")
                if avg_sharpness < 70.0:
                    warnings.append(f"Average frame sharpness is low ({avg_sharpness:.1f}); face may appear blurry.")
                if metrics["frozen_frame_pct"] > 30.0:
                    warnings.append(f"High frozen frame ratio ({metrics['frozen_frame_pct']:.1f}%); video may appear static.")
                if not metrics["lip_motion_detected"]:
                    warnings.append(f"Low lip/mouth motion detected ({avg_lip_motion:.2f}); avatar mouth appears motionless.")

            except Exception as e:
                warnings.append(f"Frame visual analysis notice: {e}")

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
        run_visual_watch: bool = False,
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

        # 2b. Lip motion realism gate during speech playback
        is_speech_active = audio_m.get("duration", 0.0) >= 1.5
        lip_detected = video_m.get("lip_motion_detected", False)
        lip_score = video_m.get("lip_motion_score", 0.0)

        if is_speech_active and not lip_detected:
            lip_err = (
                f"LIP_MOTION_ABSENT: Active speech is playing ({audio_m.get('duration', 0.0):.1f}s) "
                f"but mouth/lip movement is undetectable (lip motion score: {lip_score:.2f} < 1.00). "
                "Neural lip-sync required via LatentSync on Google Colab."
            )
            all_errors.append(lip_err)
            remediations.append(
                "Mouth movement is absent while speech is playing. Execute 03_fullbody_latentsync_studio.ipynb "
                "on Google Colab GPU (Tesla T4) to synthesize neural LatentSync lip-sync."
            )

        # 2c. Optional Visual Watch analysis (scene-changes, hero frames, hook microscope)
        visual_watch_m = {}
        if run_visual_watch and os.path.exists(video_path):
            try:
                from .visual_watcher import VisualWatcher
                watcher = VisualWatcher()
                v_res = watcher.watch_video(video_path)
                visual_watch_m = v_res.to_dict()
                if v_res.is_static_footage:
                    all_warnings.append("Visual Watcher: Video footage classified as static / single continuous shot without scene progression.")
            except Exception as e:
                all_warnings.append(f"Visual Watcher notice: {e}")

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
            visual_watch_metrics=visual_watch_m,
            warnings=all_warnings,
            errors=all_errors,
            remediations=remediations,
        )

        if save_report:
            report_path = os.path.splitext(video_path)[0] + "_quality_audit.json"
            report.save(report_path)

        return report
