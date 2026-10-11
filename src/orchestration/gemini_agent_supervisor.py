"""
Gemini Managed Agent Supervisor & Autonomous Self-Annealing QA Diagnostic Loop.
Decomposes user prompts into 5-part cinematic formulas for Diffusion Transformers,
and diagnoses QualityInspector audit scorecards to generate actionable remediations.
"""

import os
import re
import json
from typing import Optional, Dict, Any


class GeminiAgentSupervisor:
    """
    Supervisor layer integrating Gemini for creative prompt expansion
    and autonomous quality assurance scorecard remediation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.agent_tag = "antigravity-preview-05-2026"
        self.fallback_model = "gemini-3.5-flash"
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def expand_creative_prompt(self, brief: str) -> Dict[str, str]:
        """
        Expands user brief into a 5-part cinematic formula for Diffusion Transformers
        ([Subject] + [Setting & Lighting] + [Camera Motion] + [Atmosphere] + [Technical Fidelity]).
        Calls Gemini API if client available, with deterministic offline fallback.
        """
        fallback = {
            "positive_prompt": (
                f"Cinematic medium shot of {brief}, dramatic studio lighting, soft key light, "
                f"subtle natural head and torso gestures, crisp 4k resolution, photorealistic, 24fps"
            ),
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted, morphing, cartoon",
        }

        if self._client is not None:
            try:
                prompt_instruction = (
                    "You are an expert AI video director. Expand the provided brief into a 5-part cinematic prompt "
                    "for Diffusion Transformers ([Subject] + [Setting & Lighting] + [Camera Motion] + [Atmosphere] + [Technical Fidelity]). "
                    "Return ONLY a JSON object with 'positive_prompt' and 'negative_prompt'."
                )
                if hasattr(self._client, "models") and hasattr(self._client.models, "generate_content"):
                    response = self._client.models.generate_content(
                        model=self.fallback_model,
                        contents=f"{prompt_instruction}\n\nBrief: {brief}",
                    )
                elif hasattr(self._client, "generate_content"):
                    response = self._client.generate_content(
                        f"{prompt_instruction}\n\nBrief: {brief}"
                    )
                else:
                    response = self._client(f"{prompt_instruction}\n\nBrief: {brief}")

                text = getattr(response, "text", None)
                if text is None:
                    if isinstance(response, str):
                        text = response
                    elif isinstance(response, dict):
                        text = json.dumps(response)
                    else:
                        text = str(response)

                text = text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                data = json.loads(text)
                if isinstance(data, dict) and "positive_prompt" in data and "negative_prompt" in data:
                    return {
                        "positive_prompt": str(data["positive_prompt"]),
                        "negative_prompt": str(data["negative_prompt"]),
                    }
            except Exception:
                pass

        return fallback

    def diagnose_audit_scorecard(self, report_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Inspects QualityInspector scorecard metrics (audio, video, sync)
        and outputs autonomous remediation instructions for self-annealing pipelines.
        """
        audio_metrics = report_dict.get("audio_metrics")
        if not isinstance(audio_metrics, dict):
            audio_metrics = report_dict

        video_metrics = report_dict.get("video_metrics")
        if not isinstance(video_metrics, dict):
            video_metrics = report_dict

        sync_metrics = report_dict.get("sync_metrics")
        if not isinstance(sync_metrics, dict):
            sync_metrics = report_dict

        # Audio metrics evaluation:
        # snr_db < 15.0 or clipping_ratio > 0.01
        snr_raw = audio_metrics.get("snr_db")
        snr_db = float(snr_raw) if snr_raw is not None else 20.0
        clipping_ratio = audio_metrics.get("clipping_ratio")
        if clipping_ratio is None and "clipping_pct" in audio_metrics:
            clipping_ratio = float(audio_metrics["clipping_pct"]) / 100.0
        if clipping_ratio is None:
            clipping_ratio = 0.0
        else:
            clipping_ratio = float(clipping_ratio)

        if snr_db < 15.0 or clipping_ratio > 0.01:
            audio_action = "Increase spectral gating to 0.85 and apply hard limiter at -1.0 dBFS."
        else:
            audio_action = "Audio verified clean."

        # Video metrics evaluation:
        # face_detected_ratio < 0.70: "Retarget bounding box or fallback from LatentSync to LivePortrait photo reenactment."
        # elif laplacian_sharpness < 60.0: "Enable GFPGAN facial restoration and RIFE 60fps interpolation."
        # else: "Video frames verified stable."
        face_detected_ratio = video_metrics.get("face_detected_ratio")
        if face_detected_ratio is None and "face_detection_rate" in video_metrics:
            face_detected_ratio = float(video_metrics["face_detection_rate"]) / 100.0
        if face_detected_ratio is None:
            face_detected_ratio = 1.0
        else:
            face_detected_ratio = float(face_detected_ratio)

        laplacian_sharpness = video_metrics.get("laplacian_sharpness")
        if laplacian_sharpness is None and "sharpness" in video_metrics:
            laplacian_sharpness = float(video_metrics["sharpness"])
        if laplacian_sharpness is None:
            laplacian_sharpness = 100.0
        else:
            laplacian_sharpness = float(laplacian_sharpness)

        if face_detected_ratio < 0.70:
            video_action = "Retarget bounding box or fallback from LatentSync to LivePortrait photo reenactment."
        elif laplacian_sharpness < 60.0:
            video_action = "Enable GFPGAN facial restoration and RIFE 60fps interpolation."
        else:
            video_action = "Video frames verified stable."

        # Sync metrics evaluation:
        # av_duration_delta_sec > 0.30: "Re-encode audio with FFmpeg asetpts/atempo to match video frame count exactly."
        # else: "AV sync within tolerance."
        av_duration_delta_sec = sync_metrics.get("av_duration_delta_sec")
        if av_duration_delta_sec is None and "av_duration_delta" in sync_metrics:
            av_duration_delta_sec = float(sync_metrics["av_duration_delta"])
        if av_duration_delta_sec is None:
            av_duration_delta_sec = 0.0
        else:
            av_duration_delta_sec = float(av_duration_delta_sec)

        if av_duration_delta_sec > 0.30:
            sync_action = "Re-encode audio with FFmpeg asetpts/atempo to match video frame count exactly."
        else:
            sync_action = "AV sync within tolerance."

        return {
            "audio_action": audio_action,
            "video_action": video_action,
            "sync_action": sync_action,
        }
