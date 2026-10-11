"""
Unit tests for Gemini Managed Agent Supervisor and Autonomous QA Diagnostic Loop.
Validates prompt decomposition, audit scorecard diagnosis, API mocking, and fallbacks.
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestGeminiAgentSupervisor(unittest.TestCase):
    def test_supervisor_initialization_defaults(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        with patch.dict(os.environ, {}, clear=True):
            supervisor = GeminiAgentSupervisor()
            self.assertIsNone(supervisor.api_key)
            self.assertEqual(supervisor.agent_tag, "antigravity-preview-05-2026")
            self.assertEqual(supervisor.fallback_model, "gemini-3.5-flash")
            self.assertIsNone(supervisor._client)

    def test_supervisor_initialization_with_api_key(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        with patch("google.genai.Client", return_value=MagicMock()) as mock_client_cls:
            supervisor = GeminiAgentSupervisor(api_key="explicit-secret-key")
            self.assertEqual(supervisor.api_key, "explicit-secret-key")
            self.assertEqual(supervisor.agent_tag, "antigravity-preview-05-2026")
            self.assertEqual(supervisor.fallback_model, "gemini-3.5-flash")
            self.assertIsNotNone(supervisor._client)

    def test_offline_prompt_expansion(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        with patch.dict(os.environ, {}, clear=True):
            supervisor = GeminiAgentSupervisor()
            brief = "female founder in modern glass boardroom delivering quarterly update"
            expanded = supervisor.expand_creative_prompt(brief)

            self.assertIsInstance(expanded, dict)
            self.assertIn("positive_prompt", expanded)
            self.assertIn("negative_prompt", expanded)

            expected_positive = (
                f"Cinematic medium shot of {brief}, dramatic studio lighting, soft key light, "
                f"subtle natural head and torso gestures, crisp 4k resolution, photorealistic, 24fps"
            )
            expected_negative = (
                "worst quality, inconsistent motion, blurry, jittery, distorted, morphing, cartoon"
            )
            self.assertEqual(expanded["positive_prompt"], expected_positive)
            self.assertEqual(expanded["negative_prompt"], expected_negative)

    def test_mocked_api_prompt_expansion_success(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor(api_key="fake-test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "positive_prompt": "Ultra realistic cinematic portrait of a lead engineer on Mars colony",
            "negative_prompt": "unrealistic shadows, oversaturated, low res"
        })
        mock_client.models.generate_content.return_value = mock_response
        supervisor._client = mock_client

        result = supervisor.expand_creative_prompt("lead engineer on Mars colony")
        self.assertEqual(
            result["positive_prompt"],
            "Ultra realistic cinematic portrait of a lead engineer on Mars colony"
        )
        self.assertEqual(
            result["negative_prompt"],
            "unrealistic shadows, oversaturated, low res"
        )

    def test_mocked_api_prompt_expansion_markdown_codeblock(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor(api_key="fake-test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```json\n" + json.dumps({
            "positive_prompt": "Cinematic shot with markdown fences",
            "negative_prompt": "grainy, artifacts"
        }) + "\n```"
        mock_client.models.generate_content.return_value = mock_response
        supervisor._client = mock_client

        result = supervisor.expand_creative_prompt("shot with markdown fences")
        self.assertEqual(result["positive_prompt"], "Cinematic shot with markdown fences")
        self.assertEqual(result["negative_prompt"], "grainy, artifacts")

    def test_mocked_api_prompt_expansion_api_failure_fallback(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor(api_key="fake-test-key")
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API rate limit exceeded")
        supervisor._client = mock_client

        brief = "doctor explaining MRI scan"
        result = supervisor.expand_creative_prompt(brief)
        self.assertIn("Cinematic medium shot of doctor explaining MRI scan", result["positive_prompt"])
        self.assertIn("worst quality", result["negative_prompt"])

    def test_diagnose_audit_scorecard_passing(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        passing_report = {
            "audio_metrics": {"snr_db": 22.5, "clipping_ratio": 0.002},
            "video_metrics": {"face_detected_ratio": 0.95, "laplacian_sharpness": 85.0},
            "sync_metrics": {"av_duration_delta_sec": 0.08}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(passing_report)

        self.assertIsInstance(diagnosis, dict)
        self.assertEqual(diagnosis["audio_action"], "Audio verified clean.")
        self.assertEqual(diagnosis["video_action"], "Video frames verified stable.")
        self.assertEqual(diagnosis["sync_action"], "AV sync within tolerance.")

    def test_diagnose_audit_scorecard_audio_remediation_snr(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        low_snr_report = {
            "audio_metrics": {"snr_db": 11.2, "clipping_ratio": 0.000},
            "video_metrics": {"face_detected_ratio": 0.92, "laplacian_sharpness": 90.0},
            "sync_metrics": {"av_duration_delta_sec": 0.05}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(low_snr_report)
        self.assertEqual(
            diagnosis["audio_action"],
            "Increase spectral gating to 0.85 and apply hard limiter at -1.0 dBFS."
        )

    def test_diagnose_audit_scorecard_audio_remediation_clipping(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        clipped_report = {
            "audio_metrics": {"snr_db": 20.0, "clipping_ratio": 0.025},
            "video_metrics": {"face_detected_ratio": 0.92, "laplacian_sharpness": 90.0},
            "sync_metrics": {"av_duration_delta_sec": 0.05}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(clipped_report)
        self.assertEqual(
            diagnosis["audio_action"],
            "Increase spectral gating to 0.85 and apply hard limiter at -1.0 dBFS."
        )

    def test_diagnose_audit_scorecard_video_remediation_face_ratio(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        low_face_report = {
            "audio_metrics": {"snr_db": 22.0, "clipping_ratio": 0.000},
            "video_metrics": {"face_detected_ratio": 0.65, "laplacian_sharpness": 85.0},
            "sync_metrics": {"av_duration_delta_sec": 0.05}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(low_face_report)
        self.assertEqual(
            diagnosis["video_action"],
            "Retarget bounding box or fallback from LatentSync to LivePortrait photo reenactment."
        )

    def test_diagnose_audit_scorecard_video_remediation_sharpness(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        blurry_report = {
            "audio_metrics": {"snr_db": 22.0, "clipping_ratio": 0.000},
            "video_metrics": {"face_detected_ratio": 0.88, "laplacian_sharpness": 52.0},
            "sync_metrics": {"av_duration_delta_sec": 0.05}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(blurry_report)
        self.assertEqual(
            diagnosis["video_action"],
            "Enable GFPGAN facial restoration and RIFE 60fps interpolation."
        )

    def test_diagnose_audit_scorecard_sync_remediation(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        desync_report = {
            "audio_metrics": {"snr_db": 22.0, "clipping_ratio": 0.000},
            "video_metrics": {"face_detected_ratio": 0.88, "laplacian_sharpness": 75.0},
            "sync_metrics": {"av_duration_delta_sec": 0.42}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(desync_report)
        self.assertEqual(
            diagnosis["sync_action"],
            "Re-encode audio with FFmpeg asetpts/atempo to match video frame count exactly."
        )

    def test_diagnose_audit_scorecard_tolerates_nested_inspector_keys(self):
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        supervisor = GeminiAgentSupervisor()
        inspector_style_report = {
            "audio_metrics": {"snr_db": 25.0, "clipping_pct": 0.005},
            "video_metrics": {"face_detection_rate": 95.0, "sharpness": 80.0},
            "sync_metrics": {"av_duration_delta": 0.02}
        }
        diagnosis = supervisor.diagnose_audit_scorecard(inspector_style_report)
        self.assertEqual(diagnosis["audio_action"], "Audio verified clean.")
        self.assertEqual(diagnosis["video_action"], "Video frames verified stable.")
        self.assertEqual(diagnosis["sync_action"], "AV sync within tolerance.")

    def test_src_orchestration_package_exports(self):
        import src.orchestration as orchestration
        from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor

        self.assertTrue(hasattr(orchestration, "GeminiAgentSupervisor"))
        self.assertIs(orchestration.GeminiAgentSupervisor, GeminiAgentSupervisor)
        self.assertIn("GeminiAgentSupervisor", orchestration.__all__)


if __name__ == "__main__":
    unittest.main()
