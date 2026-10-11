"""
Unit tests for VideoProductionPipeline modes and engine integrations (Task 5).
Tests:
- KokoroEngine integration (Zero-VRAM narration alternative to F5-TTS).
- GenerativeVideoEngine integration (LTX-Video / Wan2.1 generative scene diffusion).
- MotionInterpolator integration (RIFE 60fps frame interpolation flag).
- GeminiAgentSupervisor integration (Creative prompt expansion and QA diagnostic loop).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import VideoProductionPipeline
from src.audio.kokoro_engine import KokoroEngine
from src.video.generative_engine import GenerativeVideoEngine
from src.enhancers.motion_interpolator import MotionInterpolator
from src.orchestration.gemini_agent_supervisor import GeminiAgentSupervisor


class TestPipelineModes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.tmp_dir.name, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_pipeline_initialization_attributes(self):
        """Pipeline must initialize all new engines without requiring cloud credentials."""
        pipeline = VideoProductionPipeline(use_drive=False)
        self.assertIsInstance(pipeline.kokoro, KokoroEngine)
        self.assertIsInstance(pipeline.generative, GenerativeVideoEngine)
        self.assertIsInstance(pipeline.interpolator, MotionInterpolator)
        self.assertIsInstance(pipeline.supervisor, GeminiAgentSupervisor)

    def test_produce_video_f5_missing_ref_audio_raises_value_error(self):
        """F5-TTS requires valid reference audio and raises ValueError if missing."""
        pipeline = VideoProductionPipeline(use_drive=False)
        with self.assertRaises(ValueError) as ctx:
            pipeline.produce_video(
                script_text="Testing F5 audio requirement",
                tts_engine="f5",
                ref_audio_path=None,
            )
        self.assertIn("reference audio", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            pipeline.produce_video(
                script_text="Testing F5 audio non-existent path",
                tts_engine="f5",
                ref_audio_path="/non_existent_folder/missing_voice.wav",
            )
        self.assertIn("reference audio", str(ctx2.exception).lower())

    def test_produce_video_unknown_tts_engine_raises_value_error(self):
        """Unsupported tts_engine values must raise ValueError."""
        pipeline = VideoProductionPipeline(use_drive=False)
        with self.assertRaises(ValueError) as ctx:
            pipeline.produce_video(
                script_text="Testing invalid TTS engine",
                tts_engine="nonexistent_engine",
            )
        self.assertIn("unknown tts_engine", str(ctx.exception).lower())

    @patch("src.pipeline.flush_vram")
    @patch("src.pipeline.copy_output_to_drive")
    def test_produce_video_kokoro_narration_without_ref_audio(self, mock_copy, mock_flush):
        """KokoroEngine provides narration without requiring ref_audio_path."""
        pipeline = VideoProductionPipeline(use_drive=False)
        pipeline.workspace["outputs"] = self.output_dir

        dummy_audio = os.path.join(self.output_dir, "speech.wav")
        with open(dummy_audio, "wb") as f:
            f.write(b"RIFFdummyWAVdata")

        dummy_video = os.path.join(self.output_dir, "scene.mp4")
        with open(dummy_video, "wb") as f:
            f.write(b"MOCK_VIDEO_DATA")

        mock_copy.side_effect = lambda path, *args: path

        with patch.object(pipeline.kokoro, "synthesize", return_value=dummy_audio) as mock_synth, \
             patch.object(pipeline.kokoro, "unload") as mock_kokoro_unload, \
             patch.object(pipeline.voice_cloner, "unload") as mock_voice_unload, \
             patch.object(pipeline.generative, "generate_scene", return_value=dummy_video) as mock_gen_scene:

            result = pipeline.produce_video(
                script_text="Welcome to the future of AI video generation.",
                tts_engine="kokoro",
                ref_audio_path=None,
                engine="generative",
                denoise_audio=False,
                auto_verify=False,
            )

            mock_synth.assert_called_once()
            call_kwargs = mock_synth.call_args[1] if mock_synth.call_args[1] else {}
            call_args = mock_synth.call_args[0]
            # Verify script_text is passed to Kokoro synthesize
            if "text" in call_kwargs:
                self.assertEqual(call_kwargs["text"], "Welcome to the future of AI video generation.")
            else:
                self.assertEqual(call_args[0], "Welcome to the future of AI video generation.")

            # Both voice engines must be unloaded during VRAM fence
            mock_kokoro_unload.assert_called_once()
            mock_voice_unload.assert_called_once()
            mock_flush.assert_called()
            mock_gen_scene.assert_called_once()
            self.assertTrue(result.endswith(".mp4") or "scene" in result)

    @patch("src.pipeline.flush_vram")
    @patch("src.pipeline.copy_output_to_drive")
    def test_produce_video_with_interpolate_60fps(self, mock_copy, mock_flush):
        """Setting interpolate_60fps=True routes the video through MotionInterpolator."""
        pipeline = VideoProductionPipeline(use_drive=False)
        pipeline.workspace["outputs"] = self.output_dir

        dummy_audio = os.path.join(self.output_dir, "speech.wav")
        with open(dummy_audio, "wb") as f:
            f.write(b"RIFFdummyWAVdata")

        dummy_video = os.path.join(self.output_dir, "scene.mp4")
        interp_video = os.path.join(self.output_dir, "interp_scene.mp4")
        with open(dummy_video, "wb") as f:
            f.write(b"MOCK_VIDEO_DATA")
        with open(interp_video, "wb") as f:
            f.write(b"MOCK_INTERPOLATED_VIDEO_DATA")

        mock_copy.side_effect = lambda path, *args: path

        with patch.object(pipeline.kokoro, "synthesize", return_value=dummy_audio), \
             patch.object(pipeline.generative, "generate_scene", return_value=dummy_video), \
             patch.object(pipeline.interpolator, "interpolate_60fps", return_value=interp_video) as mock_interp:

            result = pipeline.produce_video(
                script_text="Interpolation 60fps testing.",
                tts_engine="kokoro",
                engine="generative",
                interpolate_60fps=True,
                denoise_audio=False,
                auto_verify=False,
            )

            mock_interp.assert_called_once()
            input_arg = mock_interp.call_args[0][0] if mock_interp.call_args[0] else mock_interp.call_args[1].get("video_path")
            self.assertEqual(input_arg, dummy_video)
            self.assertEqual(result, interp_video)

    @patch("src.pipeline.flush_vram")
    @patch("src.pipeline.copy_output_to_drive")
    def test_produce_video_with_engine_generative_and_ltx_video(self, mock_copy, mock_flush):
        """Engine options 'generative' and 'ltx-video' invoke GenerativeVideoEngine."""
        pipeline = VideoProductionPipeline(use_drive=False)
        pipeline.workspace["outputs"] = self.output_dir

        dummy_audio = os.path.join(self.output_dir, "speech.wav")
        with open(dummy_audio, "wb") as f:
            f.write(b"RIFFdummyWAVdata")

        dummy_video = os.path.join(self.output_dir, "gen.mp4")
        with open(dummy_video, "wb") as f:
            f.write(b"MOCK_GEN_VIDEO")

        mock_copy.side_effect = lambda path, *args: path

        with patch.object(pipeline.kokoro, "synthesize", return_value=dummy_audio), \
             patch.object(pipeline.supervisor, "expand_creative_prompt", return_value={
                 "positive_prompt": "Cinematic shot of neon cyber laboratory, 4k",
                 "negative_prompt": "blurry",
             }) as mock_expand, \
             patch.object(pipeline.generative, "generate_scene", return_value=dummy_video) as mock_gen_scene:

            # Test 1: engine="generative"
            pipeline.produce_video(
                script_text="A high-tech lab tour.",
                tts_engine="kokoro",
                engine="generative",
                denoise_audio=False,
                auto_verify=False,
            )
            mock_expand.assert_called_with("A high-tech lab tour.")
            mock_gen_scene.assert_called_with(
                prompt="Cinematic shot of neon cyber laboratory, 4k",
                output_path=mock_gen_scene.call_args[1]["output_path"],
                negative_prompt="blurry",
            )

            # Test 2: engine="ltx-video"
            mock_gen_scene.reset_mock()
            mock_expand.reset_mock()
            pipeline.produce_video(
                script_text="A high-tech lab tour 2.",
                tts_engine="kokoro",
                engine="ltx-video",
                denoise_audio=False,
                auto_verify=False,
            )
            mock_expand.assert_called_with("A high-tech lab tour 2.")
            mock_gen_scene.assert_called_once()

    @patch("src.pipeline.flush_vram")
    @patch("src.pipeline.copy_output_to_drive")
    def test_produce_video_generate_scene_prompt_avatar_integration(self, mock_copy, mock_flush):
        """When generate_scene_prompt is passed with an avatar engine (latentsync), it generates the base scene video."""
        pipeline = VideoProductionPipeline(use_drive=False)
        pipeline.workspace["outputs"] = self.output_dir

        dummy_audio = os.path.join(self.output_dir, "speech.wav")
        with open(dummy_audio, "wb") as f:
            f.write(b"RIFFdummyWAVdata")

        dummy_scene = os.path.join(self.output_dir, "base_scene.mp4")
        dummy_avatar_video = os.path.join(self.output_dir, "avatar_synced.mp4")
        with open(dummy_scene, "wb") as f:
            f.write(b"MOCK_BASE_SCENE")
        with open(dummy_avatar_video, "wb") as f:
            f.write(b"MOCK_AVATAR_VIDEO")

        mock_copy.side_effect = lambda path, *args: path

        with patch.object(pipeline.kokoro, "synthesize", return_value=dummy_audio), \
             patch.object(pipeline.supervisor, "expand_creative_prompt", return_value={
                 "positive_prompt": "Cinematic modern executive boardroom, warm sunlight",
                 "negative_prompt": "cartoon",
             }) as mock_expand, \
             patch.object(pipeline.generative, "generate_scene", return_value=dummy_scene) as mock_gen_scene, \
             patch.object(pipeline.latentsync, "sync_lips", return_value=dummy_avatar_video) as mock_sync:

            result = pipeline.produce_video(
                script_text="Executive quarterly summary.",
                tts_engine="kokoro",
                generate_scene_prompt="Executive boardroom setting",
                engine="latentsync",
                denoise_audio=False,
                auto_verify=False,
            )

            mock_expand.assert_called_with("Executive boardroom setting")
            mock_gen_scene.assert_called_once()
            mock_sync.assert_called_once()
            # Ensure the generated scene became the base video for latentsync
            sync_kwargs = mock_sync.call_args[1] if mock_sync.call_args[1] else {}
            sync_args = mock_sync.call_args[0]
            video_input = sync_kwargs.get("video_path") or (sync_args[0] if sync_args else None)
            self.assertEqual(video_input, dummy_scene)
            self.assertEqual(result, dummy_avatar_video)

    @patch("src.pipeline.flush_vram")
    @patch("src.pipeline.copy_output_to_drive")
    def test_produce_video_auto_verify_attaches_supervisor_scorecard(self, mock_copy, mock_flush):
        """When auto_verify=True, QualityInspector audit and supervisor diagnosis are invoked."""
        pipeline = VideoProductionPipeline(use_drive=False)
        pipeline.workspace["outputs"] = self.output_dir

        dummy_audio = os.path.join(self.output_dir, "speech.wav")
        dummy_video = os.path.join(self.output_dir, "video.mp4")
        with open(dummy_audio, "wb") as f:
            f.write(b"RIFFdummyWAVdata")
        with open(dummy_video, "wb") as f:
            f.write(b"MOCK_VIDEO_DATA")

        mock_copy.side_effect = lambda path, *args: path

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "overall_score": 92.5,
            "status": "PASSED",
            "audio_metrics": {"snr_db": 22.0, "clipping_ratio": 0.0},
            "video_metrics": {"face_detected_ratio": 0.95, "laplacian_sharpness": 110.0},
            "sync_metrics": {"av_duration_delta_sec": 0.05},
        }

        with patch.object(pipeline.kokoro, "synthesize", return_value=dummy_audio), \
             patch.object(pipeline.generative, "generate_scene", return_value=dummy_video), \
             patch.object(pipeline.inspector, "audit_deliverable", return_value=mock_report) as mock_audit, \
             patch.object(pipeline.supervisor, "diagnose_audit_scorecard", return_value={
                 "audio_action": "Audio verified clean.",
                 "video_action": "Video frames verified stable.",
                 "sync_action": "AV sync within tolerance.",
             }) as mock_diagnose:

            pipeline.produce_video(
                script_text="Quality test line.",
                tts_engine="kokoro",
                engine="generative",
                denoise_audio=False,
                auto_verify=True,
            )

            mock_audit.assert_called_once()
            mock_diagnose.assert_called_once()
            report_arg = mock_diagnose.call_args[0][0]
            self.assertEqual(report_arg["overall_score"], 92.5)


if __name__ == "__main__":
    unittest.main()
