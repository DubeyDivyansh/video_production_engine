"""
Unit tests for Speech Synthesis Engines (Kokoro-82M and Voice Cloner).
Ensures zero-VRAM compliance, lazy loading, and robust CPU fallback.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestKokoroEngine(unittest.TestCase):
    def test_kokoro_engine_initialization_and_cpu_fallback(self):
        from src.audio.kokoro_engine import KokoroEngine

        engine = KokoroEngine(device="cpu")
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.sample_rate, 24000)
        self.assertIn("af_heart", engine.available_voices)
        # Verify lazy loading - pipeline should not be instantiated upon __init__
        self.assertIsNone(engine._pipeline)

    @patch("src.audio.kokoro_engine.soundfile.write")
    def test_kokoro_synthesize_mocked(self, mock_sf_write):
        from src.audio.kokoro_engine import KokoroEngine

        engine = KokoroEngine(device="cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_kokoro.wav")

            # Mock internal inference
            with patch.object(engine, "_generate_raw_audio", return_value=[0.0] * 24000):
                result_path = engine.synthesize(
                    text="Hello world from zero VRAM audio engine.",
                    voice="af_heart",
                    output_path=out_file
                )
                self.assertEqual(result_path, out_file)
                mock_sf_write.assert_called_once()

    def test_kokoro_generate_raw_audio_fallback(self):
        from src.audio.kokoro_engine import KokoroEngine
        import numpy as np

        engine = KokoroEngine(device="cpu")
        audio = engine._generate_raw_audio(text="Hello fallback", voice="af_heart")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_kokoro_synthesize_file_creation(self):
        from src.audio.kokoro_engine import KokoroEngine

        engine = KokoroEngine(device="cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_file.wav")
            result_path = engine.synthesize(
                text="Zero VRAM file creation test.",
                voice="af_heart",
                output_path=out_file
            )
            self.assertEqual(result_path, out_file)
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(os.path.getsize(out_file), 44)  # Valid WAV header size

    def test_kokoro_unload(self):
        from src.audio.kokoro_engine import KokoroEngine

        engine = KokoroEngine(device="cpu")
        engine._pipeline = "MOCK_ACTIVE"
        engine.unload()
        self.assertIsNone(engine._pipeline)

    def test_audio_init_exports(self):
        import src.audio as audio_mod

        self.assertTrue(hasattr(audio_mod, "KokoroEngine"))
        self.assertTrue(hasattr(audio_mod, "F5VoiceCloner"))
        self.assertIn("KokoroEngine", audio_mod.__all__)
        self.assertIn("F5VoiceCloner", audio_mod.__all__)


if __name__ == "__main__":
    unittest.main()
