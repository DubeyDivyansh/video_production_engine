"""
F5-TTS Voice Cloning Engine
Flow-Matching Diffusion Transformer for faithful zero-shot voice synthesis.
"""

import os
import torch
from ..utils.media import trim_audio_prompt

try:
    import soundfile as sf
except ImportError:
    sf = None


class F5VoiceCloner:
    """
    Zero-shot voice cloning pipeline wrapper for F5-TTS.
    Runs on Tesla T4 with ~2.5 - 2.8 GB peak VRAM.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None

    def load_model(self):
        """
        Lazily instantiates the F5-TTS model on the selected device.
        """
        if self.model is None:
            try:
                from f5_tts.api import F5TTS
            except ImportError:
                raise ImportError(
                    "f5-tts is not installed. Run 'pip install f5-tts vocos soundfile' first."
                )
            
            print(f"[F5-TTS] Initializing model on device: {self.device}")
            self.model = F5TTS(device=self.device)
            print("[F5-TTS] Model loaded successfully.")

    def clone_voice(
        self,
        ref_audio_path: str,
        gen_text: str,
        output_wav_path: str,
        ref_text: str = "",
        max_prompt_duration: float = 8.0,
    ) -> str:
        """
        Synthesizes speech in the voice of the reference audio speaker.

        Args:
            ref_audio_path: Path to the reference audio WAV file.
            gen_text: Target text to speak.
            output_wav_path: File path where synthesized WAV should be written.
            ref_text: Transcription of the reference prompt (optional, improves cloning).
            max_prompt_duration: Maximum duration in seconds for the prompt conditioning.

        Returns:
            Absolute path to the generated WAV file.
        """
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        # Automatically trim prompt if needed for optimal flow-matching guidance
        prompt_path = "/tmp/f5_trimmed_prompt.wav" if os.name != "nt" else "f5_trimmed_prompt.wav"
        trim_audio_prompt(ref_audio_path, prompt_path, duration_sec=max_prompt_duration)

        self.load_model()

        print(f"[F5-TTS] Generating speech for text: '{gen_text}'")
        wav, sr, _ = self.model.infer(
            ref_file=prompt_path,
            ref_text=ref_text,
            gen_text=gen_text,
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
        sf.write(output_wav_path, wav, sr)
        print(f"[F5-TTS] Audio generated successfully: {output_wav_path}")
        return output_wav_path

    def unload(self):
        """
        Unloads model references to allow full VRAM garbage collection.
        """
        if self.model is not None:
            del self.model
            self.model = None
            print("[F5-TTS] Model unloaded from memory.")
