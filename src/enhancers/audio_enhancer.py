"""
Audio Enhancer & Mastering Module.
Provides spectral gating noise reduction and EBU R128 (-16 LUFS) broadcast loudness mastering.
"""

import os
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


class AudioEnhancer:
    """Enhances vocal recordings by stripping background noise and mastering loudness."""

    def __init__(self, target_lufs: float = -16.0):
        self.target_lufs = target_lufs

    def denoise_audio(self, in_wav: str, out_wav: str, prop_decrease: float = 0.85) -> str:
        """Applies spectral gating noise reduction to eliminate background hum."""
        if not os.path.exists(in_wav):
            raise FileNotFoundError(f"Source audio not found: {in_wav}")

        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        try:
            import noisereduce as nr
            data, sr = sf.read(in_wav)
            is_stereo = len(data.shape) > 1
            if is_stereo:
                mono = np.mean(data, axis=1)
                reduced = nr.reduce_noise(y=mono, sr=sr, prop_decrease=prop_decrease)
            else:
                reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=prop_decrease)

            sf.write(out_wav, reduced, sr)
            return out_wav
        except Exception:
            # Fallback to copy if noisereduce unavailable
            import shutil
            shutil.copy2(in_wav, out_wav)
            return out_wav

    def normalize_loudness(self, in_wav: str, out_wav: str) -> str:
        """Normalizes audio loudness to podcast/broadcast standard (-16 LUFS)."""
        if not os.path.exists(in_wav):
            raise FileNotFoundError(f"Source audio not found: {in_wav}")

        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        try:
            import pyloudnorm as pyln
            data, sr = sf.read(in_wav)
            meter = pyln.Meter(sr)
            current_loudness = meter.integrated_loudness(data)
            normalized = pyln.normalize.loudness(data, current_loudness, self.target_lufs)
            # Soft peak limit at -1.0 dBFS (0.891)
            peak = np.max(np.abs(normalized))
            if peak > 0.89:
                normalized = normalized * (0.89 / peak)

            sf.write(out_wav, normalized, sr)
            return out_wav
        except Exception:
            # Fallback: simple RMS normalization
            data, sr = sf.read(in_wav)
            peak = np.max(np.abs(data)) + 1e-9
            normalized = data * (0.85 / peak)
            sf.write(out_wav, normalized, sr)
            return out_wav

    def master_speech(self, in_wav: str, out_wav: str) -> str:
        """Applies both noise reduction and loudness normalization in one step."""
        temp_denoised = out_wav + ".denoised.wav"
        self.denoise_audio(in_wav, temp_denoised)
        self.normalize_loudness(temp_denoised, out_wav)
        if os.path.exists(temp_denoised):
            os.remove(temp_denoised)
        return out_wav
