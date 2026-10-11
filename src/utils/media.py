import os
import subprocess
import urllib.request
import shutil


def check_ffmpeg() -> bool:
    """Checks if ffmpeg binary is accessible in system PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed or not found in system PATH.")
    return True


def get_media_duration(file_path: str) -> float:
    """Returns the duration of a video or audio file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 10.0  # Fallback duration if ffprobe cannot read stream


def extract_audio_from_video(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> str:
    """
    Extracts a 16kHz mono PCM WAV file from an MP4 video using ffmpeg.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav_path


def trim_audio_prompt(input_wav: str, output_wav: str, duration_sec: float = 7.5) -> str:
    """
    Trims the beginning of an audio file to the optimal duration (5-10s)
    recommended for F5-TTS zero-shot voice prompt conditioning.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-t", str(duration_sec),
        output_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav


def download_file(url: str, output_path: str) -> str:
    """
    Downloads a remote asset to the local path if it doesn't already exist.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[Media] Downloading {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path
