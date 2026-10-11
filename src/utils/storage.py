"""
Storage & Google Drive Persistence Manager
Ensures checkpoints, inputs, and rendered outputs persist across ephemeral Colab sessions.
"""

import os
import shutil


DEFAULT_BASE_DIR = "/content/drive/MyDrive/video_gen_platform"


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mounts Google Drive inside Google Colab environment.
    """
    try:
        from google.colab import drive
        if not os.path.exists(mount_point):
            drive.mount(mount_point)
            print(f"[Storage] Successfully mounted Google Drive at {mount_point}")
        else:
            print(f"[Storage] Google Drive already mounted at {mount_point}")
        return True
    except ImportError:
        print("[Storage] Not running in Google Colab. Using local filesystem.")
        return False
    except Exception as e:
        print(f"[Storage] Notice during drive mount: {e}")
        return False


def get_workspace_dirs(base_dir: str = DEFAULT_BASE_DIR) -> dict:
    """
    Creates and returns directory paths for checkpoints, inputs, and outputs.
    """
    dirs = {
        "base": base_dir,
        "checkpoints": os.path.join(base_dir, "checkpoints"),
        "inputs": os.path.join(base_dir, "inputs"),
        "outputs": os.path.join(base_dir, "outputs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def copy_output_to_drive(local_path: str, base_dir: str = DEFAULT_BASE_DIR) -> str:
    """
    Copies a generated artifact (WAV/MP4) from Colab local disk to persistent Google Drive.
    """
    if not os.path.exists(local_path):
        return ""
    
    filename = os.path.basename(local_path)
    drive_dest = os.path.join(base_dir, "outputs", filename)
    try:
        os.makedirs(os.path.dirname(drive_dest), exist_ok=True)
        shutil.copy2(local_path, drive_dest)
        print(f"[Storage] Output persisted to Google Drive: {drive_dest}")
        return drive_dest
    except Exception as e:
        print(f"[Storage] Could not persist {filename} to Google Drive: {e}")
        return local_path
