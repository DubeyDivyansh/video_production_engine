"""
Memory Management & VRAM Fencing for Google Colab GPU (Tesla T4)
"""

import gc
import os
import subprocess
import torch


def configure_t4_allocator():
    """
    Configures PyTorch memory allocator to avoid fragmentation on Tesla T4.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def flush_vram(verbose: bool = True):
    """
    Frees GPU cache and forces garbage collection.
    Essential to run between audio synthesis (F5-TTS) and video synthesis (LatentSync/SadTalker).
    """
    if torch.cuda.is_available():
        before_allocated = torch.cuda.memory_allocated() / (1024**2)
        before_reserved = torch.cuda.memory_reserved() / (1024**2)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        after_allocated = torch.cuda.memory_allocated() / (1024**2)
        after_reserved = torch.cuda.memory_reserved() / (1024**2)

        if verbose:
            print(f"[VRAM Fence] Allocated: {before_allocated:.1f}MB -> {after_allocated:.1f}MB")
            print(f"[VRAM Fence] Reserved:  {before_reserved:.1f}MB -> {after_reserved:.1f}MB")
            free_mb = 15360 - after_reserved
            print(f"[VRAM Fence] Free T4 VRAM available: {free_mb:.1f}MB")
    else:
        if verbose:
            print("[VRAM Fence] No CUDA GPU detected.")


def check_gpu_status():
    """
    Runs nvidia-smi and prints formatted GPU diagnostics.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        capability = torch.cuda.get_device_capability(0)
        print(f"[GPU Status] {name} | VRAM: {total_mem:.2f} GB | Compute Capability: {capability}")
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                print(f"[GPU Memory] Used/Free: {res.stdout.strip()}")
        except Exception:
            pass
    else:
        print("[GPU Status] Running in CPU mode.")
