"""
Top-Level Video Production Pipeline Orchestrator
Coordinates End-to-End Execution: Audio Synthesis -> VRAM Fence -> Video Generation -> Storage
"""

import os
from .audio.f5_engine import F5VoiceCloner
from .video.sadtalker_engine import SadTalkerEngine
from .video.latentsync_engine import LatentSyncEngine
from .utils.memory import flush_vram, check_gpu_status
from .utils.storage import mount_google_drive, copy_output_to_drive, get_workspace_dirs


class VideoProductionPipeline:
    """
    Unified Production Pipeline for generating personalized AI videos.
    Designed for zero-cost, zero-OOM execution on Google Colab T4 GPU (16GB).
    """

    def __init__(self, use_drive: bool = True):
        self.use_drive = use_drive
        if self.use_drive:
            mount_google_drive()
        
        self.workspace = get_workspace_dirs()
        self.voice_cloner = F5VoiceCloner()
        self.sadtalker = SadTalkerEngine()
        self.latentsync = LatentSyncEngine()

    def produce_video(
        self,
        face_image_path: str,
        script_text: str,
        ref_audio_path: str,
        output_name: str = "production_video.mp4",
        engine: str = "sadtalker",
        base_video_path: str = None,
    ) -> str:
        """
        Executes the full production pipeline from script to final video.

        Args:
            face_image_path: Path to user's photo (passport or screenshot).
            script_text: The custom text to be spoken.
            ref_audio_path: The reference audio of user's voice for cloning.
            output_name: Filename for the rendered MP4 video.
            engine: 'sadtalker' for still portrait animation or 'latentsync' for full-body video sync.
            base_video_path: Required if engine is 'latentsync' (source video with body movement).

        Returns:
            Path to the final synchronized MP4 video.
        """
        print("=" * 60)
        print("🎬 STARTING VIDEO PRODUCTION PIPELINE")
        print("=" * 60)
        check_gpu_status()

        # Step 1: Synthesize Cloned Voice
        print("\n--- [Phase 1/3] Voice Cloning (F5-TTS) ---")
        temp_audio_path = os.path.join(self.workspace["outputs"], "generated_speech.wav")
        cloned_wav = self.voice_cloner.clone_voice(
            ref_audio_path=ref_audio_path,
            gen_text=script_text,
            output_wav_path=temp_audio_path,
        )

        # Step 2: Memory Fence - Unload audio model and purge VRAM
        print("\n--- [Phase 2/3] VRAM Memory Fence ---")
        self.voice_cloner.unload()
        flush_vram()

        # Step 3: Video Animation
        print(f"\n--- [Phase 3/3] Video Animation Engine: {engine.upper()} ---")
        temp_video_dest = os.path.join(self.workspace["outputs"], output_name)

        if engine.lower() == "sadtalker":
            rendered_video = self.sadtalker.generate(
                source_image=face_image_path,
                driven_audio=cloned_wav,
                still_mode=True,
                use_enhancer=True,
            )
        elif engine.lower() == "latentsync":
            if not base_video_path or not os.path.exists(base_video_path):
                raise ValueError("LatentSync requires a valid 'base_video_path' input.")
            rendered_video = self.latentsync.sync_lips(
                video_path=base_video_path,
                audio_path=cloned_wav,
                output_video_path=temp_video_dest,
            )
        else:
            raise ValueError(f"Unknown engine: {engine}. Choose 'sadtalker' or 'latentsync'.")

        # Step 4: Persist to Google Drive
        final_video_path = copy_output_to_drive(rendered_video)

        print("\n" + "=" * 60)
        print(f"🎉 PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Deliverable: {final_video_path}")
        print("=" * 60)
        return final_video_path
