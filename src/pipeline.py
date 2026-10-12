"""
Top-Level Video Production Pipeline Orchestrator.
Coordinates End-to-End Execution:
Avatar Profile Resolution -> Audio Synthesis & Mastering -> VRAM Fence -> Video Generation -> Face Restoration -> Background Compositing -> Cloud Storage
"""

import os
from typing import Optional

from .audio.f5_engine import F5VoiceCloner
from .audio.kokoro_engine import KokoroEngine
from .video.sadtalker_engine import SadTalkerEngine
from .video.latentsync_engine import LatentSyncEngine
from .video.liveportrait_engine import LivePortraitEngine
from .video.musetalk_engine import MuseTalkEngine
from .video.generative_engine import GenerativeVideoEngine
from .avatar.manager import AvatarManager
from .enhancers.background import BackgroundReplacer
from .enhancers.face_enhancer import FaceEnhancer
from .enhancers.audio_enhancer import AudioEnhancer
from .enhancers.motion_interpolator import MotionInterpolator
from .enhancers.studio_compositor import StudioCompositor
from .orchestration.gemini_agent_supervisor import GeminiAgentSupervisor
from .verification.inspector import QualityInspector, VerificationReport
from .utils.memory import flush_vram, check_gpu_status
from .utils.storage import mount_google_drive, copy_output_to_drive, get_workspace_dirs


class VideoProductionPipeline:
    """
    Unified Production Pipeline for generating personalized AI videos.
    Designed for zero-cost, zero-OOM execution on Google Colab T4 GPU (16GB).
    Supports:
    - Persistent Avatar Knowledge Base (AvatarManager)
    - Multi-engine dispatching: LatentSync, SadTalker, LivePortrait, MuseTalk, Generative (LTX-Video/Wan)
    - Speech engines: F5-TTS (voice cloning) & Kokoro-82M (high-speed zero-VRAM narration)
    - Studio Compositor: Background removal/replacement, Face super-resolution, Audio mastering, RIFE 60fps
    - Autonomous QA supervision & self-annealing diagnostic loop (Gemini)
    """

    def __init__(self, use_drive: bool = True):
        self.use_drive = use_drive
        if self.use_drive:
            mount_google_drive()

        self.workspace = get_workspace_dirs()
        self.avatar_manager = AvatarManager(use_drive=self.use_drive)
        self.voice_cloner = F5VoiceCloner()
        self.kokoro = KokoroEngine()
        self.sadtalker = SadTalkerEngine()
        self.latentsync = LatentSyncEngine()
        self.liveportrait = LivePortraitEngine()
        self.musetalk = MuseTalkEngine()
        self.generative = GenerativeVideoEngine()
        self.bg_replacer = BackgroundReplacer()
        self.face_enhancer = FaceEnhancer()
        self.audio_enhancer = AudioEnhancer()
        self.interpolator = MotionInterpolator()
        self.compositor = StudioCompositor()
        self.supervisor = GeminiAgentSupervisor()
        self.inspector = QualityInspector()

    def produce_video(
        self,
        script_text: str,
        avatar_id: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        face_image_path: Optional[str] = None,
        body_image_path: Optional[str] = None,
        base_video_path: Optional[str] = None,
        output_name: str = "production_video.mp4",
        engine: str = "latentsync",
        motion_mode: str = "subtle_breathing",
        studio_backdrop: str = "#FFFFFF",
        denoise_audio: bool = True,
        enhance_face: bool = False,
        replace_background: Optional[str] = None,
        color_backdrop: Optional[str] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.15,
        add_subtitles: bool = False,
        auto_verify: bool = True,
        tts_engine: str = "f5",
        generate_scene_prompt: Optional[str] = None,
        interpolate_60fps: bool = False,
    ) -> str:
        """
        Executes the full production pipeline from script to final video.

        Args:
            script_text: Custom text to be spoken by the avatar.
            avatar_id: Optional ID of persistent avatar profile on Google Drive.
            ref_audio_path: Reference audio of user's voice (overrides avatar profile).
            face_image_path: Path to user's photo (overrides avatar profile).
            body_image_path: Path to user's whole-body suit/pose image (overrides avatar profile).
            base_video_path: Path to base video with body movement (overrides avatar profile).
            output_name: Filename for the rendered MP4 video.
            engine: Engine choice ('latentsync', 'sadtalker', 'liveportrait', 'musetalk', 'generative', 'ltx-video').
            motion_mode: Motion dynamics for still body images ('subtle_breathing' or 'static_hold').
            studio_backdrop: Hex color (default '#FFFFFF') for studio background canvas.
            denoise_audio: Whether to run spectral noise reduction and -16 LUFS mastering.
            enhance_face: Whether to apply GFPGAN face restoration & boundary softening.
            replace_background: Path to image or video to composite as background.
            color_backdrop: Hex color (e.g., '#00FF00') to apply as solid background.
            bgm_path: Path to background music track (MP3/WAV) to mix with auto-ducking.
            bgm_volume: Volume level of background music (default 0.15).
            add_subtitles: Whether to generate and burn styled subtitles into the video.
            auto_verify: Whether to run automated quality inspection and verification report.
            tts_engine: Voice synthesis engine ('f5' for voice cloning, 'kokoro' for zero-VRAM narration).
            generate_scene_prompt: Optional prompt to synthesize diffusion video scenes via LTX-Video.
            interpolate_60fps: Whether to upsample output video to broadcast 60fps via RIFE optical flow.

        Returns:
            Path to the final synchronized MP4 video.
        """
        print("=" * 60)
        print("[*] STARTING VIDEO PRODUCTION PIPELINE")
        print("=" * 60)
        check_gpu_status()

        # Step 0: Resolve Avatar Profile Assets if avatar_id specified
        if avatar_id:
            profile = self.avatar_manager.get_avatar(avatar_id)
            print(f"[*] Loaded Avatar Profile: '{avatar_id}'")
            if not ref_audio_path and profile.optimal_audio_path:
                ref_audio_path = profile.optimal_audio_path
                print(f"    - Using optimal audio: {ref_audio_path}")
            if not face_image_path and profile.optimal_image_path:
                face_image_path = profile.optimal_image_path
                print(f"    - Using optimal image: {face_image_path}")
            if not body_image_path and profile.optimal_body_pose_path:
                body_image_path = profile.optimal_body_pose_path
                print(f"    - Using optimal body pose: {body_image_path}")
            if not base_video_path and profile.optimal_video_path:
                base_video_path = profile.optimal_video_path
                print(f"    - Using optimal base video: {base_video_path}")

        tts_mode = tts_engine.lower()
        if tts_mode not in ("f5", "kokoro"):
            raise ValueError(
                f"Unknown tts_engine: '{tts_engine}'. Choose from 'f5', 'kokoro'."
            )

        if tts_mode == "f5":
            if not ref_audio_path or not os.path.exists(ref_audio_path):
                raise ValueError(
                    f"Valid reference audio is required for F5-TTS. Provided: {ref_audio_path}"
                )

        # Step 1: Synthesize Voice
        temp_audio_path = os.path.join(self.workspace["outputs"], "generated_speech.wav")
        if tts_mode == "kokoro":
            print("\n--- [Phase 1/5] Speech Narration (Kokoro-82M) ---")
            cloned_wav = self.kokoro.synthesize(
                text=script_text,
                output_path=temp_audio_path,
            )
        else:
            print("\n--- [Phase 1/5] Voice Cloning (F5-TTS) ---")
            cloned_wav = self.voice_cloner.clone_voice(
                ref_audio_path=ref_audio_path,
                gen_text=script_text,
                output_wav_path=temp_audio_path,
            )

        # Step 2: Audio Mastering (Denoise + -16 LUFS)
        if denoise_audio:
            print("\n--- [Phase 2/5] Studio Audio Mastering ---")
            mastered_wav = os.path.join(self.workspace["outputs"], "mastered_speech.wav")
            self.audio_enhancer.master_speech(cloned_wav, mastered_wav)
            cloned_wav = mastered_wav

        # Step 3: VRAM Memory Fence
        print("\n--- [Phase 3/5] VRAM Memory Fence ---")
        self.voice_cloner.unload()
        self.kokoro.unload()
        flush_vram()

        # Step 4: Video Animation
        print(f"\n--- [Phase 4/5] Video Animation Engine: {engine.upper()} ---")
        temp_video_dest = os.path.join(self.workspace["outputs"], output_name)
        eng = engine.lower()

        rendered_video = None

        # Generative Scene Prompt Handling
        if generate_scene_prompt:
            print(f"\n[*] Generating base scene from prompt: '{generate_scene_prompt}'")
            expanded = self.supervisor.expand_creative_prompt(generate_scene_prompt)
            pos_prompt = expanded.get("positive_prompt", generate_scene_prompt) if isinstance(expanded, dict) else str(expanded)
            neg_prompt = expanded.get("negative_prompt", "worst quality, blurry, jittery, distorted, morphing") if isinstance(expanded, dict) else "worst quality, blurry, jittery, distorted, morphing"

            gen_scene_dest = temp_video_dest if eng in ("generative", "ltx-video") else os.path.join(self.workspace["outputs"], f"gen_scene_{output_name}")
            gen_video = self.generative.generate_scene(
                prompt=pos_prompt,
                output_path=gen_scene_dest,
                negative_prompt=neg_prompt,
            )
            if eng in ("generative", "ltx-video"):
                rendered_video = gen_video
            else:
                base_video_path = gen_video

        if eng in ("generative", "ltx-video"):
            if rendered_video is None:
                scene_prompt = generate_scene_prompt or script_text
                expanded = self.supervisor.expand_creative_prompt(scene_prompt)
                pos_prompt = expanded.get("positive_prompt", scene_prompt) if isinstance(expanded, dict) else str(expanded)
                neg_prompt = expanded.get("negative_prompt", "worst quality, blurry, jittery, distorted, morphing") if isinstance(expanded, dict) else "worst quality, blurry, jittery, distorted, morphing"
                rendered_video = self.generative.generate_scene(
                    prompt=pos_prompt,
                    output_path=temp_video_dest,
                    negative_prompt=neg_prompt,
                )
        elif eng == "latentsync":
            if not base_video_path or not os.path.exists(base_video_path):
                source_image = body_image_path or face_image_path
                if source_image and os.path.exists(source_image):
                    print(f"[*] Synthesizing full-body base video from image: {source_image} (mode: {motion_mode})...")
                    from src.video.fullbody_animator import FullBodyAnimator
                    animator = FullBodyAnimator(bg_color_hex=studio_backdrop)
                    duration_s = self._get_audio_duration(cloned_wav)
                    synth_video_path = os.path.join(self.workspace["outputs"], "fullbody_base_synth.mp4")
                    animator.synthesize_base_video(
                        image_path=source_image,
                        duration_s=duration_s,
                        output_video_path=synth_video_path,
                        motion_mode=motion_mode,
                        matting_enhancer=self.bg_replacer,
                    )
                    base_video_path = synth_video_path
                else:
                    raise ValueError("LatentSync requires a valid 'base_video_path' or 'body_image_path'.")
            rendered_video = self.latentsync.sync_lips(
                video_path=base_video_path,
                audio_path=cloned_wav,
                output_video_path=temp_video_dest,
            )
        elif eng == "sadtalker":
            if not face_image_path or not os.path.exists(face_image_path):
                raise ValueError("SadTalker requires a valid 'face_image_path'.")
            rendered_video = self.sadtalker.generate(
                source_image=face_image_path,
                driven_audio=cloned_wav,
                still_mode=True,
                use_enhancer=True,
            )
        elif eng == "liveportrait":
            if not face_image_path or not base_video_path:
                raise ValueError("LivePortrait requires both 'face_image_path' and 'base_video_path'.")
            rendered_video = self.liveportrait.generate(
                source_image=face_image_path,
                driving_video=base_video_path,
                output_dir=self.workspace["outputs"],
            )
        elif eng == "musetalk":
            if not base_video_path or not os.path.exists(base_video_path):
                raise ValueError("MuseTalk requires a valid 'base_video_path'.")
            rendered_video = self.musetalk.sync_lips(
                video_path=base_video_path,
                audio_path=cloned_wav,
                output_dir=self.workspace["outputs"],
            )
        else:
            raise ValueError(
                f"Unknown engine: {engine}. Choose from 'latentsync', 'sadtalker', 'liveportrait', 'musetalk', 'generative', 'ltx-video'."
            )

        current_video = rendered_video

        # Step 5: Post-Production Enhancers (Face Super-Resolution & Background Replacement)
        print("\n--- [Phase 5/5] Studio Post-Processing ---")
        if enhance_face:
            print("[*] Applying Face Super-Resolution (GFPGAN)...")
            enhanced_video = os.path.join(self.workspace["outputs"], f"enhanced_{output_name}")
            current_video = self.face_enhancer.enhance_video(current_video, enhanced_video)

        if replace_background and os.path.exists(replace_background):
            print(f"[*] Replacing background with: '{replace_background}'...")
            bg_video = os.path.join(self.workspace["outputs"], f"studio_{output_name}")
            current_video = self.bg_replacer.replace_background(current_video, replace_background, bg_video)
        elif color_backdrop:
            # Parse hex color into BGR
            hex_c = color_backdrop.lstrip("#")
            rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            print(f"[*] Applying solid studio backdrop: {color_backdrop}...")
            bg_video = os.path.join(self.workspace["outputs"], f"color_bg_{output_name}")
            current_video = self.bg_replacer.apply_color_backdrop(current_video, bg_video, color_bgr=bgr)

        if interpolate_60fps:
            print("[*] Applying 60fps Motion Interpolation (RIFE)...")
            interpolated_video = os.path.join(self.workspace["outputs"], f"interpolated_{output_name}")
            current_video = self.interpolator.interpolate_60fps(current_video, interpolated_video)

        if add_subtitles:
            print("[*] Generating and burning styled subtitles...")
            from .utils.media import get_media_duration
            duration = get_media_duration(current_video)
            srt_path = os.path.join(self.workspace["outputs"], "subtitles.srt")
            self.compositor.generate_srt_from_script(script_text, duration, srt_path)
            sub_video = os.path.join(self.workspace["outputs"], f"subtitled_{output_name}")
            current_video = self.compositor.burn_subtitles(current_video, srt_path, sub_video)

        if bgm_path and os.path.exists(bgm_path):
            print(f"[*] Mixing background music: '{bgm_path}' with volume ducking...")
            bgm_video = os.path.join(self.workspace["outputs"], f"bgm_{output_name}")
            current_video = self.compositor.mix_bgm(current_video, bgm_path, bgm_video, bgm_volume=bgm_volume, enable_ducking=True)

        # Step 6: Autonomous Quality Verification
        if auto_verify:
            print("\n--- [Phase 6/6] Autonomous Quality Verification ---")
            report = self.inspector.audit_deliverable(
                video_path=current_video,
                speech_wav_path=cloned_wav,
                script_text=script_text,
            )
            report.print_summary()
            scorecard_dict = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            diagnosis = self.supervisor.diagnose_audit_scorecard(scorecard_dict)
            print(f"[*] Supervisor QA Diagnosis: {diagnosis}")
            audit_dest = os.path.splitext(current_video)[0] + "_quality_audit.json"
            if os.path.exists(audit_dest) and self.use_drive:
                copy_output_to_drive(audit_dest, os.path.basename(audit_dest))

        final_dest = copy_output_to_drive(current_video, output_name)
        print(f"\n[*] Production complete! Video saved to: {final_dest}")
        return final_dest

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Determines the precise duration in seconds of an audio file.
        """
        try:
            import soundfile as sf
            return float(sf.info(audio_path).duration)
        except Exception:
            try:
                import wave
                with wave.open(audio_path, "rb") as f:
                    return float(f.getnframes()) / float(f.getframerate())
            except Exception:
                from .utils.media import get_media_duration
                return float(get_media_duration(audio_path))


