"""
MLB Video Pipeline - Text-to-Speech Engine

Generates natural narration using ElevenLabs API.
Handles voice selection, audio generation, and file management.

Compatible with ElevenLabs SDK v1.x

Usage:
    from src.audio.tts_engine import TTSEngine

    tts = TTSEngine()

    # Generate narration
    audio_path = tts.generate_narration(
        "Welcome to today's game recap...",
        output_name="game_recap_12345"
    )

    # List available voices
    voices = tts.list_voices()
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import hashlib

from elevenlabs import ElevenLabs

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker


logger = get_logger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using ElevenLabs.

    Generates natural-sounding narration for video scripts.
    Supports multiple voices and voice settings customization.

    Compatible with ElevenLabs SDK v1.x
    """

    # ElevenLabs pricing (approximate, per 1K characters)
    PRICE_PER_1K_CHARS = 0.18

    # Recommended voices for sports narration
    RECOMMENDED_VOICES = {
        "adam": "pNInz6obpgDQGcFmaJgB",  # Deep, authoritative
        "josh": "TxGEqnHWrfWFTfGW9XjX",  # Energetic
        "sam": "yoZ06aMxZJJ28mfd3POQ",   # Clear, neutral
        "rachel": "21m00Tcm4TlvDq8ikWAM", # Professional female
    }

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize the TTS engine.

        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use for narration
            output_dir: Directory for audio output
        """
        self.api_key = api_key or settings.elevenlabs_api_key
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")

        self.voice_id = voice_id or settings.elevenlabs_voice_id
        self.output_dir = output_dir or settings.audio_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ElevenLabs client (SDK v1.x)
        self.client = ElevenLabs(api_key=self.api_key)
        self.cost_tracker = get_cost_tracker()

        # Default voice settings
        self.voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        }

        logger.info(f"TTSEngine initialized with voice: {self.voice_id}")

    # =========================================================================
    # Voice Management
    # =========================================================================

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices.

        Returns:
            List of voice information dictionaries
        """
        try:
            response = self.client.voices.get_all()
            voices = []

            for voice in response.voices:
                voices.append({
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": getattr(voice, 'category', None),
                    "labels": getattr(voice, 'labels', {}),
                })

            return voices
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise

    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice to use for generation.

        Args:
            voice_id: ElevenLabs voice ID
        """
        self.voice_id = voice_id
        logger.info(f"Voice set to: {voice_id}")

    def set_voice_by_name(self, name: str) -> bool:
        """
        Set voice by name (from recommended voices).

        Args:
            name: Voice name (adam, josh, sam, rachel)

        Returns:
            True if voice was found and set
        """
        name = name.lower()
        if name in self.RECOMMENDED_VOICES:
            self.voice_id = self.RECOMMENDED_VOICES[name]
            logger.info(f"Voice set to {name}: {self.voice_id}")
            return True

        logger.warning(f"Voice '{name}' not found in recommended voices")
        return False

    def configure_voice(
        self,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
    ) -> None:
        """
        Configure voice settings.

        Args:
            stability: Voice stability (0-1, higher = more consistent)
            similarity_boost: Voice clarity (0-1, higher = clearer)
            style: Style exaggeration (0-1)
        """
        self.voice_settings = {
            "stability": max(0, min(1, stability)),
            "similarity_boost": max(0, min(1, similarity_boost)),
            "style": max(0, min(1, style)),
            "use_speaker_boost": True,
        }
        logger.info(f"Voice settings updated: stability={stability}, similarity={similarity_boost}")

    # =========================================================================
    # Audio Generation
    # =========================================================================

    def generate_narration(
        self,
        text: str,
        output_name: str | None = None,
        model: str = "eleven_turbo_v2_5",
    ) -> Path:
        """
        Generate audio narration from text.

        Args:
            text: Script text to convert to speech
            output_name: Base name for output file
            model: ElevenLabs model to use (eleven_turbo_v2_5, eleven_multilingual_v2, etc.)

        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate output filename
        if output_name is None:
            text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
            output_name = f"narration_{text_hash}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{output_name}_{timestamp}.mp3"

        logger.info(f"Generating narration: {len(text)} characters")

        if settings.dry_run:
            logger.info("DRY RUN: Would generate audio")
            # Create empty placeholder file
            output_path.touch()
            return output_path

        try:
            # Generate audio using SDK v1.x text_to_speech API
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=model,
                voice_settings={
                    "stability": self.voice_settings["stability"],
                    "similarity_boost": self.voice_settings["similarity_boost"],
                    "style": self.voice_settings["style"],
                    "use_speaker_boost": self.voice_settings["use_speaker_boost"],
                },
            )

            # Save audio to file
            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            # Track cost
            char_count = len(text)
            cost = (char_count / 1000) * self.PRICE_PER_1K_CHARS
            self.cost_tracker.log_api_call("elevenlabs", cost, characters=char_count)

            logger.info(f"Audio saved to {output_path} (${cost:.4f})")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate narration: {e}")
            raise

    def generate_segments(
        self,
        segments: list[dict[str, str]],
        output_prefix: str = "segment",
    ) -> list[Path]:
        """
        Generate multiple audio segments.

        Useful for scripts with different sections or speakers.

        Args:
            segments: List of {"text": str, "voice_id": str (optional)}
            output_prefix: Prefix for output filenames

        Returns:
            List of paths to generated audio files
        """
        audio_paths = []

        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            voice_id = segment.get("voice_id", self.voice_id)

            # Temporarily change voice if specified
            original_voice = self.voice_id
            if voice_id != self.voice_id:
                self.voice_id = voice_id

            try:
                output_name = f"{output_prefix}_{i:03d}"
                path = self.generate_narration(text, output_name)
                audio_paths.append(path)
            finally:
                self.voice_id = original_voice

        logger.info(f"Generated {len(audio_paths)} audio segments")
        return audio_paths

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of an audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            from moviepy.editor import AudioFileClip

            with AudioFileClip(str(audio_path)) as audio:
                return audio.duration
        except ImportError:
            logger.warning("moviepy not available, cannot get audio duration")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def estimate_duration(self, text: str, wpm: int = 150) -> float:
        """
        Estimate narration duration without generating audio.

        Args:
            text: Script text
            wpm: Words per minute speaking rate

        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        return (word_count / wpm) * 60

    def estimate_cost(self, text: str) -> float:
        """
        Estimate cost for generating narration.

        Args:
            text: Script text

        Returns:
            Estimated cost in USD
        """
        char_count = len(text)
        return (char_count / 1000) * self.PRICE_PER_1K_CHARS

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def check_character_limit(self) -> dict[str, int]:
        """
        Check remaining character allocation.

        Returns:
            Dictionary with usage information
        """
        try:
            user_info = self.client.user.get()
            subscription = user_info.subscription
            return {
                "character_count": subscription.character_count,
                "character_limit": subscription.character_limit,
                "remaining": subscription.character_limit - subscription.character_count,
            }
        except Exception as e:
            logger.error(f"Failed to check character limit: {e}")
            return {"error": str(e)}

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()

    def health_check(self) -> bool:
        """
        Check if the ElevenLabs API is accessible.

        Returns:
            True if API is responding
        """
        try:
            self.client.user.get()
            return True
        except Exception:
            return False
