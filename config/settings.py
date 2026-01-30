"""
MLB Video Pipeline - Application Settings

Central configuration management using Pydantic for validation.
All settings can be overridden via environment variables.

Usage:
    from config.settings import Settings
    settings = Settings()
    print(settings.openai_api_key)
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden by setting the corresponding
    environment variable (uppercase with underscores).
    """

    # =========================================================================
    # Project Paths
    # =========================================================================

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )
    data_dir: Path = Field(default=None, description="Data storage directory")
    models_dir: Path = Field(default=None, description="ML models directory")
    outputs_dir: Path = Field(default=None, description="Generated outputs directory")
    logs_dir: Path = Field(default=None, description="Log files directory")

    # =========================================================================
    # API Keys
    # =========================================================================

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT-4"
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key for TTS"
    )
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # Default "Rachel" voice
        description="ElevenLabs voice ID"
    )
    youtube_api_key: Optional[str] = Field(
        default=None,
        description="YouTube Data API key"
    )

    # =========================================================================
    # AWS Settings
    # =========================================================================

    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_region: str = Field(default="us-east-1")
    s3_bucket_name: str = Field(default="mlb-video-pipeline")

    # =========================================================================
    # MLB API Settings
    # =========================================================================

    mlb_api_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="MLB API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum API retry attempts"
    )
    max_games_per_day: int = Field(
        default=15,
        description="Maximum games to process per day"
    )

    # =========================================================================
    # Cost Controls
    # =========================================================================

    daily_cost_limit: float = Field(
        default=10.0,
        ge=0,
        description="Maximum daily API spend in USD"
    )
    openai_max_tokens: int = Field(
        default=500,
        ge=100,
        le=4000,
        description="Max tokens per GPT request"
    )
    elevenlabs_monthly_chars: int = Field(
        default=100000,
        description="Monthly character limit for ElevenLabs"
    )

    # =========================================================================
    # Video Settings
    # =========================================================================

    video_width: int = Field(default=1080, description="Video width in pixels")
    video_height: int = Field(default=1920, description="Video height in pixels")
    video_fps: int = Field(default=30, ge=24, le=60)
    video_quality: str = Field(default="high")

    # =========================================================================
    # Logging
    # =========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # =========================================================================
    # Feature Flags
    # =========================================================================

    enable_cost_tracking: bool = Field(default=True)
    enable_youtube_upload: bool = Field(default=False)
    dry_run: bool = Field(default=False, description="Run without making API calls")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("data_dir", "models_dir", "outputs_dir", "logs_dir", mode="before")
    @classmethod
    def set_default_paths(cls, v, info):
        """Set default directory paths based on base_dir."""
        if v is None:
            field_name = info.field_name
            base_dir = info.data.get("base_dir")
            if base_dir:
                return base_dir / field_name.replace("_dir", "")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw data directory."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed data directory."""
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        """Path to cache directory."""
        return self.data_dir / "cache"

    @property
    def video_output_dir(self) -> Path:
        """Path to video output directory."""
        return self.outputs_dir / "videos"

    @property
    def audio_output_dir(self) -> Path:
        """Path to audio output directory."""
        return self.outputs_dir / "audio"

    @property
    def scripts_output_dir(self) -> Path:
        """Path to scripts output directory."""
        return self.outputs_dir / "scripts"

    def validate_api_keys(self) -> dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "openai": bool(self.openai_api_key),
            "elevenlabs": bool(self.elevenlabs_api_key),
            "youtube": bool(self.youtube_api_key),
            "aws": bool(self.aws_access_key_id and self.aws_secret_access_key),
        }

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.models_dir / "training_data",
            self.video_output_dir,
            self.audio_output_dir,
            self.scripts_output_dir,
            self.outputs_dir / "thumbnails",
            self.logs_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience: Create default instance
settings = get_settings()
