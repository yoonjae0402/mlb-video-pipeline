"""
MLB Video Pipeline - Custom Exceptions

Hierarchical exception classes for the pipeline:
- DataFetchError: API and data retrieval failures
- ModelPredictionError: ML model issues
- ScriptGenerationError: Content generation failures
- AudioGenerationError: TTS failures
- VideoGenerationError: Video rendering issues
- ConfigurationError: Missing/invalid config
- UploadError: Platform upload failures
- ValidationError: Data validation issues
- RateLimitError: API rate limiting

Each exception includes context (game_id, timestamp, etc.) for debugging.

Usage:
    from src.utils.exceptions import DataFetchError, ScriptGenerationError

    raise DataFetchError(
        "Failed to fetch game data",
        source="MLB-StatsAPI",
        game_id="748589",
        status_code=503
    )
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


# =============================================================================
# Base Exception
# =============================================================================

class PipelineError(Exception):
    """
    Base exception class for the MLB video pipeline.

    All pipeline exceptions inherit from this class and include:
    - message: Human-readable error description
    - timestamp: When the error occurred (UTC)
    - context: Additional debugging information

    Attributes:
        message (str): Error message
        timestamp (str): ISO 8601 timestamp
        context (dict): Additional context data
        error_code (str): Optional error code for categorization
    """

    error_code: str = "PIPELINE_ERROR"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the pipeline error.

        Args:
            message: Human-readable error description
            error_code: Optional error code override
            **kwargs: Additional context to include in error
        """
        self.message = message
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.context = kwargs

        if error_code:
            self.error_code = error_code

        super().__init__(self.message)

    def __str__(self) -> str:
        """Format error as string with context."""
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if context_str:
            return f"[{self.error_code}] {self.message} ({context_str})"
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, **{self.context!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context,
        }


# =============================================================================
# Data Fetching Exceptions
# =============================================================================

class DataFetchError(PipelineError):
    """
    Raised when fetching data from an external source fails.

    Common scenarios:
    - MLB Stats API timeout or error
    - Network connectivity issues
    - Invalid API responses
    - Authentication failures

    Example:
        raise DataFetchError(
            "Failed to fetch game schedule",
            source="MLB-StatsAPI",
            game_id="748589",
            status_code=503
        )
    """

    error_code = "DATA_FETCH_ERROR"

    def __init__(
        self,
        message: str,
        source: str,
        game_id: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize data fetch error.

        Args:
            message: Error description
            source: Data source (e.g., "MLB-StatsAPI", "ESPN")
            game_id: Related game ID if applicable
            status_code: HTTP status code if applicable
            **kwargs: Additional context
        """
        super().__init__(
            message,
            source=source,
            game_id=game_id,
            status_code=status_code,
            **kwargs
        )


class RateLimitError(DataFetchError):
    """
    Raised when an API rate limit is exceeded.

    Includes retry information when available.

    Example:
        raise RateLimitError(
            "Rate limit exceeded",
            source="OpenAI",
            retry_after=60
        )
    """

    error_code = "RATE_LIMIT_ERROR"

    def __init__(
        self,
        message: str,
        source: str,
        retry_after: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize rate limit error.

        Args:
            message: Error description
            source: API that rate limited
            retry_after: Seconds to wait before retry (if provided)
            **kwargs: Additional context
        """
        super().__init__(
            message,
            source=source,
            status_code=429,
            retry_after=retry_after,
            **kwargs
        )
        self.retry_after = retry_after


# =============================================================================
# Model Exceptions
# =============================================================================

class ModelPredictionError(PipelineError):
    """
    Raised when a machine learning model fails.

    Scenarios:
    - Model loading failure
    - Invalid input data
    - Prediction timeout
    - Out of memory

    Example:
        raise ModelPredictionError(
            "Model failed to generate prediction",
            model_name="viewer_predictor_v2",
            input_shape=(1, 25)
        )
    """

    error_code = "MODEL_ERROR"

    def __init__(
        self,
        message: str,
        model_name: str,
        model_version: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize model prediction error.

        Args:
            message: Error description
            model_name: Name of the model that failed
            model_version: Model version if applicable
            **kwargs: Additional context (input_shape, etc.)
        """
        super().__init__(
            message,
            model_name=model_name,
            model_version=model_version,
            **kwargs
        )


# =============================================================================
# Content Generation Exceptions
# =============================================================================

class ScriptGenerationError(PipelineError):
    """
    Raised when script generation fails.

    Scenarios:
    - OpenAI API error
    - Content filtering rejection
    - Invalid game data input
    - Token limit exceeded

    Example:
        raise ScriptGenerationError(
            "Script generation failed",
            model="gpt-4o",
            game_id="748589",
            reason="content_filter"
        )
    """

    error_code = "SCRIPT_ERROR"

    def __init__(
        self,
        message: str,
        model: str,
        game_id: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize script generation error.

        Args:
            message: Error description
            model: LLM model used (e.g., "gpt-4o")
            game_id: Related game ID if applicable
            reason: Specific failure reason
            **kwargs: Additional context
        """
        super().__init__(
            message,
            model=model,
            game_id=game_id,
            reason=reason,
            **kwargs
        )


class AudioGenerationError(PipelineError):
    """
    Raised during text-to-speech generation.

    Scenarios:
    - ElevenLabs API error
    - Invalid voice ID
    - Character limit exceeded
    - Audio encoding failure

    Example:
        raise AudioGenerationError(
            "TTS generation failed",
            provider="ElevenLabs",
            voice_id="ABC123",
            character_count=5000
        )
    """

    error_code = "AUDIO_ERROR"

    def __init__(
        self,
        message: str,
        provider: str,
        voice_id: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize audio generation error.

        Args:
            message: Error description
            provider: TTS provider (e.g., "ElevenLabs")
            voice_id: Voice ID used if applicable
            **kwargs: Additional context
        """
        super().__init__(
            message,
            provider=provider,
            voice_id=voice_id,
            **kwargs
        )


class VideoGenerationError(PipelineError):
    """
    Raised during video rendering or compilation.

    Scenarios:
    - FFmpeg encoding error
    - Missing assets
    - Invalid resolution/format
    - Insufficient disk space
    - Memory allocation failure

    Example:
        raise VideoGenerationError(
            "Video encoding failed",
            step="audio_sync",
            output_path="/path/to/output.mp4",
            ffmpeg_error="Exit code 1"
        )
    """

    error_code = "VIDEO_ERROR"

    def __init__(
        self,
        message: str,
        step: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize video generation error.

        Args:
            message: Error description
            step: Pipeline step that failed (e.g., "render", "audio_sync")
            output_path: Output file path if applicable
            **kwargs: Additional context
        """
        super().__init__(
            message,
            step=step,
            output_path=output_path,
            **kwargs
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(PipelineError):
    """
    Raised for configuration issues.

    Scenarios:
    - Missing required API key
    - Invalid configuration value
    - Missing environment variable
    - Config file not found

    Example:
        raise ConfigurationError(
            "Missing required API key",
            key="OPENAI_API_KEY"
        )
    """

    error_code = "CONFIG_ERROR"

    def __init__(
        self,
        message: str,
        key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize configuration error.

        Args:
            message: Error description
            key: Configuration key that's problematic
            expected_type: Expected type for the configuration value
            **kwargs: Additional context
        """
        super().__init__(
            message,
            key=key,
            expected_type=expected_type,
            **kwargs
        )


# =============================================================================
# Upload Exceptions
# =============================================================================

class UploadError(PipelineError):
    """
    Raised when uploading to a platform fails.

    Scenarios:
    - YouTube API error
    - Authentication failure
    - Rate limiting
    - File too large
    - Network timeout

    Example:
        raise UploadError(
            "YouTube upload failed",
            platform="YouTube",
            video_path="/path/to/video.mp4",
            error_code="quotaExceeded"
        )
    """

    error_code = "UPLOAD_ERROR"

    def __init__(
        self,
        message: str,
        platform: str,
        video_path: str,
        api_error: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize upload error.

        Args:
            message: Error description
            platform: Target platform (e.g., "YouTube", "TikTok")
            video_path: Path to the video file
            api_error: Platform-specific error code/message
            **kwargs: Additional context
        """
        super().__init__(
            message,
            platform=platform,
            video_path=video_path,
            api_error=api_error,
            **kwargs
        )


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(PipelineError):
    """
    Raised when data validation fails.

    Scenarios:
    - Invalid game data structure
    - Script validation failure
    - Audio/video file validation failure
    - Input data out of range

    Example:
        raise ValidationError(
            "Invalid game data",
            field="home_score",
            value=-1,
            reason="Score must be non-negative"
        )
    """

    error_code = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        reason: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize validation error.

        Args:
            message: Error description
            field: Field that failed validation
            value: The invalid value
            reason: Specific validation failure reason
            **kwargs: Additional context
        """
        super().__init__(
            message,
            field=field,
            value=value,
            reason=reason,
            **kwargs
        )


# =============================================================================
# Budget Exceptions
# =============================================================================

class BudgetExceededError(PipelineError):
    """
    Raised when API budget limits are exceeded.

    Example:
        raise BudgetExceededError(
            "Daily budget exceeded",
            budget_type="daily",
            current=10.50,
            limit=10.00
        )
    """

    error_code = "BUDGET_ERROR"

    def __init__(
        self,
        message: str,
        budget_type: str,
        current: float,
        limit: float,
        **kwargs: Any
    ):
        """
        Initialize budget exceeded error.

        Args:
            message: Error description
            budget_type: "daily" or "monthly"
            current: Current spending
            limit: Budget limit
            **kwargs: Additional context
        """
        super().__init__(
            message,
            budget_type=budget_type,
            current=current,
            limit=limit,
            **kwargs
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def wrap_exception(
    original: Exception,
    pipeline_error_class: type,
    message: Optional[str] = None,
    **context: Any
) -> PipelineError:
    """
    Wrap a standard exception in a pipeline exception.

    Args:
        original: The original exception
        pipeline_error_class: The pipeline exception class to use
        message: Override message (defaults to str(original))
        **context: Additional context to add

    Returns:
        A new pipeline exception wrapping the original

    Example:
        try:
            requests.get(url)
        except requests.RequestException as e:
            raise wrap_exception(e, DataFetchError, source="MLB-API")
    """
    error_message = message or str(original)
    context['original_error'] = type(original).__name__
    context['original_message'] = str(original)

    return pipeline_error_class(error_message, **context)
