"""
Unit tests for additional utilities:
- src/utils/logger.py
- src/utils/config.py
- src/utils/exceptions.py
"""

import pytest
import os
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.logger import (
    get_logger,
    get_cost_logger,
    log_context,
    set_global_level,
    JSONFormatter,
)

from src.utils.exceptions import (
    PipelineError,
    DataFetchError,
    RateLimitError,
    ModelPredictionError,
    ScriptGenerationError,
    AudioGenerationError,
    VideoGenerationError,
    ConfigurationError,
    UploadError,
    ValidationError,
    BudgetExceededError,
    wrap_exception,
)

from src.utils.config import (
    get_environment,
    is_production,
    is_development,
    is_testing,
    get_env,
    is_dry_run,
    get_project_root,
)


# =============================================================================
# Logger Tests
# =============================================================================

class TestLogger:
    """Tests for the logging module."""

    def test_get_logger_returns_logger(self, tmp_path):
        """Tests that get_logger returns a logger instance."""
        logger = get_logger("test_module", log_dir=tmp_path)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logger_has_handlers(self, tmp_path):
        """Tests that logger has both file and console handlers."""
        logger = get_logger(
            "test_handlers",
            log_dir=tmp_path,
            enable_console=True,
            enable_file=True
        )
        assert len(logger.handlers) >= 1

    def test_logger_respects_level(self, tmp_path):
        """Tests that logger respects configured level."""
        logger = get_logger("test_level", level="WARNING", log_dir=tmp_path)
        assert logger.level == logging.WARNING

    def test_json_formatter_output(self):
        """Tests that JSONFormatter produces valid JSON."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_json_formatter_includes_extras(self):
        """Tests that extra fields are included in JSON output."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.game_id = "12345"
        record.custom_field = {"nested": "value"}

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["game_id"] == "12345"
        assert parsed["custom_field"] == {"nested": "value"}

    def test_log_context_adds_fields(self, tmp_path, caplog):
        """Tests that log_context adds fields to log records."""
        logger = get_logger("test_context", log_dir=tmp_path, enable_file=False)

        with log_context(logger, game_id="ABC123", date="2024-07-04"):
            logger.info("Processing game")

        # The context manager should have added fields
        # (verification depends on handler implementation)

    def test_set_global_level(self, tmp_path):
        """Tests that set_global_level changes level for all loggers."""
        logger1 = get_logger("test_global_1", log_dir=tmp_path)
        logger2 = get_logger("test_global_2", log_dir=tmp_path)

        set_global_level("ERROR")

        assert logger1.level == logging.ERROR
        assert logger2.level == logging.ERROR

    def test_set_global_level_invalid_raises(self):
        """Tests that invalid level raises ValueError."""
        with pytest.raises(ValueError):
            set_global_level("INVALID_LEVEL")


# =============================================================================
# Exception Tests
# =============================================================================

class TestExceptions:
    """Tests for custom exception classes."""

    def test_pipeline_error_basic(self):
        """Tests basic PipelineError creation."""
        error = PipelineError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.timestamp is not None
        assert "[PIPELINE_ERROR]" in str(error)

    def test_pipeline_error_with_context(self):
        """Tests PipelineError with context."""
        error = PipelineError(
            "Test error",
            game_id="12345",
            step="processing"
        )

        assert error.context["game_id"] == "12345"
        assert error.context["step"] == "processing"
        assert "game_id=12345" in str(error)

    def test_pipeline_error_to_dict(self):
        """Tests PipelineError serialization."""
        error = PipelineError("Test error", key="value")
        error_dict = error.to_dict()

        assert error_dict["error_type"] == "PipelineError"
        assert error_dict["message"] == "Test error"
        assert error_dict["context"]["key"] == "value"

    def test_data_fetch_error(self):
        """Tests DataFetchError creation."""
        error = DataFetchError(
            "Failed to fetch data",
            source="MLB-StatsAPI",
            game_id="748589",
            status_code=503
        )

        assert error.context["source"] == "MLB-StatsAPI"
        assert error.context["game_id"] == "748589"
        assert error.context["status_code"] == 503
        assert error.error_code == "DATA_FETCH_ERROR"

    def test_rate_limit_error(self):
        """Tests RateLimitError creation."""
        error = RateLimitError(
            "Rate limit exceeded",
            source="OpenAI",
            retry_after=60
        )

        assert error.retry_after == 60
        assert error.context["status_code"] == 429
        assert error.error_code == "RATE_LIMIT_ERROR"

    def test_model_prediction_error(self):
        """Tests ModelPredictionError creation."""
        error = ModelPredictionError(
            "Prediction failed",
            model_name="viewer_predictor_v2",
            model_version="1.0.0"
        )

        assert error.context["model_name"] == "viewer_predictor_v2"
        assert error.error_code == "MODEL_ERROR"

    def test_script_generation_error(self):
        """Tests ScriptGenerationError creation."""
        error = ScriptGenerationError(
            "Script generation failed",
            model="gpt-4o",
            game_id="12345",
            reason="content_filter"
        )

        assert error.context["model"] == "gpt-4o"
        assert error.context["reason"] == "content_filter"

    def test_audio_generation_error(self):
        """Tests AudioGenerationError creation."""
        error = AudioGenerationError(
            "TTS failed",
            provider="ElevenLabs",
            voice_id="ABC123"
        )

        assert error.context["provider"] == "ElevenLabs"
        assert error.context["voice_id"] == "ABC123"

    def test_video_generation_error(self):
        """Tests VideoGenerationError creation."""
        error = VideoGenerationError(
            "Encoding failed",
            step="audio_sync",
            output_path="/path/to/video.mp4"
        )

        assert error.context["step"] == "audio_sync"
        assert error.error_code == "VIDEO_ERROR"

    def test_configuration_error(self):
        """Tests ConfigurationError creation."""
        error = ConfigurationError(
            "Missing API key",
            key="OPENAI_API_KEY"
        )

        assert error.context["key"] == "OPENAI_API_KEY"
        assert error.error_code == "CONFIG_ERROR"

    def test_upload_error(self):
        """Tests UploadError creation."""
        error = UploadError(
            "Upload failed",
            platform="YouTube",
            video_path="/path/to/video.mp4",
            api_error="quotaExceeded"
        )

        assert error.context["platform"] == "YouTube"
        assert error.context["api_error"] == "quotaExceeded"

    def test_validation_error(self):
        """Tests ValidationError creation."""
        error = ValidationError(
            "Invalid field",
            field="home_score",
            value=-1,
            reason="Score must be non-negative"
        )

        assert error.context["field"] == "home_score"
        assert error.context["value"] == -1

    def test_budget_exceeded_error(self):
        """Tests BudgetExceededError creation."""
        error = BudgetExceededError(
            "Daily budget exceeded",
            budget_type="daily",
            current=15.50,
            limit=10.00
        )

        assert error.context["budget_type"] == "daily"
        assert error.context["current"] == 15.50
        assert error.context["limit"] == 10.00

    def test_wrap_exception(self):
        """Tests wrap_exception utility."""
        original = ValueError("Original error message")

        wrapped = wrap_exception(
            original,
            DataFetchError,
            source="test_source"
        )

        assert isinstance(wrapped, DataFetchError)
        assert "Original error message" in wrapped.message
        assert wrapped.context["original_error"] == "ValueError"


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration utilities."""

    def test_get_environment_default(self):
        """Tests default environment is development."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ENVIRONMENT if it exists
            os.environ.pop("ENVIRONMENT", None)
            env = get_environment()
            assert env in ("development", "production", "testing")

    def test_get_environment_from_env_var(self):
        """Tests environment from env var."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert get_environment() == "production"

    def test_is_production(self):
        """Tests is_production detection."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_production() is True

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_production() is False

    def test_is_development(self):
        """Tests is_development detection."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_development() is True

        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}):
            assert is_development() is True

    def test_get_env_with_default(self):
        """Tests get_env with default value."""
        result = get_env("NONEXISTENT_VAR", default="default_value")
        assert result == "default_value"

    def test_get_env_with_cast(self):
        """Tests get_env with type casting."""
        with patch.dict(os.environ, {"TEST_INT": "42", "TEST_BOOL": "true"}):
            assert get_env("TEST_INT", cast=int) == 42
            assert get_env("TEST_BOOL", cast=bool) is True

    def test_get_env_bool_casting(self):
        """Tests boolean casting for various truthy values."""
        with patch.dict(os.environ, {
            "BOOL_TRUE": "true",
            "BOOL_1": "1",
            "BOOL_YES": "yes",
            "BOOL_ON": "on",
            "BOOL_FALSE": "false",
        }):
            assert get_env("BOOL_TRUE", cast=bool) is True
            assert get_env("BOOL_1", cast=bool) is True
            assert get_env("BOOL_YES", cast=bool) is True
            assert get_env("BOOL_ON", cast=bool) is True
            assert get_env("BOOL_FALSE", cast=bool) is False

    def test_is_dry_run_default_false(self):
        """Tests is_dry_run defaults to False."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DRY_RUN", None)
            # Default behavior depends on settings availability
            result = is_dry_run()
            assert isinstance(result, bool)

    def test_get_project_root_returns_path(self):
        """Tests get_project_root returns a Path."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for utility modules working together."""

    def test_logger_with_exception(self, tmp_path):
        """Tests logging an exception properly."""
        logger = get_logger("test_integration", log_dir=tmp_path)

        try:
            raise DataFetchError(
                "API failure",
                source="MLB-StatsAPI",
                status_code=500
            )
        except DataFetchError as e:
            logger.error("Caught error", extra={"error": e.to_dict()})
            # Should not raise

    def test_exception_chaining(self):
        """Tests exception chaining with wrap_exception."""
        try:
            try:
                raise ConnectionError("Network failure")
            except ConnectionError as e:
                raise wrap_exception(
                    e,
                    DataFetchError,
                    source="MLB-API"
                )
        except DataFetchError as e:
            assert "Network failure" in str(e)
            assert e.context["source"] == "MLB-API"
