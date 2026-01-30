"""
MLB Video Pipeline - Utility Modules

Core utilities used across all pipeline modules:
- logger: Logging with JSON format and rotation
- validators: Data validation for games, scripts, audio, video
- cost_tracker: API usage and cost monitoring
- exceptions: Custom exception classes
- config: Configuration loading and validation

Quick imports:
    from src.utils import get_logger, validate_game_data, CostTracker
"""

# Logger exports
from .logger import (
    get_logger,
    get_cost_logger,
    log_context,
    set_global_level,
)

# Validator exports
from .validators import (
    validate_game_data,
    validate_player_stats,
    validate_script,
    validate_audio,
    validate_video,
    validate_api_response,
    validate_date_range,
    sanitize_filename,
    GameData,
    PlayerStats,
)

# Cost tracker exports
from .cost_tracker import (
    CostTracker,
    get_cost_tracker,
    estimate_openai_cost,
    estimate_elevenlabs_cost,
)

# Exception exports
from .exceptions import (
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

# Config exports
from .config import (
    get_config,
    get_env,
    require_key,
    require_keys,
    is_production,
    is_development,
    is_testing,
    is_feature_enabled,
    is_dry_run,
    get_project_root,
    get_data_dir,
    get_logs_dir,
    get_outputs_dir,
    ensure_directories,
    get_api_config,
    validate_api_keys,
    get_config_summary,
)


__all__ = [
    # Logger
    "get_logger",
    "get_cost_logger",
    "log_context",
    "set_global_level",
    # Validators
    "validate_game_data",
    "validate_player_stats",
    "validate_script",
    "validate_audio",
    "validate_video",
    "validate_api_response",
    "validate_date_range",
    "sanitize_filename",
    "GameData",
    "PlayerStats",
    # Cost tracker
    "CostTracker",
    "get_cost_tracker",
    "estimate_openai_cost",
    "estimate_elevenlabs_cost",
    # Exceptions
    "PipelineError",
    "DataFetchError",
    "RateLimitError",
    "ModelPredictionError",
    "ScriptGenerationError",
    "AudioGenerationError",
    "VideoGenerationError",
    "ConfigurationError",
    "UploadError",
    "ValidationError",
    "BudgetExceededError",
    "wrap_exception",
    # Config
    "get_config",
    "get_env",
    "require_key",
    "require_keys",
    "is_production",
    "is_development",
    "is_testing",
    "is_feature_enabled",
    "is_dry_run",
    "get_project_root",
    "get_data_dir",
    "get_logs_dir",
    "get_outputs_dir",
    "ensure_directories",
    "get_api_config",
    "validate_api_keys",
    "get_config_summary",
]
