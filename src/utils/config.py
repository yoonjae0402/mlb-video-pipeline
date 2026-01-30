"""
MLB Video Pipeline - Configuration Utilities

Convenience wrapper around config.settings with additional helpers:
- Environment detection (development vs production)
- Configuration validation
- Feature flag helpers
- Path utilities

Usage:
    from src.utils.config import get_config, is_production, require_key

    config = get_config()
    print(config.openai_api_key)

    # Require a key or raise an error
    api_key = require_key("OPENAI_API_KEY")

    # Check environment
    if is_production():
        # Use production settings
        pass
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union
from functools import lru_cache

from .exceptions import ConfigurationError


# Try to import settings, fall back to manual loading if not available
try:
    from config.settings import Settings, get_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False


T = TypeVar('T')


# =============================================================================
# Environment Detection
# =============================================================================

def get_environment() -> str:
    """
    Get the current environment name.

    Checks ENVIRONMENT env var, defaults to "development".

    Returns:
        Environment name: "development", "production", "testing"
    """
    return os.environ.get("ENVIRONMENT", "development").lower()


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() in ("development", "dev", "local")


def is_testing() -> bool:
    """Check if running in test environment."""
    env = get_environment()
    return env in ("testing", "test") or "pytest" in os.environ.get("_", "")


# =============================================================================
# Configuration Access
# =============================================================================

@lru_cache()
def get_config():
    """
    Get the configuration settings instance.

    Returns the Pydantic Settings object with all configuration values.
    Cached for performance.

    Returns:
        Settings instance

    Raises:
        ConfigurationError: If settings module is not available
    """
    if not SETTINGS_AVAILABLE:
        raise ConfigurationError(
            "Settings module not available. Ensure config/settings.py exists.",
            key="config.settings"
        )
    return get_settings()


def get_env(
    key: str,
    default: Optional[T] = None,
    cast: Optional[type] = None
) -> Union[str, T]:
    """
    Get an environment variable with optional type casting.

    Args:
        key: Environment variable name
        default: Default value if not set
        cast: Type to cast to (int, bool, float, etc.)

    Returns:
        Environment variable value or default

    Example:
        >>> port = get_env("PORT", 8000, cast=int)
        >>> debug = get_env("DEBUG", False, cast=bool)
    """
    value = os.environ.get(key)

    if value is None:
        return default

    if cast is None:
        return value

    # Handle boolean casting specially
    if cast is bool:
        return value.lower() in ("true", "1", "yes", "on")

    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def require_key(key: str, env_var: Optional[str] = None) -> str:
    """
    Require a configuration key to be set.

    Checks both the Settings object and environment variables.

    Args:
        key: Configuration key name (e.g., "openai_api_key")
        env_var: Override environment variable name

    Returns:
        The configuration value

    Raises:
        ConfigurationError: If the key is not set

    Example:
        >>> api_key = require_key("openai_api_key")
        >>> # Or check env var directly
        >>> api_key = require_key("OPENAI_API_KEY", env_var="OPENAI_API_KEY")
    """
    # Try settings first
    if SETTINGS_AVAILABLE:
        config = get_config()
        value = getattr(config, key, None)
        if value:
            return value

    # Try environment variable
    env_key = env_var or key.upper()
    value = os.environ.get(env_key)
    if value:
        return value

    raise ConfigurationError(
        f"Required configuration key not set: {key}",
        key=key,
        env_var=env_key
    )


def require_keys(keys: List[str]) -> Dict[str, str]:
    """
    Require multiple configuration keys to be set.

    Args:
        keys: List of configuration key names

    Returns:
        Dictionary of key -> value

    Raises:
        ConfigurationError: If any key is not set (lists all missing)
    """
    values = {}
    missing = []

    for key in keys:
        try:
            values[key] = require_key(key)
        except ConfigurationError:
            missing.append(key)

    if missing:
        raise ConfigurationError(
            f"Missing required configuration keys: {', '.join(missing)}",
            missing_keys=missing
        )

    return values


# =============================================================================
# Feature Flags
# =============================================================================

def is_feature_enabled(feature: str) -> bool:
    """
    Check if a feature flag is enabled.

    Checks Settings first, then environment variable ENABLE_{FEATURE}.

    Args:
        feature: Feature name (e.g., "cost_tracking")

    Returns:
        True if feature is enabled
    """
    # Try settings
    if SETTINGS_AVAILABLE:
        config = get_config()
        attr_name = f"enable_{feature}"
        value = getattr(config, attr_name, None)
        if value is not None:
            return bool(value)

    # Try environment variable
    env_key = f"ENABLE_{feature.upper()}"
    env_value = os.environ.get(env_key, "").lower()
    return env_value in ("true", "1", "yes", "on")


def is_dry_run() -> bool:
    """Check if running in dry run mode (no actual API calls)."""
    if SETTINGS_AVAILABLE:
        return get_config().dry_run
    return get_env("DRY_RUN", False, cast=bool)


# =============================================================================
# Path Utilities
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    if SETTINGS_AVAILABLE:
        return get_config().base_dir
    # Fall back to finding the directory containing src/
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / "src").exists() and (current / "config").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_data_dir() -> Path:
    """Get the data directory path."""
    if SETTINGS_AVAILABLE:
        return get_config().data_dir
    return get_project_root() / "data"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    if SETTINGS_AVAILABLE:
        return get_config().logs_dir
    return get_project_root() / "logs"


def get_outputs_dir() -> Path:
    """Get the outputs directory path."""
    if SETTINGS_AVAILABLE:
        return get_config().outputs_dir
    return get_project_root() / "outputs"


def ensure_directories() -> None:
    """
    Create all required directories if they don't exist.

    Creates:
    - data/raw, data/processed, data/cache
    - outputs/videos, outputs/audio, outputs/scripts, outputs/thumbnails
    - logs/
    - models/training_data
    """
    if SETTINGS_AVAILABLE:
        get_config().ensure_directories()
        return

    # Manual directory creation
    directories = [
        get_data_dir() / "raw",
        get_data_dir() / "processed",
        get_data_dir() / "cache",
        get_outputs_dir() / "videos",
        get_outputs_dir() / "audio",
        get_outputs_dir() / "scripts",
        get_outputs_dir() / "thumbnails",
        get_logs_dir(),
        get_project_root() / "models" / "training_data",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API Configuration
# =============================================================================

def get_api_config(service: str) -> Dict[str, Any]:
    """
    Get API configuration for a specific service.

    Args:
        service: Service name (openai, elevenlabs, youtube, aws)

    Returns:
        Dictionary with service configuration

    Example:
        >>> config = get_api_config("openai")
        >>> print(config["api_key"])
    """
    service = service.lower()

    if service == "openai":
        return {
            "api_key": get_env("OPENAI_API_KEY"),
            "max_tokens": get_env("OPENAI_MAX_TOKENS", 500, cast=int),
            "model": get_env("OPENAI_MODEL", "gpt-4o"),
        }

    elif service == "elevenlabs":
        return {
            "api_key": get_env("ELEVENLABS_API_KEY"),
            "voice_id": get_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            "monthly_limit": get_env("ELEVENLABS_MONTHLY_CHARS", 100000, cast=int),
        }

    elif service == "youtube":
        return {
            "api_key": get_env("YOUTUBE_API_KEY"),
            "client_id": get_env("YOUTUBE_CLIENT_ID"),
            "client_secret": get_env("YOUTUBE_CLIENT_SECRET"),
        }

    elif service == "aws":
        return {
            "access_key_id": get_env("AWS_ACCESS_KEY_ID"),
            "secret_access_key": get_env("AWS_SECRET_ACCESS_KEY"),
            "region": get_env("AWS_REGION", "us-east-1"),
            "s3_bucket": get_env("S3_BUCKET_NAME", "mlb-video-pipeline"),
        }

    else:
        return {}


def validate_api_keys(services: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Validate that API keys are configured for specified services.

    Args:
        services: List of services to check (default: all)

    Returns:
        Dictionary of service -> is_configured

    Example:
        >>> status = validate_api_keys(["openai", "elevenlabs"])
        >>> if not status["openai"]:
        ...     print("OpenAI API key not configured")
    """
    all_services = ["openai", "elevenlabs", "youtube", "aws"]
    services = services or all_services

    status = {}
    for service in services:
        config = get_api_config(service)
        # Check if primary key exists
        if service == "aws":
            status[service] = bool(
                config.get("access_key_id") and config.get("secret_access_key")
            )
        else:
            status[service] = bool(config.get("api_key"))

    return status


# =============================================================================
# Configuration Summary
# =============================================================================

def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration (safe to log).

    Returns:
        Dictionary with configuration overview (no secrets)
    """
    api_status = validate_api_keys()

    return {
        "environment": get_environment(),
        "project_root": str(get_project_root()),
        "api_keys_configured": api_status,
        "features": {
            "cost_tracking": is_feature_enabled("cost_tracking"),
            "youtube_upload": is_feature_enabled("youtube_upload"),
            "dry_run": is_dry_run(),
        },
        "paths": {
            "data": str(get_data_dir()),
            "logs": str(get_logs_dir()),
            "outputs": str(get_outputs_dir()),
        }
    }
