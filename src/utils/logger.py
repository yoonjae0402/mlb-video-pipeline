"""
MLB Video Pipeline - Logging System

Configurable logging with:
- Rotating file handlers (JSON format for parsing)
- Rich console output for development
- Environment-based log levels (development vs production)
- Context-aware logging with extra fields
- Thread-safe implementation

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Fetching MLB games for 2024-07-04")
    logger.error("API call failed", extra={"game_id": 12345, "error": str(e)})
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional, Any
from contextlib import contextmanager

# Optional rich import for pretty console output
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
DEFAULT_LOG_LEVEL = "INFO"
MAX_BYTES = 10 * 1024 * 1024  # 10MB per log file
BACKUP_COUNT = 5  # Keep 5 backup files


# =============================================================================
# JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.

    Includes:
    - ISO 8601 timestamps with timezone
    - Module context (name, function, line)
    - Exception info if present
    - Any extra fields passed to the log call
    """

    # Fields that are part of the standard LogRecord and should be excluded from extras
    STANDARD_FIELDS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName', 'levelname',
        'levelno', 'lineno', 'module', 'msecs', 'pathname', 'process',
        'processName', 'relativeCreated', 'stack_info', 'exc_info', 'exc_text',
        'thread', 'threadName', 'taskName', 'message'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        # Build the base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_FIELDS and not key.startswith('_'):
                # Ensure value is JSON serializable
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class SimpleFormatter(logging.Formatter):
    """Simple text formatter for console output when Rich is not available."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# =============================================================================
# Logger Factory
# =============================================================================

# Global lock for thread-safe logger creation
_logger_lock = Lock()
_configured_loggers: set = set()


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    log_file: str = "pipeline.log",
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """
    Get or create a configured logger instance.

    Features:
    - Rotating file handler with JSON formatting
    - Console handler with Rich formatting (if available)
    - Thread-safe singleton pattern per logger name
    - Environment-based log levels

    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Falls back to LOG_LEVEL env var, then "INFO"
        log_dir: Directory for log files (default: project/logs)
        log_file: Name of the log file (default: pipeline.log)
        enable_console: Whether to log to console
        enable_file: Whether to log to file

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing game", extra={"game_id": 12345})
    """
    with _logger_lock:
        logger = logging.getLogger(name)

        # If already configured, return existing logger
        if name in _configured_loggers:
            return logger

        # Determine log level from parameter, env, or default
        if level is None:
            level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
        else:
            level = level.upper()

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            level = DEFAULT_LOG_LEVEL

        logger.setLevel(level)
        logger.propagate = False  # Prevent duplicate logs

        # Skip if handlers already exist
        if logger.hasHandlers():
            _configured_loggers.add(name)
            return logger

        # --- File Handler (JSON) ---
        if enable_file:
            log_path = log_dir or DEFAULT_LOG_DIR
            log_path = Path(log_path)
            log_path.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path / log_file,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)

        # --- Console Handler ---
        if enable_console:
            if RICH_AVAILABLE and os.environ.get("ENVIRONMENT") != "production":
                console_handler = RichHandler(
                    rich_tracebacks=True,
                    show_time=True,
                    show_path=True,
                )
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(SimpleFormatter())

            console_handler.setLevel(level)
            logger.addHandler(console_handler)

        _configured_loggers.add(name)
        return logger


# =============================================================================
# Cost Logger
# =============================================================================

class CostLogFormatter(logging.Formatter):
    """Formatter specifically for cost tracking entries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format cost log entry as JSON."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": getattr(record, 'service', 'unknown'),
            "model": getattr(record, 'model', 'unknown'),
            "cost_usd": getattr(record, 'cost_usd', 0.0),
            "usage": getattr(record, 'usage', {}),
            "details": getattr(record, 'details', ''),
        }
        return json.dumps(log_entry, default=str)


def get_cost_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get a dedicated logger for tracking API costs.

    Logs to a separate file (costs.jsonl) with minimal console output.
    Each line is a valid JSON object for easy parsing.

    Args:
        log_dir: Directory for log files (default: project/logs)

    Returns:
        Logger configured for cost tracking
    """
    with _logger_lock:
        logger = logging.getLogger("cost_tracker")

        if "cost_tracker" in _configured_loggers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.hasHandlers():
            _configured_loggers.add("cost_tracker")
            return logger

        log_path = log_dir or DEFAULT_LOG_DIR
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        # Use .jsonl extension for JSON Lines format
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "costs.jsonl",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(CostLogFormatter())
        logger.addHandler(file_handler)

        _configured_loggers.add("cost_tracker")
        return logger


# =============================================================================
# Context Manager for Extra Fields
# =============================================================================

class LogContext:
    """
    Context manager for adding extra fields to all log messages within a block.

    Usage:
        with LogContext(logger, game_id="12345", date="2024-07-04"):
            logger.info("Processing game")  # Includes game_id and date
            logger.info("Game processed")   # Includes game_id and date
    """

    def __init__(self, logger: logging.Logger, **extra_fields: Any):
        self.logger = logger
        self.extra_fields = extra_fields
        self._original_factory = None

    def __enter__(self):
        self._original_factory = logging.getLogRecordFactory()
        extra = self.extra_fields

        def record_factory(*args, **kwargs):
            record = self._original_factory(*args, **kwargs)
            for key, value in extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self._original_factory)
        return False


@contextmanager
def log_context(logger: logging.Logger, **extra_fields: Any):
    """
    Context manager decorator for adding extra fields to log messages.

    Usage:
        with log_context(logger, game_id="12345"):
            logger.info("Processing game")
    """
    ctx = LogContext(logger, **extra_fields)
    try:
        yield ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


# =============================================================================
# Utility Functions
# =============================================================================

def set_global_level(level: str) -> None:
    """
    Set log level for all configured loggers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = level.upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}")

    for name in _configured_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def get_log_file_path(log_file: str = "pipeline.log") -> Path:
    """Get the full path to a log file."""
    return DEFAULT_LOG_DIR / log_file


# =============================================================================
# Module-level convenience logger
# =============================================================================

# Create a module-level logger for quick access
_root_logger = None


def log() -> logging.Logger:
    """Get the root pipeline logger."""
    global _root_logger
    if _root_logger is None:
        _root_logger = get_logger("mlb_pipeline")
    return _root_logger
