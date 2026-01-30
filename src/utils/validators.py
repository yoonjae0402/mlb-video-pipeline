"""
MLB Video Pipeline - Data Validators

Comprehensive validation utilities for:
- MLB game data structure
- Generated scripts
- Audio files (format, duration, etc.)
- Video files (resolution, duration, file size)

Uses Pydantic for structured data validation with clear error messages.

Usage:
    from src.utils.validators import validate_game_data, validate_script

    is_valid, errors = validate_game_data(game_dict)
    is_valid, message = validate_script(script_text)
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


# =============================================================================
# Pydantic Models for Validation
# =============================================================================

class GameData(BaseModel):
    """
    Pydantic model to validate the structure of MLB game data.

    Required fields:
    - game_id: Unique identifier for the game
    - date: Game date in YYYY-MM-DD format
    - home_team: Name of the home team
    - away_team: Name of the away team
    - home_score: Final score for home team
    - away_score: Final score for away team

    Optional fields:
    - highlights: URL or description of game highlights
    - venue: Stadium name
    - status: Game status (Final, In Progress, etc.)
    - inning: Current or final inning
    - winning_pitcher: Name of winning pitcher
    - losing_pitcher: Name of losing pitcher
    - save_pitcher: Name of save pitcher (if applicable)
    """
    game_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the game"
    )
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Game date in YYYY-MM-DD format"
    )
    home_team: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Name of the home team"
    )
    away_team: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Name of the away team"
    )
    home_score: int = Field(
        ...,
        ge=0,
        le=50,  # Reasonable max score for validation
        description="Final score for the home team"
    )
    away_score: int = Field(
        ...,
        ge=0,
        le=50,
        description="Final score for the away team"
    )

    # Optional fields
    highlights: Optional[str] = Field(
        None,
        description="URL or description of game highlights"
    )
    venue: Optional[str] = Field(
        None,
        max_length=100,
        description="Stadium name"
    )
    status: Optional[str] = Field(
        None,
        description="Game status (Final, In Progress, etc.)"
    )
    inning: Optional[int] = Field(
        None,
        ge=1,
        le=30,  # Extra innings can go long
        description="Current or final inning"
    )
    winning_pitcher: Optional[str] = Field(None)
    losing_pitcher: Optional[str] = Field(None)
    save_pitcher: Optional[str] = Field(None)

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Ensure date is valid and not in the future."""
        try:
            parsed = datetime.strptime(v, "%Y-%m-%d")
            # Allow some buffer for timezone differences
            if parsed.date() > datetime.now().date():
                raise ValueError("Game date cannot be in the future")
            return v
        except ValueError as e:
            if "Game date" in str(e):
                raise
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    @model_validator(mode='after')
    def validate_teams_different(self) -> 'GameData':
        """Ensure home and away teams are different."""
        if self.home_team.lower() == self.away_team.lower():
            raise ValueError("Home and away teams must be different")
        return self


class PlayerStats(BaseModel):
    """
    Pydantic model for validating player statistics.

    Supports both batting and pitching statistics with reasonable ranges.
    """
    player_id: int = Field(..., description="Unique identifier for the player")
    player_name: Optional[str] = Field(None, max_length=100)
    team: Optional[str] = Field(None, max_length=50)
    games_played: int = Field(..., ge=0, le=200)

    # Batting stats
    at_bats: int = Field(default=0, ge=0)
    runs: int = Field(default=0, ge=0)
    hits: int = Field(default=0, ge=0)
    doubles: int = Field(default=0, ge=0)
    triples: int = Field(default=0, ge=0)
    home_runs: int = Field(default=0, ge=0)
    rbi: int = Field(default=0, ge=0)
    walks: int = Field(default=0, ge=0)
    strikeouts: int = Field(default=0, ge=0)
    stolen_bases: int = Field(default=0, ge=0)
    batting_average: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Pitching stats
    innings_pitched: float = Field(default=0.0, ge=0.0)
    hits_allowed: int = Field(default=0, ge=0)
    runs_allowed: int = Field(default=0, ge=0)
    earned_runs: int = Field(default=0, ge=0)
    walks_allowed: int = Field(default=0, ge=0)
    strikeouts_thrown: int = Field(default=0, ge=0)
    era: Optional[float] = Field(default=None, ge=0.0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def validate_stats_consistency(self) -> 'PlayerStats':
        """Validate that statistics are logically consistent."""
        # Hits can't exceed at bats
        if self.hits > self.at_bats:
            raise ValueError("Hits cannot exceed at bats")
        # Earned runs can't exceed total runs allowed
        if self.earned_runs > self.runs_allowed:
            raise ValueError("Earned runs cannot exceed runs allowed")
        return self


class ScriptData(BaseModel):
    """Pydantic model for validating video script structure."""
    content: str = Field(..., min_length=50, max_length=10000)
    game_id: Optional[str] = Field(None)
    word_count: Optional[int] = Field(None, ge=10)
    estimated_duration_seconds: Optional[float] = Field(None, ge=5.0, le=300.0)

    @field_validator('content')
    @classmethod
    def validate_no_placeholders(cls, v: str) -> str:
        """Check for placeholder text that shouldn't be in final script."""
        placeholders = [
            ' placeholder ',
            '[INSERT',
            '[TODO',
            'REPLACE THIS',
            '[FILL IN',
            '{{',
            '}}',
        ]
        for placeholder in placeholders:
            if placeholder.lower() in v.lower():
                raise ValueError(f"Script contains placeholder text: {placeholder}")
        return v


# =============================================================================
# Validation Functions
# =============================================================================

def validate_game_data(game: dict) -> Tuple[bool, List[str]]:
    """
    Validates a dictionary containing MLB game data.

    Args:
        game: Dictionary with game data

    Returns:
        Tuple of (is_valid, list_of_errors)
        - is_valid: True if all validations pass
        - errors: List of error messages (empty if valid)

    Example:
        >>> is_valid, errors = validate_game_data({
        ...     "game_id": "748589",
        ...     "date": "2024-07-21",
        ...     "home_team": "New York Yankees",
        ...     "away_team": "Boston Red Sox",
        ...     "home_score": 5,
        ...     "away_score": 2
        ... })
        >>> print(is_valid)  # True
    """
    try:
        GameData.model_validate(game)
        return True, []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err['loc']) if err['loc'] else "root"
            errors.append(f"{loc}: {err['msg']}")
        return False, errors


def validate_player_stats(stats: dict) -> Tuple[bool, List[str]]:
    """
    Validates a dictionary of player statistics.

    Args:
        stats: Dictionary with player statistics

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        PlayerStats.model_validate(stats)
        return True, []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err['loc']) if err['loc'] else "root"
            errors.append(f"{loc}: {err['msg']}")
        return False, errors


def validate_script(
    script: str,
    min_len: int = 100,
    max_len: int = 5000,
    check_placeholders: bool = True
) -> Tuple[bool, str]:
    """
    Validates a generated video script.

    Checks:
    - Script is a non-empty string
    - Length is within acceptable range
    - No placeholder text remains
    - Basic formatting requirements

    Args:
        script: The script content to validate
        min_len: Minimum acceptable length (default: 100)
        max_len: Maximum acceptable length (default: 5000)
        check_placeholders: Whether to check for placeholder text

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if script passes all checks
        - error_message: Description of issue (empty if valid)

    Example:
        >>> is_valid, error = validate_script("This is a valid script...")
    """
    if not isinstance(script, str):
        return False, "Script must be a string"

    if not script.strip():
        return False, "Script cannot be empty"

    script_len = len(script)
    if script_len < min_len:
        return False, f"Script too short ({script_len} chars). Minimum: {min_len}"

    if script_len > max_len:
        return False, f"Script too long ({script_len} chars). Maximum: {max_len}"

    # Check for placeholder text
    if check_placeholders:
        placeholders = [
            (' placeholder ', 'placeholder text'),
            ('[insert', 'INSERT bracket'),
            ('[todo', 'TODO bracket'),
            ('replace this', 'REPLACE THIS'),
            ('[fill in', 'FILL IN bracket'),
            ('{{', 'template braces'),
            ('}}', 'template braces'),
            ('lorem ipsum', 'Lorem Ipsum'),
        ]
        script_lower = script.lower()
        for pattern, description in placeholders:
            if pattern in script_lower:
                return False, f"Script contains {description}"

    # Check for minimum word count (rough estimate of meaningful content)
    word_count = len(script.split())
    if word_count < 20:
        return False, f"Script has too few words ({word_count}). Minimum: 20"

    return True, ""


def validate_audio(
    audio_path: str,
    min_duration: float = 5.0,
    max_duration: float = 300.0,
    allowed_formats: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validates an audio file's existence and properties.

    Uses ffprobe for accurate audio analysis when available.

    Args:
        audio_path: Path to the audio file
        min_duration: Minimum required duration in seconds (default: 5)
        max_duration: Maximum allowed duration in seconds (default: 300)
        allowed_formats: List of allowed extensions (default: mp3, wav, m4a)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = validate_audio("/path/to/audio.mp3")
    """
    if allowed_formats is None:
        allowed_formats = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']

    path = Path(audio_path)

    # Check file exists
    if not path.exists():
        return False, f"Audio file not found: {audio_path}"

    # Check file extension
    if path.suffix.lower() not in allowed_formats:
        return False, f"Invalid audio format: {path.suffix}. Allowed: {allowed_formats}"

    # Check file size
    file_size = path.stat().st_size
    if file_size == 0:
        return False, "Audio file is empty"

    # Minimum size check (very small files are likely corrupted)
    if file_size < 1024:  # Less than 1KB
        return False, f"Audio file too small ({file_size} bytes). Likely corrupted"

    # Try to get duration using ffprobe
    duration = _get_media_duration(audio_path)
    if duration is not None:
        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s). Minimum: {min_duration}s"
        if duration > max_duration:
            return False, f"Audio too long ({duration:.1f}s). Maximum: {max_duration}s"

    return True, ""


def validate_video(
    video_path: str,
    min_duration: float = 5.0,
    max_duration: float = 600.0,
    min_width: int = 720,
    min_height: int = 480,
    max_file_size_mb: float = 500.0,
    allowed_formats: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validates a video file's existence and properties.

    Uses ffprobe for accurate video analysis when available.

    Args:
        video_path: Path to the video file
        min_duration: Minimum duration in seconds (default: 5)
        max_duration: Maximum duration in seconds (default: 600)
        min_width: Minimum video width in pixels (default: 720)
        min_height: Minimum video height in pixels (default: 480)
        max_file_size_mb: Maximum file size in MB (default: 500)
        allowed_formats: List of allowed extensions (default: mp4, mov, avi, mkv)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = validate_video("/path/to/video.mp4")
    """
    if allowed_formats is None:
        allowed_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

    path = Path(video_path)

    # Check file exists
    if not path.exists():
        return False, f"Video file not found: {video_path}"

    # Check file extension
    if path.suffix.lower() not in allowed_formats:
        return False, f"Invalid video format: {path.suffix}. Allowed: {allowed_formats}"

    # Check file size
    file_size = path.stat().st_size
    if file_size == 0:
        return False, "Video file is empty"

    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        return False, f"Video too large ({file_size_mb:.1f}MB). Maximum: {max_file_size_mb}MB"

    # Get video properties using ffprobe
    props = _get_video_properties(video_path)
    if props:
        duration = props.get('duration')
        width = props.get('width')
        height = props.get('height')

        if duration is not None:
            if duration < min_duration:
                return False, f"Video too short ({duration:.1f}s). Minimum: {min_duration}s"
            if duration > max_duration:
                return False, f"Video too long ({duration:.1f}s). Maximum: {max_duration}s"

        if width is not None and height is not None:
            if width < min_width:
                return False, f"Video width too small ({width}px). Minimum: {min_width}px"
            if height < min_height:
                return False, f"Video height too small ({height}px). Minimum: {min_height}px"

    return True, ""


# =============================================================================
# Helper Functions
# =============================================================================

def _get_media_duration(file_path: str) -> Optional[float]:
    """
    Get media duration using ffprobe.

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'json',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = data.get('format', {}).get('duration')
            if duration:
                return float(duration)
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, FileNotFoundError):
        pass
    return None


def _get_video_properties(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get video properties using ffprobe.

    Returns:
        Dictionary with duration, width, height, or None if unable to determine
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration:stream=width,height',
            '-select_streams', 'v:0',
            '-of', 'json',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            props = {}

            # Get duration
            duration = data.get('format', {}).get('duration')
            if duration:
                props['duration'] = float(duration)

            # Get resolution from first video stream
            streams = data.get('streams', [])
            if streams:
                props['width'] = streams[0].get('width')
                props['height'] = streams[0].get('height')

            return props if props else None
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, FileNotFoundError):
        pass
    return None


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitizes a string to be used as a valid filename.

    - Replaces spaces with underscores
    - Removes invalid characters
    - Limits length
    - Handles edge cases

    Args:
        filename: Input string
        max_length: Maximum filename length (default: 255)

    Returns:
        Sanitized filename safe for most filesystems

    Example:
        >>> sanitize_filename("Game: Yankees vs Red Sox (2024)")
        'Game_Yankees_vs_Red_Sox_2024'
    """
    if not isinstance(filename, str):
        return ""

    # Replace spaces with underscores
    sanitized = filename.replace(" ", "_")

    # Remove invalid characters (keep alphanumeric, underscore, hyphen, dot)
    sanitized = re.sub(r'[^\w\-\.]', '', sanitized)

    # Remove consecutive underscores/hyphens
    sanitized = re.sub(r'[_\-]+', '_', sanitized)

    # Remove leading/trailing underscores/hyphens/dots
    sanitized = sanitized.strip('._-')

    # Ensure we don't have an empty result
    if not sanitized:
        sanitized = "unnamed"

    # Limit length (preserve extension if present)
    if len(sanitized) > max_length:
        # Check for extension
        parts = sanitized.rsplit('.', 1)
        if len(parts) == 2 and len(parts[1]) <= 10:
            ext = '.' + parts[1]
            name = parts[0][:max_length - len(ext)]
            sanitized = name + ext
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def validate_api_response(
    response: dict,
    required_fields: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate an API response has required fields.

    Args:
        response: API response dictionary
        required_fields: List of required field names (supports dot notation)

    Returns:
        Tuple of (is_valid, list_of_missing_fields)

    Example:
        >>> is_valid, missing = validate_api_response(
        ...     {"game": {"id": 123}},
        ...     ["game.id", "game.date"]
        ... )
    """
    missing = []

    for field in required_fields:
        parts = field.split('.')
        value = response
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                missing.append(field)
                break

    return len(missing) == 0, missing


def validate_date_range(
    start_date: str,
    end_date: str,
    max_days: int = 365
) -> Tuple[bool, str]:
    """
    Validate a date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_days: Maximum allowed days in range (default: 365)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"

    if end < start:
        return False, "End date must be after start date"

    days = (end - start).days
    if days > max_days:
        return False, f"Date range too large ({days} days). Maximum: {max_days}"

    return True, ""
