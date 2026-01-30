"""
Unit tests for src/utils/validators.py

Tests validation functions for:
- Game data validation
- Script validation
- Audio file validation
- Video file validation
- Player stats validation
- Utility functions
"""

import pytest
import os
from pathlib import Path
from datetime import datetime, timedelta

from src.utils.validators import (
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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_game_data():
    """Returns a dictionary with valid game data."""
    # Use yesterday's date to avoid future date validation issues
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "game_id": "748589",
        "date": yesterday,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_score": 5,
        "away_score": 2,
        "highlights": "https://example.com/highlights.mp4"
    }


@pytest.fixture
def valid_player_stats():
    """Returns a dictionary with valid player statistics."""
    return {
        "player_id": 660271,
        "player_name": "Shohei Ohtani",
        "team": "Los Angeles Dodgers",
        "games_played": 150,
        "at_bats": 500,
        "runs": 100,
        "hits": 175,
        "doubles": 35,
        "triples": 5,
        "home_runs": 45,
        "rbi": 100,
        "walks": 80,
        "strikeouts": 120,
        "stolen_bases": 30,
        "batting_average": 0.350,
    }


@pytest.fixture
def temp_file(tmp_path):
    """Creates a temporary file and returns its path."""
    def _create_file(filename: str, content: str = "", size: int = None):
        file_path = tmp_path / filename
        if size is not None:
            # Create file with specific size
            file_path.write_bytes(b'x' * size)
        else:
            file_path.write_text(content)
        return str(file_path)
    return _create_file


# =============================================================================
# Tests for validate_game_data
# =============================================================================

class TestValidateGameData:
    """Tests for the validate_game_data function."""

    def test_valid_game_data_passes(self, valid_game_data):
        """Tests that valid game data passes validation."""
        is_valid, errors = validate_game_data(valid_game_data)
        assert is_valid
        assert not errors

    def test_missing_required_field_fails(self, valid_game_data):
        """Tests that missing a required field fails validation."""
        del valid_game_data["game_id"]
        is_valid, errors = validate_game_data(valid_game_data)
        assert not is_valid
        assert any("game_id" in e for e in errors)

    def test_invalid_date_format_fails(self, valid_game_data):
        """Tests that an invalid date format fails validation."""
        valid_game_data["date"] = "2024/07/21"
        is_valid, errors = validate_game_data(valid_game_data)
        assert not is_valid
        assert any("date" in e for e in errors)

    def test_negative_score_fails(self, valid_game_data):
        """Tests that a negative score fails validation."""
        valid_game_data["home_score"] = -1
        is_valid, errors = validate_game_data(valid_game_data)
        assert not is_valid
        assert any("home_score" in e for e in errors)

    def test_same_team_fails(self, valid_game_data):
        """Tests that home and away being same team fails."""
        valid_game_data["away_team"] = "New York Yankees"
        is_valid, errors = validate_game_data(valid_game_data)
        assert not is_valid
        assert any("different" in e.lower() for e in errors)

    def test_optional_fields_allowed_missing(self, valid_game_data):
        """Tests that optional fields can be missing."""
        del valid_game_data["highlights"]
        is_valid, errors = validate_game_data(valid_game_data)
        assert is_valid
        assert not errors

    def test_extra_fields_allowed(self, valid_game_data):
        """Tests that extra fields don't cause validation failure."""
        valid_game_data["extra_field"] = "some value"
        is_valid, errors = validate_game_data(valid_game_data)
        assert is_valid


class TestValidatePlayerStats:
    """Tests for the validate_player_stats function."""

    def test_valid_stats_passes(self, valid_player_stats):
        """Tests that valid player stats pass validation."""
        is_valid, errors = validate_player_stats(valid_player_stats)
        assert is_valid
        assert not errors

    def test_missing_player_id_fails(self, valid_player_stats):
        """Tests that missing player_id fails."""
        del valid_player_stats["player_id"]
        is_valid, errors = validate_player_stats(valid_player_stats)
        assert not is_valid
        assert any("player_id" in e for e in errors)

    def test_hits_exceed_at_bats_fails(self, valid_player_stats):
        """Tests that hits > at_bats fails consistency check."""
        valid_player_stats["hits"] = 600  # More than at_bats (500)
        is_valid, errors = validate_player_stats(valid_player_stats)
        assert not is_valid
        assert any("exceed" in e.lower() for e in errors)

    def test_negative_stat_fails(self, valid_player_stats):
        """Tests that negative stats fail validation."""
        valid_player_stats["home_runs"] = -5
        is_valid, errors = validate_player_stats(valid_player_stats)
        assert not is_valid


# =============================================================================
# Tests for validate_script
# =============================================================================

class TestValidateScript:
    """Tests for the validate_script function."""

    def test_valid_script_passes(self):
        """Tests a valid script."""
        script = "This is a valid script with plenty of content. " * 10
        is_valid, message = validate_script(script)
        assert is_valid
        assert message == ""

    def test_too_short_fails(self):
        """Tests a script that is too short."""
        script = "too short"
        is_valid, message = validate_script(script)
        assert not is_valid
        assert "too short" in message.lower()

    def test_too_long_fails(self):
        """Tests a script that is too long."""
        script = "a" * 6000
        is_valid, message = validate_script(script, max_len=5500)
        assert not is_valid
        assert "too long" in message.lower()

    def test_placeholder_text_fails(self):
        """Tests a script with placeholder text."""
        script = "This is a long script that contains placeholder text that should fail. " * 5
        is_valid, message = validate_script(script)
        assert not is_valid
        assert "placeholder" in message.lower()

    def test_insert_bracket_fails(self):
        """Tests script with [INSERT] placeholder."""
        script = "Welcome to today's game [INSERT TEAM NAME HERE] versus the Yankees. " * 5
        is_valid, message = validate_script(script)
        assert not is_valid
        assert "INSERT" in message

    def test_template_braces_fails(self):
        """Tests script with {{ }} template markers."""
        script = "Today's game features {{home_team}} vs {{away_team}}. " * 10
        is_valid, message = validate_script(script)
        assert not is_valid

    def test_empty_script_fails(self):
        """Tests that empty script fails."""
        is_valid, message = validate_script("")
        assert not is_valid
        assert "empty" in message.lower()

    def test_non_string_fails(self):
        """Tests that non-string input fails."""
        is_valid, message = validate_script(12345)
        assert not is_valid
        assert "string" in message.lower()

    def test_custom_length_limits(self):
        """Tests custom min/max length parameters."""
        # Script with enough words to pass word count check
        script = "This is a valid test script with many words to pass all validation checks. " * 3
        is_valid, _ = validate_script(script, min_len=10, max_len=500)
        assert is_valid

        short_script = "word " * 25  # 25 words but short
        is_valid, message = validate_script(short_script, min_len=500)
        assert not is_valid


# =============================================================================
# Tests for validate_audio
# =============================================================================

class TestValidateAudio:
    """Tests for the validate_audio function."""

    def test_valid_audio_passes(self, temp_file):
        """Tests a valid (non-empty) audio file."""
        # Create a file larger than 1KB to pass minimum size check
        audio_path = temp_file("test.mp3", size=2048)
        is_valid, message = validate_audio(audio_path)
        assert is_valid
        assert message == ""

    def test_file_not_found_fails(self):
        """Tests a non-existent audio file."""
        is_valid, message = validate_audio("nonexistent.mp3")
        assert not is_valid
        assert "not found" in message.lower()

    def test_empty_file_fails(self, temp_file):
        """Tests an empty audio file."""
        audio_path = temp_file("empty.mp3")
        is_valid, message = validate_audio(audio_path)
        assert not is_valid
        assert "empty" in message.lower()

    def test_very_small_file_fails(self, temp_file):
        """Tests a file that's too small (likely corrupted)."""
        audio_path = temp_file("tiny.mp3", size=100)
        is_valid, message = validate_audio(audio_path)
        assert not is_valid
        assert "small" in message.lower() or "corrupted" in message.lower()

    def test_invalid_format_fails(self, temp_file):
        """Tests an invalid audio format."""
        audio_path = temp_file("test.xyz", content="some content")
        is_valid, message = validate_audio(audio_path)
        assert not is_valid
        assert "format" in message.lower()

    def test_valid_formats(self, temp_file):
        """Tests various valid audio formats."""
        for ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg"]:
            audio_path = temp_file(f"test{ext}", size=2048)
            is_valid, _ = validate_audio(audio_path)
            assert is_valid, f"Format {ext} should be valid"


# =============================================================================
# Tests for validate_video
# =============================================================================

class TestValidateVideo:
    """Tests for the validate_video function."""

    def test_valid_video_passes(self, temp_file):
        """Tests a valid (non-empty) video file."""
        video_path = temp_file("test.mp4", size=10240)  # 10KB
        is_valid, message = validate_video(video_path)
        assert is_valid
        assert message == ""

    def test_file_not_found_fails(self):
        """Tests a non-existent video file."""
        is_valid, message = validate_video("nonexistent.mp4")
        assert not is_valid
        assert "not found" in message.lower()

    def test_empty_file_fails(self, temp_file):
        """Tests an empty video file."""
        video_path = temp_file("empty.mp4")
        is_valid, message = validate_video(video_path)
        assert not is_valid
        assert "empty" in message.lower()

    def test_file_too_large_fails(self, temp_file):
        """Tests a video file that exceeds size limit."""
        # Create a path but don't actually create huge file
        video_path = temp_file("large.mp4", size=1024)  # Small file
        # Override file size check by testing with small limit
        is_valid, message = validate_video(video_path, max_file_size_mb=0.0001)
        assert not is_valid
        assert "large" in message.lower()

    def test_invalid_format_fails(self, temp_file):
        """Tests an invalid video format."""
        video_path = temp_file("test.xyz", size=1024)
        is_valid, message = validate_video(video_path)
        assert not is_valid
        assert "format" in message.lower()

    def test_valid_formats(self, temp_file):
        """Tests various valid video formats."""
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            video_path = temp_file(f"test{ext}", size=2048)
            is_valid, _ = validate_video(video_path)
            assert is_valid, f"Format {ext} should be valid"


# =============================================================================
# Tests for sanitize_filename
# =============================================================================

class TestSanitizeFilename:
    """Tests for the sanitize_filename function."""

    def test_basic_sanitization(self):
        """Tests basic filename sanitization."""
        assert sanitize_filename("test file.txt") == "test_file.txt"

    def test_removes_invalid_chars(self):
        """Tests removal of invalid characters."""
        result = sanitize_filename("Game: Yankees vs Red Sox (2024)")
        assert ":" not in result
        assert "(" not in result
        assert ")" not in result
        assert result == "Game_Yankees_vs_Red_Sox_2024"

    def test_consecutive_underscores(self):
        """Tests that consecutive separators are collapsed."""
        result = sanitize_filename("test___file---name")
        assert "___" not in result
        assert "---" not in result

    def test_strips_edge_chars(self):
        """Tests stripping of leading/trailing special chars."""
        result = sanitize_filename("...test...")
        assert not result.startswith(".")
        assert not result.endswith(".")

    def test_max_length(self):
        """Tests maximum length enforcement."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_preserves_extension(self):
        """Tests that file extension is preserved when truncating."""
        long_name = "a" * 300 + ".mp4"
        result = sanitize_filename(long_name)
        assert result.endswith(".mp4")
        assert len(result) <= 255

    def test_empty_input(self):
        """Tests handling of empty input."""
        # Empty and whitespace-only strings return "unnamed" for safety
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"

    def test_non_string_input(self):
        """Tests handling of non-string input."""
        assert sanitize_filename(12345) == ""
        assert sanitize_filename(None) == ""


# =============================================================================
# Tests for validate_api_response
# =============================================================================

class TestValidateApiResponse:
    """Tests for the validate_api_response function."""

    def test_valid_response_passes(self):
        """Tests that valid response with all fields passes."""
        response = {
            "game": {
                "id": 123,
                "date": "2024-07-04"
            },
            "status": "Final"
        }
        is_valid, missing = validate_api_response(
            response,
            ["game.id", "game.date", "status"]
        )
        assert is_valid
        assert not missing

    def test_missing_field_fails(self):
        """Tests that missing field is detected."""
        response = {"game": {"id": 123}}
        is_valid, missing = validate_api_response(
            response,
            ["game.id", "game.date"]
        )
        assert not is_valid
        assert "game.date" in missing

    def test_nested_missing_field(self):
        """Tests detection of missing nested field."""
        response = {"data": {}}
        is_valid, missing = validate_api_response(
            response,
            ["data.games.count"]
        )
        assert not is_valid
        assert "data.games.count" in missing


# =============================================================================
# Tests for validate_date_range
# =============================================================================

class TestValidateDateRange:
    """Tests for the validate_date_range function."""

    def test_valid_range_passes(self):
        """Tests valid date range."""
        is_valid, message = validate_date_range("2024-01-01", "2024-01-31")
        assert is_valid
        assert message == ""

    def test_end_before_start_fails(self):
        """Tests that end date before start fails."""
        is_valid, message = validate_date_range("2024-12-31", "2024-01-01")
        assert not is_valid
        assert "after" in message.lower()

    def test_range_too_large_fails(self):
        """Tests that range exceeding max days fails."""
        is_valid, message = validate_date_range(
            "2024-01-01", "2025-06-01",
            max_days=100
        )
        assert not is_valid
        assert "large" in message.lower()

    def test_invalid_date_format_fails(self):
        """Tests that invalid date format fails."""
        is_valid, message = validate_date_range("01-01-2024", "01-31-2024")
        assert not is_valid
        assert "format" in message.lower()

    def test_same_day_valid(self):
        """Tests that same start and end date is valid."""
        is_valid, message = validate_date_range("2024-07-04", "2024-07-04")
        assert is_valid
