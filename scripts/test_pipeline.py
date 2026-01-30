#!/usr/bin/env python3
"""
MLB Video Pipeline - End-to-End Test Script

Tests all components of the pipeline to verify setup is correct.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --component data
    python scripts/test_pipeline.py --skip-api
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.utils.logger import setup_logging


console = Console()


def test_config() -> tuple[bool, str]:
    """Test configuration loading."""
    try:
        from config.settings import Settings, get_settings
        from config.league_config import MLB_TEAMS

        s = get_settings()
        assert s.base_dir.exists(), "Base directory doesn't exist"
        assert len(MLB_TEAMS) == 30, f"Expected 30 teams, got {len(MLB_TEAMS)}"

        return True, "Configuration loaded successfully"
    except Exception as e:
        return False, str(e)


def test_directories() -> tuple[bool, str]:
    """Test directory structure."""
    try:
        settings.ensure_directories()

        required_dirs = [
            settings.data_dir,
            settings.models_dir,
            settings.outputs_dir,
            settings.logs_dir,
        ]

        missing = [d for d in required_dirs if not d.exists()]
        if missing:
            return False, f"Missing directories: {missing}"

        return True, "All directories exist"
    except Exception as e:
        return False, str(e)


def test_data_fetcher() -> tuple[bool, str]:
    """Test MLB data fetcher."""
    try:
        from src.data.fetcher import MLBDataFetcher

        fetcher = MLBDataFetcher()

        # Test health check
        if not fetcher.health_check():
            return False, "MLB API not accessible"

        # Test fetching games
        games = fetcher.get_games_for_date("2024-07-04")
        if not isinstance(games, list):
            return False, "get_games_for_date didn't return a list"

        return True, f"Fetched {len(games)} games for 2024-07-04"
    except Exception as e:
        return False, str(e)


def test_data_processor() -> tuple[bool, str]:
    """Test data processor."""
    try:
        from src.data.processor import DataProcessor

        processor = DataProcessor()

        # Test with sample data
        sample_games = [
            {
                "game_id": 1,
                "game_date": "2024-07-04",
                "home_team_id": 147,
                "away_team_id": 111,
                "home_score": 5,
                "away_score": 3,
                "status": "Final",
            }
        ]

        df = processor.process_games(sample_games)
        assert len(df) == 1, "Expected 1 processed game"

        return True, "Data processor working correctly"
    except Exception as e:
        return False, str(e)


def test_model() -> tuple[bool, str]:
    """Test prediction model."""
    try:
        import torch
        from src.models.predictor import PlayerPredictor

        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        # Test forward pass
        x = torch.randn(5, 10)
        y = model(x)

        assert y.shape == (5, 4), f"Wrong output shape: {y.shape}"

        return True, "Model forward pass successful"
    except Exception as e:
        return False, str(e)


def test_script_generator(skip_api: bool = True) -> tuple[bool, str]:
    """Test script generator."""
    try:
        from src.content.script_generator import ScriptGenerator
        from src.content.prompts import format_prompt, PROMPTS

        # Test prompt formatting
        assert len(PROMPTS) >= 4, "Missing prompt templates"

        if skip_api:
            return True, "Prompt templates loaded (API call skipped)"

        # Test API call
        gen = ScriptGenerator()
        script = gen.generate_game_recap({
            "game_id": 1,
            "game_date": "2024-07-04",
            "home_team_id": 147,
            "away_team_id": 111,
            "home_score": 5,
            "away_score": 3,
            "key_stats": {},
            "notable_performances": "Test",
        }, duration=30)

        return True, f"Generated script: {len(script)} chars"
    except ValueError as e:
        if "API key" in str(e):
            return True, "Script generator initialized (no API key)"
        return False, str(e)
    except Exception as e:
        return False, str(e)


def test_tts_engine(skip_api: bool = True) -> tuple[bool, str]:
    """Test TTS engine."""
    try:
        from src.audio.tts_engine import TTSEngine

        if skip_api:
            return True, "TTS module imported (API call skipped)"

        tts = TTSEngine()
        duration = tts.estimate_duration("This is a test script for the TTS engine.")

        return True, f"TTS initialized, estimated duration: {duration:.1f}s"
    except ValueError as e:
        if "API key" in str(e):
            return True, "TTS module works (no API key)"
        return False, str(e)
    except Exception as e:
        return False, str(e)


def test_video_generator() -> tuple[bool, str]:
    """Test video generator."""
    try:
        from src.video.generator import VideoGenerator
        from src.video.templates import TEMPLATES

        assert len(TEMPLATES) >= 3, "Missing video templates"

        gen = VideoGenerator(template="modern_dark")

        # Test frame generation
        frame = gen.create_intro_frame(
            title="Test Video",
            subtitle="Testing",
            date="2024-07-04"
        )

        assert frame.size == (gen.width, gen.height), "Wrong frame size"

        return True, f"Generated test frame: {frame.size}"
    except Exception as e:
        return False, str(e)


def test_youtube_uploader() -> tuple[bool, str]:
    """Test YouTube uploader."""
    try:
        from src.upload.youtube import YouTubeUploader

        uploader = YouTubeUploader()

        # Just test initialization
        return True, "YouTube uploader initialized"
    except Exception as e:
        return False, str(e)


def test_validators() -> tuple[bool, str]:
    """Test validators."""
    try:
        from src.utils.validators import (
            validate_date,
            validate_team_id,
            validate_game_data,
        )

        assert validate_date("2024-07-04"), "Valid date failed"
        assert not validate_date("invalid"), "Invalid date passed"
        assert validate_team_id(147), "Valid team ID failed"
        assert not validate_team_id(999), "Invalid team ID passed"

        return True, "All validators working"
    except Exception as e:
        return False, str(e)


def test_logger() -> tuple[bool, str]:
    """Test logging setup."""
    try:
        from src.utils.logger import get_logger, CostTracker

        logger = get_logger("test")
        logger.info("Test log message")

        tracker = CostTracker()
        assert tracker.check_budget(100), "Budget check failed"

        return True, "Logging system working"
    except Exception as e:
        return False, str(e)


@click.command()
@click.option(
    "--component", "-c",
    type=click.Choice([
        "config", "directories", "data", "processor",
        "model", "script", "tts", "video", "youtube",
        "validators", "logger"
    ]),
    help="Test specific component only"
)
@click.option(
    "--skip-api",
    is_flag=True,
    default=True,
    help="Skip tests that require API calls"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def main(component: str | None, skip_api: bool, verbose: bool):
    """Run pipeline tests."""

    setup_logging(level="DEBUG" if verbose else "WARNING")

    console.print("[bold blue]MLB Video Pipeline - Test Suite[/bold blue]")
    console.print()

    tests = {
        "config": ("Configuration", test_config),
        "directories": ("Directories", test_directories),
        "data": ("Data Fetcher", test_data_fetcher),
        "processor": ("Data Processor", test_data_processor),
        "model": ("Prediction Model", test_model),
        "script": ("Script Generator", lambda: test_script_generator(skip_api)),
        "tts": ("TTS Engine", lambda: test_tts_engine(skip_api)),
        "video": ("Video Generator", test_video_generator),
        "youtube": ("YouTube Uploader", test_youtube_uploader),
        "validators": ("Validators", test_validators),
        "logger": ("Logger", test_logger),
    }

    # Filter to specific component if requested
    if component:
        tests = {component: tests[component]}

    # Run tests
    results = []
    for key, (name, test_func) in tests.items():
        try:
            passed, message = test_func()
            results.append((name, passed, message))
        except Exception as e:
            results.append((name, False, str(e)))

    # Display results
    table = Table(title="Test Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    passed_count = 0
    for name, passed, message in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        if passed:
            passed_count += 1
        table.add_row(name, status, message)

    console.print(table)
    console.print()

    # Summary
    total = len(results)
    if passed_count == total:
        console.print(f"[bold green]All {total} tests passed![/bold green]")
    else:
        console.print(f"[bold yellow]{passed_count}/{total} tests passed[/bold yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
