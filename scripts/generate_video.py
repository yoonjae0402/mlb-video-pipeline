#!/usr/bin/env python3
"""
MLB Video Pipeline - Generate Video Script

CLI tool to generate a complete video from game data.

Usage:
    python scripts/generate_video.py --game-id 12345
    python scripts/generate_video.py --date 2024-07-04 --team NYY
    python scripts/generate_video.py --type player_spotlight --player "Aaron Judge"
"""

import sys
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.league_config import get_team_by_abbreviation
from src.data.fetcher import MLBDataFetcher
from src.content.script_generator import ScriptGenerator
from src.audio.tts_engine import TTSEngine
from src.video.generator import VideoGenerator
from src.utils.logger import setup_logging, get_logger


console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--game-id", "-g",
    type=int,
    help="MLB game ID to create video for"
)
@click.option(
    "--date", "-d",
    type=str,
    help="Date (YYYY-MM-DD) - picks first completed game"
)
@click.option(
    "--team", "-t",
    type=str,
    help="Team abbreviation (use with --date)"
)
@click.option(
    "--type", "-T",
    type=click.Choice(["game_recap", "player_spotlight", "prediction"]),
    default="game_recap",
    help="Type of video to generate"
)
@click.option(
    "--duration",
    type=int,
    default=60,
    help="Target video duration in seconds"
)
@click.option(
    "--template",
    type=str,
    default="modern_dark",
    help="Video template name"
)
@click.option(
    "--skip-audio",
    is_flag=True,
    help="Skip audio generation (use placeholder)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Don't call external APIs"
)
@click.option(
    "--output", "-o",
    type=str,
    help="Output filename"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def main(
    game_id: int | None,
    date: str | None,
    team: str | None,
    type: str,
    duration: int,
    template: str,
    skip_audio: bool,
    dry_run: bool,
    output: str | None,
    verbose: bool,
):
    """Generate a complete video from MLB game data."""

    setup_logging(level="DEBUG" if verbose else "INFO")

    # Set dry run mode
    if dry_run:
        settings.dry_run = True

    console.print("[bold blue]MLB Video Pipeline - Video Generator[/bold blue]")
    console.print()

    # Validate inputs
    if not game_id and not date:
        console.print("[yellow]No game specified, using yesterday's games...[/yellow]")
        from datetime import timedelta
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Step 1: Fetch game data
        task = progress.add_task("Fetching game data...", total=None)

        fetcher = MLBDataFetcher()

        if game_id:
            games = [{"game_id": game_id}]  # Will be populated with details
            game_data = fetcher.get_game_details(game_id)
        else:
            games = fetcher.get_games_for_date(date)

            # Filter by team if specified
            if team:
                result = get_team_by_abbreviation(team)
                if result:
                    team_id, _ = result
                    games = [
                        g for g in games
                        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id
                    ]

            # Pick first completed game
            completed = [g for g in games if g.get("status") == "Final"]
            if completed:
                game_data = completed[0]
            elif games:
                game_data = games[0]
            else:
                console.print("[red]No games found[/red]")
                sys.exit(1)

        progress.update(task, description="[green]Game data fetched!")
        console.print(f"  Game: {game_data.get('away_team_name', 'Away')} @ {game_data.get('home_team_name', 'Home')}")

        # Step 2: Generate script
        task = progress.add_task("Generating script...", total=None)

        # Prepare game data for script
        script_data = {
            "game_id": game_data.get("game_id"),
            "game_date": game_data.get("game_date", date),
            "home_team_id": game_data.get("home_team_id"),
            "away_team_id": game_data.get("away_team_id"),
            "home_score": game_data.get("home_score", 0),
            "away_score": game_data.get("away_score", 0),
            "venue": game_data.get("venue", ""),
            "home_record": "0-0",  # Would come from standings
            "away_record": "0-0",
            "key_stats": {},
            "notable_performances": "Key performances from the game.",
        }

        if dry_run:
            script = f"[DRY RUN] This is a placeholder script for the {type} video."
        else:
            try:
                generator = ScriptGenerator()
                script = generator.generate_game_recap(script_data, duration=duration)
            except ValueError as e:
                console.print(f"[yellow]Skipping script generation: {e}[/yellow]")
                script = f"Game Recap: {script_data.get('away_team_id')} at {script_data.get('home_team_id')}"

        progress.update(task, description="[green]Script generated!")
        console.print(f"  Script length: {len(script)} characters")

        # Step 3: Generate audio
        audio_path = None
        if not skip_audio:
            task = progress.add_task("Generating narration...", total=None)

            if dry_run:
                console.print("  [yellow]Dry run: skipping audio[/yellow]")
            else:
                try:
                    tts = TTSEngine()
                    audio_path = tts.generate_narration(script, output_name=f"narration_{game_data.get('game_id', 'unknown')}")
                    progress.update(task, description="[green]Narration generated!")
                    console.print(f"  Audio: {audio_path}")
                except ValueError as e:
                    console.print(f"  [yellow]Skipping audio: {e}[/yellow]")

        # Step 4: Generate video
        task = progress.add_task("Creating video...", total=None)

        video_gen = VideoGenerator(template=template)

        # Create scenes based on content type
        scenes = [
            {
                "type": "intro",
                "title": "GAME RECAP",
                "subtitle": f"{game_data.get('away_team_name', 'Away')} @ {game_data.get('home_team_name', 'Home')}",
                "date": str(game_data.get("game_date", date)),
                "duration": 4.0,
            },
            {
                "type": "stats",
                "title": "FINAL SCORE",
                "data": [
                    {"label": game_data.get("away_team_name", "Away"), "value": game_data.get("away_score", 0)},
                    {"label": game_data.get("home_team_name", "Home"), "value": game_data.get("home_score", 0)},
                ],
                "duration": 5.0,
            },
            {
                "type": "intro",
                "title": "Thanks for watching!",
                "subtitle": "Like & Subscribe",
                "duration": 3.0,
            },
        ]

        output_name = output or f"recap_{game_data.get('game_id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}"

        try:
            video_path = video_gen.create_video(
                scenes=scenes,
                audio_path=audio_path,
                output_name=output_name,
            )
            progress.update(task, description="[green]Video created!")
        except ImportError:
            console.print("[yellow]MoviePy not available, skipping video creation[/yellow]")
            video_path = None

    # Summary
    console.print()
    console.print("[bold green]Pipeline Complete![/bold green]")
    console.print()

    if video_path:
        console.print(f"  Video: {video_path}")
    if audio_path:
        console.print(f"  Audio: {audio_path}")

    # Cost summary
    if not dry_run:
        console.print()
        console.print("[dim]Check logs/costs.json for API cost tracking[/dim]")


if __name__ == "__main__":
    main()
